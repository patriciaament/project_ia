# agent.py
# -*- coding: utf-8 -*-

import os
import re
from typing import Optional, Dict, Any, List

from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import create_sql_query_chain


# ======================================================
# CONFIG GERAL
# ======================================================

DB_URI = "sqlite:///db/base.db"

# SKU: começa com letra, depois 3-8 dígitos, tipo A7171
_SKU_RX = re.compile(r"\b([A-Z]\d{3,8})\b", re.I)

_STOP_TOKENS = [
    " em ", " no ", " na ", " de ", " do ", " da ",
    " para ", " por ", " com ", " que ", " e ",
    " ou ", " onde ", " quando "
]
_STOP_PUNCT = r"[\,\.\?\:\;\!\|/()\[\]\n\r\t]"


# ======================================================
# FUNÇÕES UTILITÁRIAS
# ======================================================

def _extract_sku_and_client(prompt: str):
    """
    tenta achar um SKU e o nome do cliente a partir da pergunta em linguagem natural.
    ex: "o SKU A7171 no cliente Mundo da Criança..."
    retorna (sku, cliente) ou (None, None)
    """
    sku = None
    m = _SKU_RX.search(prompt or "")
    if m:
        sku = m.group(1).upper()

    cliente = None
    p = prompt or ""
    p_low = p.lower()

    idx = p_low.find("cliente")
    if idx >= 0:
        rest = p[idx + len("cliente"):].lstrip()

        # caso tenha cliente "Nome do Cliente"
        if rest.startswith('"'):
            m = re.search(r'^"([^"]+)"', rest)
            if m:
                cliente = m.group(1).strip()

        # caso tenha cliente 'Nome do Cliente'
        if not cliente and rest.startswith("'"):
            m = re.search(r"^'([^']+)'", rest)
            if m:
                cliente = m.group(1).strip()

        # caso esteja sem aspas, tipo "cliente Mundo da Criança qual é..."
        if not cliente:
            m = re.search(_STOP_PUNCT, rest)
            cut = rest[:m.start()] if m else rest
            cut_low = " " + cut.lower() + " "
            min_pos = None
            for tok in _STOP_TOKENS:
                pos = cut_low.find(tok)
                if pos != -1:
                    pos = pos - 1
                    if min_pos is None or pos < min_pos:
                        min_pos = pos
            if min_pos is not None:
                cut = cut[:min_pos]
            cliente = cut.strip(" :.-").strip()

        # limitar pra não capturar frase gigante
        if cliente:
            parts = cliente.split()
            if len(parts) > 6:
                cliente = " ".join(parts[:6]).strip()

    return sku, cliente


def _fmt_decimal_brl(x: float, casas=2) -> str:
    """
    formata número como "123,45".
    se não der pra converter pra float, volta string crua.
    """
    try:
        s = f"{float(x):.{casas}f}"
    except Exception:
        return str(x)
    return s.replace(".", ",")


def _only_sql(text: str) -> str:
    """
    extrai só o SELECT de uma saída que pode vir com explicação.
    """
    if not text:
        return ""
    txt = text.strip().strip("`").strip()
    if txt.lower().startswith("select"):
        return txt

    # tenta achar um trecho que começa com select
    m = re.search(r"(?is)\bselect\b.+", txt, re.DOTALL)
    if m:
        return m.group(0).strip()

    return txt


def _summarize_result(pergunta: str, rows: Any) -> str:
    """
    gera uma resposta humana baseada nas linhas retornadas da query.
    rows pode ser:
      - lista de dict (normal)
      - {"_error": "..."} se falhou
    """
    # erro na execução
    if isinstance(rows, dict) and "_error" in rows:
        return (
            "Tentei consultar os dados para sua pergunta, mas houve um problema ao "
            "executar a query no banco. Ainda assim, você pode visualizar a SQL gerada "
            "pra entender o que foi pedido."
        )

    # nenhuma linha
    if not rows:
        return "Não encontrei dados relevantes pra essa pergunta no banco."

    # 1 linha -> podemos tentar dar um resumo chave:valor
    if isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict):
        row = rows[0]
        partes = []
        for k, v in row.items():
            partes.append(f"{k}: {v}")
        detalhe = "; ".join(partes)
        return (
            f"Encontrei 1 registro relacionado à sua pergunta. "
            f"Principais dados: {detalhe}."
        )

    # muitas linhas -> fala nível macro
    if isinstance(rows, list):
        cols = []
        if len(rows) > 0 and isinstance(rows[0], dict):
            cols = list(rows[0].keys())
        return (
            f"Encontrei {len(rows)} linhas que respondem à pergunta. "
            f"As colunas principais recuperadas foram: {', '.join(cols)}. "
            f"Se quiser ver mais detalhado (por exemplo top 5 linhas), "
            f"posso montar outra consulta."
        )

    # fallback genérico
    return "Consulta concluída."


# ======================================================
# get_agent (principal)
# ======================================================

def get_agent(open_api_key: Optional[str] = None):
    """
    retorna a função run_query(prompt: str) que o frontend (app.py) chama.
    essa função interna faz:
      - roteia pro agente certo
      - gera/roda SQL
      - formata resposta didática
      - trata casos especiais (SKU+cliente)
    """

    # 1. valida chave
    api_key = open_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Faltou a OPENAI_API_KEY no ambiente (.env ou secrets).")

    # 2. instância 'global'
    db_main = SQLDatabase.from_uri(DB_URI)
    llm_main = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key,
    )

    # chain de fallback genérico
    query_chain = create_sql_query_chain(llm_main, db_main, k=4)

    # função utilitária pra rodar SQL com try/except
    def _run_sql_safe(sql: str):
        try:
            return db_main.run(sql)
        except Exception as e:
            return {"_error": str(e), "_sql": sql}

    # cria um agente SQL limitado a certas tabelas
    def _make_sql_agent(tables: List[str]):
        sub_db = SQLDatabase.from_uri(DB_URI, include_tables=tables)

        toolkit = SQLDatabaseToolkit(db=sub_db, llm=llm_main)

        BASE_CONTEXT = f"""
        Você é um gerador/validador de SQL focado em SQLite.
        Gere APENAS SELECTs válidos para as tabelas:
        {', '.join(tables)}.

        Regras importantes:
        - Não invente tabelas nem colunas que não existem.
        - Não use crases nem backticks.
        - Não explique a consulta no mesmo texto: apenas gere o SELECT.
        """

        memory = ConversationBufferWindowMemory(
            k=5,
            memory_key="chat_history",
            return_messages=True
        )

        return create_sql_agent(
            llm=llm_main,
            toolkit=toolkit,
            verbose=False,
            handle_parsing_errors=True,
            prefix=BASE_CONTEXT,
            memory=memory,
            max_iterations=50,
            max_execution_time=300,
        )

    # agentes especializados (cada um com escopo reduzido → menos erro e menos custo)
    agent_summary    = _make_sql_agent(["summary_country"])
    agent_posweek    = _make_sql_agent(["pos_week"])
    agent_status     = _make_sql_agent(["status_sku"])
    agent_relweek    = _make_sql_agent(["relatorio_week"])
    agent_item       = _make_sql_agent(["classificacao_items", "item_master"])
    agent_clientes   = _make_sql_agent(["classificacao_clientes"])
    # fallback que enxerga mais de uma tabela (quando nenhuma regra bate claramente)
    agent_misto      = _make_sql_agent([
        "summary_country",
        "pos_week",
        "status_sku",
        "relatorio_week",
        "classificacao_items",
        "classificacao_clientes",
        "item_master"
    ])

    # roteador: escolhe qual agente usar com base no texto
    def _route_agent(prompt: str) -> Any:
        p = prompt.lower()

        # vendas / pos / últimas semanas
        if any(x in p for x in ["pos", "semana", "lw", "últimas", "ultimas", "4 semanas", "ytd"]):
            return agent_posweek

        # status TLP / NTLP, classificação de SKU
        if any(x in p for x in ["tlp", "ntlp", "status", "classificação", "classificacao", "sku"]):
            return agent_status

        # estoque / retail price / preço sugerido
        if any(x in p for x in ["estoque", "ohi", "retail", "preço", "preco", "sugerido"]):
            return agent_relweek

        # descrição de item, marca, nível (Level_2, Level_3 etc)
        if any(x in p for x in ["descrição", "descricao", "marca", "level", "item", "sku", "produto"]):
            return agent_item

        # análises mais macro / país / bases consolidadas
        if any(x in p for x in ["resumo", "country", "mundo", "visão geral", "visao geral"]):
            return agent_summary

        # se falar explicitamente de cliente / canal
        if any(x in p for x in ["cliente", "canal", "rede"]):
            return agent_clientes

        # fallback
        return agent_misto

    # --------------------------------------------------
    # formatação final amigável pro usuário
    # --------------------------------------------------
    def _monta_texto_sku_cliente_full(sku: str, cliente: str, row: Dict[str, Any]) -> str:
        status_val = (row.get("status") or "").upper()
        classe = "TLP" if "TLP" in status_val else "NTLP"

        retail_val = row.get("retail")
        retail_fmt = _fmt_decimal_brl(retail_val, 2)

        ohi_cy  = row.get("ohi_cy")
        ohi_var = row.get("ohi_var")

        return (
            f"Para o SKU {sku} no cliente {cliente}: "
            f"ele está classificado como {classe}. "
            f"O preço sugerido de venda (Retail Price) é {retail_fmt}. "
            f"O estoque atual (OHI CY) registrado é {ohi_cy}, "
            f"com variação percentual de {ohi_var}."
        )

    def _monta_texto_sku_cliente_fallback(sku: str, cliente: str, row: Dict[str, Any]) -> str:
        status_val = (row.get("status") or "").upper()
        classe = "TLP" if "TLP" in status_val else "NTLP"

        retail_val = row.get("retail")
        retail_fmt = _fmt_decimal_brl(retail_val, 2)

        return (
            f"Para o SKU {sku}: ele está classificado como {classe} "
            f"e o preço sugerido de venda (Retail Price) é {retail_fmt}. "
            f"Não consegui confirmar estoque/variação específicos no cliente {cliente}."
        )

    # --------------------------------------------------
    # A FUNÇÃO QUE O FRONT VAI CHAMAR
    # --------------------------------------------------
    def run_query(prompt: str) -> Dict[str, Any]:
        """
        essa é a função que o app.py usa.
        retorna dict com:
          - output (resposta didática em PT-BR)
          - sql (a consulta gerada / executada)
          - rows (amostra de linhas do banco ou erro)
        """

        # ==================================================
        # CASO ESPECIAL: pergunta tipo "SKU ABC123 no cliente XYZ"
        # ==================================================
        sku, cliente = _extract_sku_and_client(prompt)
        if sku and cliente:
            # tentativa completa: SKU + cliente (estoque / variação / classe TLP/NTLP / retail)
            sql_full = f'''
SELECT
  s."OHI CY"                  AS ohi_cy,
  s."OHI Var%"                AS ohi_var,
  st."Status POS Master 2025" AS status,
  rw."RETAIL"                 AS retail
FROM summary_country s
LEFT JOIN status_sku         st ON st."SKU" = s."Item"
LEFT JOIN relatorio_week     rw ON rw."SKU" = s."Item"
LEFT JOIN classificacao_clientes cc
    ON cc."Nome Fictício" = s."Client DC Group"
WHERE s."Item" = '{sku}'
  AND (
    cc."Nome Fictício" = '{cliente}'
    OR s."Client DC Group" = '{cliente}'
  )
LIMIT 1;
'''.strip()

            rows_full = _run_sql_safe(sql_full)

            # se não deu erro e retornou dado
            if not (isinstance(rows_full, dict) and "_error" in rows_full):
                if rows_full:
                    row = rows_full[0]
                    texto = _monta_texto_sku_cliente_full(sku, cliente, row)
                    return {
                        "output": texto,
                        "sql": sql_full,
                        "rows": rows_full,
                    }

            # fallback: ignora cliente, pega só classe TLP/NTLP + retail
            sql_fb = f'''
SELECT
  st."Status POS Master 2025" AS status,
  rw."RETAIL"                 AS retail
FROM status_sku st
LEFT JOIN relatorio_week rw
    ON rw."SKU" = st."SKU"
WHERE st."SKU" = '{sku}'
LIMIT 1;
'''.strip()

            rows_fb = _run_sql_safe(sql_fb)

            # fallback funcionou
            if not (isinstance(rows_fb, dict) and "_error" in rows_fb):
                if rows_fb:
                    row = rows_fb[0]
                    texto = _monta_texto_sku_cliente_fallback(sku, cliente, row)
                    return {
                        "output": texto,
                        "sql": sql_fb,
                        "rows": rows_fb,
                    }

            # nada funcionou → mensagem clean
            texto = (
                f"Tentei consultar o SKU {sku} no cliente {cliente}, "
                f"mas não consegui recuperar esses dados agora."
            )
            return {
                "output": texto,
                "sql": sql_full,
                "rows": rows_fb,
            }

        # ==================================================
        # CASO GERAL (roteador de agente)
        # ==================================================
        agent = _route_agent(prompt)

        # 1) tenta o agente especializado gerar SQL
        try:
            agent_res = agent.invoke({"input": prompt})
            raw_out = agent_res.get("output", "")
        except Exception:
            # se der ruim, pede pro chain de fallback gerar uma SQL
            chain_raw = query_chain.invoke({
                "question": f"Gere apenas a consulta SQL para: {prompt}"
            })
            raw_out = str(chain_raw)

        # 2) isola SELECT
        sql_candidate = _only_sql(raw_out).strip()

        # se o LLM não devolveu um SELECT de verdade, manda texto direto
        if not sql_candidate.lower().startswith("select"):
            texto_final = raw_out.strip() if raw_out else "Não consegui gerar resposta estruturada."
            return {
                "output": texto_final,
                "sql": None,
                "rows": [],
            }

        # 3) roda a query construída
        rows_sample = _run_sql_safe(sql_candidate)

        # 4) gera resumo humano
        texto_final = _summarize_result(prompt, rows_sample)

        return {
            "output": texto_final,
            "sql": sql_candidate,
            "rows": rows_sample,
        }

    # <-- FIM run_query()

    return run_query
