# agent.py
# -*- coding: utf-8 -*-
import re
from typing import Optional, Dict, Any, List, Union
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import create_sql_query_chain

# ============================
# CONFIG
# ============================

DB_URI = "sqlite:///db/base.db"

# regex pra tentar capturar SKU tipo A1234 etc
_SKU_RX = re.compile(r"\b([A-Z]\d{3,8})\b", re.I)

_STOP_TOKENS = [
    " em ", " no ", " na ", " de ", " do ", " da ",
    " para ", " por ", " com ", " que ", " e ",
    " ou ", " onde ", " quando "
]
_STOP_PUNCT = r"[\,\.\?\:\;\!\|/()\[\]\n\r\t]"

# ============================
# HELPERS
# ============================

def _extract_sku_and_client(prompt: str):
    """
    Tenta pegar um SKU e um cliente a partir do texto livre.
    SKU: coisa tipo 'A7171'
    Cliente: trecho após a palavra 'cliente'
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
        rest = p[idx + len("cliente") :].lstrip()

        # caso esteja entre aspas "cliente X"
        if rest.startswith('"'):
            m = re.search(r'^"([^"]+)"', rest)
            if m:
                cliente = m.group(1).strip()

        # caso esteja entre aspas simples 'cliente X'
        if not cliente and rest.startswith("'"):
            m = re.search(r"^'([^']+)'", rest)
            if m:
                cliente = m.group(1).strip()

        # caso livre até primeira pontuação / stop word
        if not cliente:
            m = re.search(_STOP_PUNCT, rest)
            cut = rest[: m.start()] if m else rest
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

            # tira lixo tipo ":", "-" no fim
            cliente = cut.strip(" :.-").strip()

        # hard cap pra não capturar frase gigante
        if cliente:
            parts = cliente.split()
            if len(parts) > 6:
                cliente = " ".join(parts[:6]).strip()

    return sku, cliente


def _fmt_decimal_brl(x: float, casas: int = 2) -> str:
    """
    Formata número float tipo 1234.5 -> '1234,50'
    """
    try:
        s = f"{float(x):.{casas}f}"
    except Exception:
        return str(x)
    return s.replace(".", ",")


def _make_sql_agent(tables: List[str], openai_key: str):
    """
    Cria um agente SQL especializado só em certas tabelas.
    """
    db = SQLDatabase.from_uri(DB_URI, include_tables=tables)

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=openai_key,
    )

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    BASE_CONTEXT = f"""
Você é um especialista SQL focado em tabelas específicas.
Gere apenas SELECTs válidos para SQLite.
NUNCA inclua texto junto do SQL quando estiver retornando para execução.
Retorne apenas a query.
Tabelas disponíveis: {', '.join(tables)}.
Respeite nomes e capitalização das colunas exatamente.
"""

    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True
    )

    return create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=False,
        handle_parsing_errors=True,
        prefix=BASE_CONTEXT,
        memory=memory,
        max_iterations=3,
        max_execution_time=25,
    )


def _route_agent(
    prompt: str,
    agent_posweek,
    agent_status,
    agent_relweek,
    agent_item,
    agent_summary,
    agent_misto,
):
    """
    Escolhe qual agente usar com base em palavras-chave da pergunta.
    """
    p = prompt.lower()

    # foco em sell-out, vendas, POS, semanas
    if any(x in p for x in ["venda", "semana", "pos", "lw", "ytd"]):
        return agent_posweek

    # foco em classificação TLP / NTLP
    if any(x in p for x in ["tlp", "ntlp", "status"]):
        return agent_status

    # foco em preço sugerido / retail / estoque
    if any(x in p for x in ["estoque", "ohi", "variação", "retail", "preço", "price"]):
        return agent_relweek

    # foco em atributos de SKU, marca, descrição
    if any(x in p for x in ["marca", "descrição", "description", "level", "sku", "item"]):
        return agent_item

    # foco em visão mais macro/cliente
    if any(x in p for x in ["resumo", "country", "mundo"]):
        return agent_summary

    # fallback
    return agent_misto


def get_agent(open_api_key: Optional[str] = None):
    """
    Isso aqui devolve uma função run_query(prompt) que:
      1. Entende a pergunta
      2. Gera SQL usando o agente certo
      3. Executa a SQL de verdade no SQLite
      4. Resume em PT-BR com linguagem de negócio
      5. Retorna:
         - output (texto amigável)
         - sql    (a query usada)
         - rows   (amostra de linhas)
    """

    if not open_api_key:
        raise ValueError("OPENAI_API_KEY não informado para get_agent().")

    # db raiz (todas as tabelas) pra executar queries
    db = SQLDatabase.from_uri(DB_URI)

    # LLM auxiliar genérico pra fallback e pra sumarizar resultado
    llm_fallback = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=open_api_key,
    )

    # chain simples pra gerar SQL crua se o agente falhar
    query_chain = create_sql_query_chain(llm_fallback, db, k=3)

    # criamos todos os agentes especializados
    agent_summary   = _make_sql_agent(["summary_country"], open_api_key)
    agent_posweek   = _make_sql_agent(["pos_week"], open_api_key)
    agent_item      = _make_sql_agent(["classificacao_items"], open_api_key)  # se sua tabela for classificacao_clientes ou outra, ajuste aqui
    agent_status    = _make_sql_agent(["status_sku"], open_api_key)
    agent_relweek   = _make_sql_agent(["relatorio_week"], open_api_key)
    agent_clientes  = _make_sql_agent(["classificacao_clientes"], open_api_key)
    agent_misto     = _make_sql_agent(
        ["summary_country", "classificacao_items", "status_sku", "relatorio_week", "pos_week", "classificacao_clientes"],
        open_api_key
    )

    SQL_ONLY_REGEX = re.compile(r"(?is)\bselect\b.+", re.DOTALL)

    def _only_sql(text: str) -> str:
        """
        Extrai apenas o SELECT puro de uma resposta confusa.
        """
        raw = (text or "").strip().strip("`").strip()
        if raw.lower().startswith("select"):
            return raw
        m = SQL_ONLY_REGEX.search(raw)
        if m:
            return m.group(0).strip()
        return raw

    def _run_sql_safe(sql: str) -> Union[List[dict], dict]:
        """
        Executa a query no SQLite via SQLDatabase.run
        e retorna até 5 linhas (em forma de lista de dict).
        Se der erro, retorna {"_error": "..."}.
        """
        try:
            rows = db.run(sql)

            normalized = []
            for r in rows:
                if isinstance(r, dict):
                    normalized.append(r)
                else:
                    # tenta converter Row -> dict
                    try:
                        normalized.append(dict(r))
                    except Exception:
                        normalized.append({"value": str(r)})

            return normalized[:5]

        except Exception as e:
            return {"_error": str(e)}

    def _summarize_result(user_prompt: str, rows_sample):
        """
        Usa o llm_fallback pra transformar os dados em uma explicação
        clara e curta pro negócio.
        """
        # Erro na execução
        if isinstance(rows_sample, dict) and "_error" in rows_sample:
            return (
                "Gerei a consulta SQL, mas houve erro ao executar no banco:\n"
                f"{rows_sample['_error']}\n\n"
                "Segue a SQL técnica para análise."
            )

        # Sem linhas
        if not rows_sample:
            return "Consulta executada, mas não retornou resultados."

        # Monta preview de até 5 linhas pra resumir
        preview = ""
        for i, row in enumerate(rows_sample):
            preview += f"Linha {i+1}: {row}\n"

        sys_prompt = (
            "Você é um analista de dados. Eu vou te dar a pergunta do usuário e "
            "uma amostra (até 5 linhas) dos resultados da consulta ao banco. "
            "Explique em português claro e direto o que esses dados significam "
            "pro negócio. Se houver números percentuais, explique tendência. "
            "Não invente colunas que não existem."
        )

        user_block = (
            f"Pergunta do usuário:\n{user_prompt}\n\n"
            f"Amostra de resultados:\n{preview}\n\n"
            "Explique de forma executiva:"
        )

        try:
            resp = llm_fallback.invoke([
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_block},
            ])
            if hasattr(resp, "content"):
                return resp.content.strip()
            return str(resp)
        except Exception:
            # fallback bruto
            return "Resumo dos dados retornados:\n" + preview

    def run_query(prompt: str) -> Dict[str, Any]:
        """
        Pipeline:
        - Caso especial (SKU + cliente) -> responde estoque/preço/etc direto
        - Caso geral -> escolhe agente -> gera SQL -> executa -> resume
        """
        # ----- 1. Checa caso especial SKU + cliente (estoque / retail / TLP/NTLP)
        sku, cliente = _extract_sku_and_client(prompt)
        if sku and cliente:
            sql = f'''
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

            rows_sample = _run_sql_safe(sql)

            # erro ao executar
            if isinstance(rows_sample, dict) and "_error" in rows_sample:
                return {
                    "output": (
                        "Tentei consultar estoque / classificação / preço sugerido, "
                        "mas houve erro ao executar a query.\n"
                        f"Erro: {rows_sample['_error']}"
                    ),
                    "sql": sql,
                    "rows": rows_sample,
                }

            # nada encontrado
            if not rows_sample:
                return {
                    "output": "Não encontrei esse SKU/cliente na base.",
                    "sql": sql,
                    "rows": [],
                }

            # monta texto amigável manual
            row = rows_sample[0]
            status_val = (row.get("status") or "").upper()
            classe = "TLP" if "TLP" in status_val else "NTLP"

            retail_val = row.get("retail")
            retail_fmt = _fmt_decimal_brl(retail_val, 2)

            ohi_cy  = row.get("ohi_cy")
            ohi_var = row.get("ohi_var")

            texto = (
                f"Para o SKU {sku} no cliente {cliente}: "
                f"ele está classificado como {classe}. "
                f"O preço sugerido de venda (Retail Price) é {retail_fmt}. "
                f"O estoque atual (OHI CY) é {ohi_cy}, "
                f"com variação percentual de {ohi_var}."
            )

            return {
                "output": texto,
                "sql": sql,
                "rows": rows_sample,
            }

        # ----- 2. Caso geral: escolhe agente
        agent = _route_agent(
            prompt,
            agent_posweek,
            agent_status,
            agent_relweek,
            agent_item,
            agent_summary,
            agent_misto,
        )

        # ----- 3. Pede pro agente gerar SQL
        try:
            agent_res = agent.invoke({"input": prompt})
            raw_out = agent_res.get("output", "")
        except Exception:
            # fallback besta: usa create_sql_query_chain pra tentar gerar SELECT cru
            raw = query_chain.invoke({"question": f"Gere apenas a consulta SQL para: {prompt}"})
            raw_out = str(raw)

        sql = _only_sql(raw_out).strip()

        # se isso NÃO parece um SELECT, provavelmente o agente já respondeu em texto
        if not sql.lower().startswith("select"):
            texto_final = raw_out.strip() if raw_out else "Não consegui gerar resposta."
            return {
                "output": texto_final,
                "sql": None,
                "rows": [],
            }

        # ----- 4. Executa SQL no banco
        rows_sample = _run_sql_safe(sql)

        # ----- 5. Usa llm_fallback pra gerar resumo em PT-BR
        texto_final = _summarize_result(prompt, rows_sample)

        # ----- 6. Retorna pacote
        return {
            "output": texto_final,
            "sql": sql,
            "rows": rows_sample,
        }

    return run_query
