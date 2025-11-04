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

def _extract_sku(prompt: str) -> Optional[str]:
    m = _SKU_RX.search(prompt or "")
    if m:
        return m.group(1).upper()
    return None


def _extract_sku_and_client(prompt: str):
    """ainda usamos esse para o caso SKU + cliente"""
    sku = _extract_sku(prompt)

    cliente = None
    p = prompt or ""
    p_low = p.lower()

    idx = p_low.find("cliente")
    if idx >= 0:
        rest = p[idx + len("cliente"):].lstrip()

        # cliente "Nome"
        if rest.startswith('"'):
            m = re.search(r'^"([^"]+)"', rest)
            if m:
                cliente = m.group(1).strip()

        # cliente 'Nome'
        if not cliente and rest.startswith("'"):
            m = re.search(r"^'([^']+)'", rest)
            if m:
                cliente = m.group(1).strip()

        # cliente sem aspas
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

        if cliente:
            parts = cliente.split()
            if len(parts) > 6:
                cliente = " ".join(parts[:6]).strip()

    return sku, cliente


def _fmt_decimal_brl(x: float, casas=2) -> str:
    try:
        s = f"{float(x):.{casas}f}"
    except Exception:
        return str(x)
    return s.replace(".", ",")


def _only_sql(text: str) -> str:
    if not text:
        return ""
    txt = text.strip().strip("`").strip()
    if txt.lower().startswith("select"):
        return txt
    m = re.search(r"(?is)\bselect\b.+", txt, re.DOTALL)
    if m:
        return m.group(0).strip()
    return txt


def _summarize_result(pergunta: str, rows: Any) -> str:
    if isinstance(rows, dict) and "_error" in rows:
        return ("Tentei consultar os dados, mas houve um erro ao executar a query no banco. "
                "Veja a SQL gerada no painel.")
    if not rows:
        return "Não encontrei dados relevantes pra essa pergunta no banco."
    if isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict):
        row = rows[0]
        partes = [f"{k}: {v}" for k, v in row.items()]
        return "Encontrei 1 registro: " + "; ".join(partes)
    if isinstance(rows, list):
        cols = list(rows[0].keys()) if rows and isinstance(rows[0], dict) else []
        return (f"Encontrei {len(rows)} linhas que atendem à consulta. "
                f"Colunas principais: {', '.join(cols)}.")
    return "Consulta concluída."


# ======================================================
# get_agent (principal)
# ======================================================

def get_agent(open_api_key: Optional[str] = None):
    api_key = open_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Faltou a OPENAI_API_KEY no ambiente (.env ou secrets).")

    db_main = SQLDatabase.from_uri(DB_URI)
    llm_main = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    query_chain = create_sql_query_chain(llm_main, db_main, k=4)

    def _run_sql_safe(sql: str):
        try:
            return db_main.run(sql)
        except Exception as e:
            return {"_error": str(e), "_sql": sql}

    # ------------- agents menores -------------
    def _make_sql_agent(tables: List[str]):
        sub_db = SQLDatabase.from_uri(DB_URI, include_tables=tables)
        toolkit = SQLDatabaseToolkit(db=sub_db, llm=llm_main)
        base_ctx = f"""
        Você é um gerador de SQL para SQLite.
        Gere APENAS SELECTs válidos para as tabelas: {', '.join(tables)}.
        Não coloque explicação junto. Apenas o SELECT.
        """
        memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history", return_messages=True)
        return create_sql_agent(
            llm=llm_main,
            toolkit=toolkit,
            verbose=False,
            handle_parsing_errors=True,
            prefix=base_ctx,
            memory=memory,
            max_iterations=3,
            max_execution_time=20,
        )

    agent_posweek = _make_sql_agent(["pos_week"])
    agent_status = _make_sql_agent(["status_sku"])
    agent_relweek = _make_sql_agent(["relatorio_week"])
    agent_item = _make_sql_agent(["classificacao_items", "item_master"])
    agent_clientes = _make_sql_agent(["classificacao_clientes"])
    agent_summary = _make_sql_agent(["summary_country"])
    agent_misto = _make_sql_agent([
        "summary_country",
        "pos_week",
        "status_sku",
        "relatorio_week",
        "classificacao_items",
        "classificacao_clientes",
        "item_master",
    ])

    def _route_agent(prompt: str):
        p = prompt.lower()
        if any(x in p for x in ["pos", "semana", "lw", "4 semanas", "ytd"]):
            return agent_posweek
        if any(x in p for x in ["tlp", "ntlp", "status", "classificação", "classificacao"]):
            return agent_status
        if any(x in p for x in ["estoque", "ohi", "retail", "preço", "preco"]):
            return agent_relweek
        if any(x in p for x in ["descrição", "description", "item description", "level", "sku", "item"]):
            return agent_item
        if any(x in p for x in ["cliente", "canal", "rede"]):
            return agent_clientes
        if any(x in p for x in ["resumo", "country", "visão geral", "visao geral"]):
            return agent_summary
        return agent_misto

    # --------------------------------------------------
    # HOOK 1: perguntas de descrição de item
    # --------------------------------------------------
    def _maybe_answer_item_description(prompt: str):
        p = (prompt or "").lower()
        wants_description = any(
            k in p for k in [
                "descrição do item",
                "descricao do item",
                "item description",
                "o que você pode me dizer do item",
                "me diga do item",
                "descreva o item",
            ]
        )
        sku = _extract_sku(prompt)

        if not wants_description or not sku:
            return None  # não é esse caso

        # 1) tenta na tabela item_master (que é a que você mostrou)
        sql_item_master = f'''
SELECT "ITEM", "ITEM DESCRIPTION", "Level_1", "Level_2", "Level_3", "Level_4"
FROM item_master
WHERE "ITEM" = '{sku}'
LIMIT 1;
'''.strip()

        rows = _run_sql_safe(sql_item_master)
        if not (isinstance(rows, dict) and "_error" in rows) and rows:
            r = rows[0]
            desc = r.get("ITEM DESCRIPTION") or r.get("item description") or "sem descrição cadastrada"
            l1 = r.get("Level_1") or r.get("level_1")
            l2 = r.get("Level_2") or r.get("level_2")
            l3 = r.get("Level_3") or r.get("level_3")
            l4 = r.get("Level_4") or r.get("level_4")
            texto = f"O item {sku} tem a descrição: {desc}."
            niveis = [l for l in [l1, l2, l3, l4] if l]
            if niveis:
                texto += " Ele está classificado nos níveis: " + " > ".join(niveis) + "."
            return {
                "output": texto,
                "sql": sql_item_master,
                "rows": rows,
            }

        # 2) fallback na classificacao_items (caso o SQLite esteja com esse nome)
        sql_classif = f'''
SELECT "ITEM", "ITEM DESCRIPTION", "Level_1", "Level_2", "Level_3", "Level_4"
FROM classificacao_items
WHERE "ITEM" = '{sku}'
LIMIT 1;
'''.strip()
        rows2 = _run_sql_safe(sql_classif)
        if not (isinstance(rows2, dict) and "_error" in rows2) and rows2:
            r = rows2[0]
            desc = r.get("ITEM DESCRIPTION") or "sem descrição cadastrada"
            l1 = r.get("Level_1")
            l2 = r.get("Level_2")
            l3 = r.get("Level_3")
            l4 = r.get("Level_4")
            texto = f"O item {sku} tem a descrição: {desc}."
            niveis = [l for l in [l1, l2, l3, l4] if l]
            if niveis:
                texto += " Ele está classificado nos níveis: " + " > ".join(niveis) + "."
            return {
                "output": texto,
                "sql": sql_classif,
                "rows": rows2,
            }

        # 3) se nem item_master nem classificacao_items funcionarem, diz claro
        return {
            "output": f"Não encontrei a descrição do item {sku} nas tabelas disponíveis.",
            "sql": sql_item_master,
            "rows": rows,
        }

    # --------------------------------------------------
    # HOOK 2: SKU + cliente (estoque / retail / TLP)
    # --------------------------------------------------
    def _monta_texto_sku_cliente_full(sku: str, cliente: str, row: Dict[str, Any]) -> str:
        status_val = (row.get("status") or "").upper()
        classe = "TLP" if "TLP" in status_val else "NTLP"
        retail_val = row.get("retail")
        retail_fmt = _fmt_decimal_brl(retail_val, 2)
        ohi_cy = row.get("ohi_cy")
        ohi_var = row.get("ohi_var")
        return (
            f"Para o SKU {sku} no cliente {cliente}: está classificado como {classe}. "
            f"Retail Price: {retail_fmt}. Estoque (OHI CY): {ohi_cy}, variação: {ohi_var}."
        )

    def _monta_texto_sku_cliente_fallback(sku: str, cliente: str, row: Dict[str, Any]) -> str:
        status_val = (row.get("status") or "").upper()
        classe = "TLP" if "TLP" in status_val else "NTLP"
        retail_val = row.get("retail")
        retail_fmt = _fmt_decimal_brl(retail_val, 2)
        return (
            f"Para o SKU {sku}: está classificado como {classe} e o retail price é {retail_fmt}. "
            f"Não consegui confirmar os dados específicos do cliente {cliente}."
        )

    # --------------------------------------------------
    # FUNÇÃO QUE O FRONT CHAMA
    # --------------------------------------------------
    def run_query(prompt: str) -> Dict[str, Any]:
        # 1) primeiro: se for claramente pedir descrição de SKU → tratamos aqui
        desc_res = _maybe_answer_item_description(prompt)
        if desc_res is not None:
            return desc_res

        # 2) depois: caso especial SKU + cliente (estoque / retail)
        sku, cliente = _extract_sku_and_client(prompt)
        if sku and cliente:
            sql_full = f'''
SELECT
  s."OHI CY"                  AS ohi_cy,
  s."OHI Var%"                AS ohi_var,
  st."Status POS Master 2025" AS status,
  rw."RETAIL"                 AS retail
FROM summary_country s
LEFT JOIN status_sku         st ON st."SKU" = s."Item"
LEFT JOIN relatorio_week     rw ON rw."SKU" = s."Item"
LEFT JOIN classificacao_clientes cc ON cc."Nome Fictício" = s."Client DC Group"
WHERE s."Item" = '{sku}'
  AND (cc."Nome Fictício" = '{cliente}' OR s."Client DC Group" = '{cliente}')
LIMIT 1;
'''.strip()
            rows_full = _run_sql_safe(sql_full)
            if not (isinstance(rows_full, dict) and "_error" in rows_full) and rows_full:
                texto = _monta_texto_sku_cliente_full(sku, cliente, rows_full[0])
                return {"output": texto, "sql": sql_full, "rows": rows_full}

            sql_fb = f'''
SELECT
  st."Status POS Master 2025" AS status,
  rw."RETAIL"                 AS retail
FROM status_sku st
LEFT JOIN relatorio_week rw ON rw."SKU" = st."SKU"
WHERE st."SKU" = '{sku}'
LIMIT 1;
'''.strip()
            rows_fb = _run_sql_safe(sql_fb)
            if not (isinstance(rows_fb, dict) and "_error" in rows_fb) and rows_fb:
                texto = _monta_texto_sku_cliente_fallback(sku, cliente, rows_fb[0])
                return {"output": texto, "sql": sql_fb, "rows": rows_fb}

            return {
                "output": f"Tentei recuperar dados do SKU {sku} para o cliente {cliente}, mas não consegui.",
                "sql": sql_full,
                "rows": rows_full,
            }

        # 3) caso geral → roteia pra um agente
        agent = _route_agent(prompt)
        try:
            agent_res = agent.invoke({"input": prompt})
            raw_out = agent_res.get("output", "")
        except Exception:
            chain_raw = query_chain.invoke({"question": f"Gere apenas o SELECT para: {prompt}"})
            raw_out = str(chain_raw)

        sql_candidate = _only_sql(raw_out).strip()
        if not sql_candidate.lower().startswith("select"):
            return {"output": raw_out or "Não consegui gerar resposta estruturada.", "sql": None, "rows": []}

        rows_sample = _run_sql_safe(sql_candidate)
        texto_final = _summarize_result(prompt, rows_sample)

        return {
            "output": texto_final,
            "sql": sql_candidate,
            "rows": rows_sample,
        }

    return run_query
