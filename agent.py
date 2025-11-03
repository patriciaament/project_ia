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
    """Tenta achar um SKU e o nome do cliente a partir da pergunta."""
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

        # "cliente \"Mundo da Criança\""
        if rest.startswith('"'):
            m = re.search(r'^"([^"]+)"', rest)
            if m:
                cliente = m.group(1).strip()

        # "cliente 'Mundo da Criança'"
        if not cliente and rest.startswith("'"):
            m = re.search(r"^'([^']+)'", rest)
            if m:
                cliente = m.group(1).strip()

        # sem aspas
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

        # limitar tamanho
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
        return (
            "Tentei consultar os dados para sua pergunta, mas houve um problema ao "
            "executar a query no banco. Você pode ver a SQL gerada para entender o que foi pedido."
        )

    if not rows:
        return "Não encontrei dados relevantes pra essa pergunta no banco."

    if isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict):
        row = rows[0]
        partes = [f"{k}: {v}" for k, v in row.items()]
        detalhe = "; ".join(partes)
        return f"Encontrei 1 registro relacionado à sua pergunta. Principais dados: {detalhe}."

    if isinstance(rows, list):
        cols = []
        if len(rows) > 0 and isinstance(rows[0], dict):
            cols = list(rows[0].keys())
        return (
            f"Encontrei {len(rows)} linhas que respondem à pergunta. "
            f"As colunas principais recuperadas foram: {', '.join(cols)}."
        )

    return "Consulta concluída."


# ======================================================
# get_agent (principal)
# ======================================================

def get_agent(open_api_key: Optional[str] = None):
    api_key = open_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Faltou a OPENAI_API_KEY no ambiente (.env ou secrets).")

    db_main = SQLDatabase.from_uri(DB_URI)
    llm_main = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key,
    )

    query_chain = create_sql_query_chain(llm_main, db_main, k=4)

    def _run_sql_safe(sql: str):
        try:
            return db_main.run(sql)
        except Exception as e:
            return {"_error": str(e), "_sql": sql}

    def _make_sql_agent(tables: List[str]):
        sub_db = SQLDatabase.from_uri(DB_URI, include_tables=tables)
        toolkit = SQLDatabaseToolkit(db=sub_db, llm=llm_main)

        BASE_CONTEXT = f"""
        Você é um gerador/validador de SQL focado em SQLite.
        Gere APENAS SELECTs válidos para as tabelas:
        {', '.join(tables)}.
        Não invente tabelas nem colunas.
        """

        memory = ConversationBufferWindowMemory(
            k=3, memory_key="chat_history", return_messages=True
        )

        return create_sql_agent(
            llm=llm_main,
            toolkit=toolkit,
            verbose=False,
            handle_parsing_errors=True,
            prefix=BASE_CONTEXT,
            memory=memory,
            # ↓↓↓ isso que estava muito alto
            max_iterations=2,
            max_execution_time=12,
        )

    # agentes
    agent_summary = _make_sql_agent(["summary_country"])
    agent_posweek = _make_sql_agent(["pos_week"])
    agent_status = _make_sql_agent(["status_sku"])
    agent_relweek = _make_sql_agent(["relatorio_week"])
    agent_item = _make_sql_agent(["classificacao_items", "item_master"])
    agent_clientes = _make_sql_agent(["classificacao_clientes"])
    agent_misto = _make_sql_agent([
        "summary_country",
        "pos_week",
        "status_sku",
        "relatorio_week",
        "classificacao_items",
        "classificacao_clientes",
        "item_master"
    ])

    def _route_agent(prompt: str):
        p = prompt.lower()
        if any(x in p for x in ["pos", "semana", "lw", "últimas", "ultimas", "ytd"]):
            return agent_posweek
        if any(x in p for x in ["tlp", "ntlp", "status", "classificação", "classificacao", "sku"]):
            return agent_status
        if any(x in p for x in ["estoque", "ohi", "retail", "preço", "preco", "sugerido"]):
            return agent_relweek
        if any(x in p for x in ["descrição", "descricao", "marca", "level", "item", "produto"]):
            return agent_item
        if any(x in p for x in ["resumo", "country", "mundo", "visão geral", "visao geral"]):
            return agent_summary
        if any(x in p for x in ["cliente", "canal", "rede"]):
            return agent_clientes
        return agent_misto

    def _monta_texto_sku_cliente_full(sku: str, cliente: str, row: Dict[str, Any]) -> str:
        status_val = (row.get("status") or "").upper()
        classe = "TLP" if "TLP" in status_val else "NTLP"
        retail_val = row.get("retail")
        retail_fmt = _fmt_decimal_brl(retail_val, 2)
        ohi_cy = row.get("ohi_cy")
        ohi_var = row.get("ohi_var")
        return (
            f"Para o SKU {sku} no cliente {cliente}: "
            f"ele está classificado como {classe}. "
            f"O Retail Price é {retail_fmt}. "
            f"O estoque atual registrado é {ohi_cy} e a variação percentual é {ohi_var}."
        )

    def _monta_texto_sku_cliente_fallback(sku: str, cliente: str, row: Dict[str, Any]) -> str:
        status_val = (row.get("status") or "").upper()
        classe = "TLP" if "TLP" in status_val else "NTLP"
        retail_val = row.get("retail")
        retail_fmt = _fmt_decimal_brl(retail_val, 2)
        return (
            f"Para o SKU {sku}: ele está classificado como {classe} e o Retail Price é {retail_fmt}. "
            f"Não encontrei os dados específicos do cliente {cliente}."
        )

    def run_query(prompt: str) -> Dict[str, Any]:
        sku, cliente = _extract_sku_and_client(prompt)
        if sku and cliente:
            # ====== consulta completa (usa COALESCE por causa do seu erro) ======
            sql_full = f"""
SELECT
  s."OHI CY"                  AS ohi_cy,
  s."OHI Var%"                AS ohi_var,
  st."Status POS Master 2025" AS status,
  rw."RETAIL"                 AS retail
FROM summary_country s
LEFT JOIN status_sku     st ON st."SKU" = COALESCE(s."Item", s."SKU", s."Item Code")
LEFT JOIN relatorio_week rw ON rw."SKU" = COALESCE(s."Item", s."SKU", s."Item Code")
LEFT JOIN classificacao_clientes cc
    ON cc."Nome Fictício" = s."Client DC Group"
WHERE COALESCE(s."Item", s."SKU", s."Item Code") = '{sku}'
  AND (
        cc."Nome Fictício" = '{cliente}'
        OR s."Client DC Group" = '{cliente}'
      )
LIMIT 1;
""".strip()

            rows_full = _run_sql_safe(sql_full)

            if not (isinstance(rows_full, dict) and "_error" in rows_full) and rows_full:
                texto = _monta_texto_sku_cliente_full(sku, cliente, rows_full[0])
                return {"output": texto, "sql": sql_full, "rows": rows_full}

            # ====== fallback sem cliente ======
            sql_fb = f"""
SELECT
  st."Status POS Master 2025" AS status,
  rw."RETAIL"                 AS retail
FROM status_sku st
LEFT JOIN relatorio_week rw
    ON rw."SKU" = st."SKU"
WHERE st."SKU" = '{sku}'
LIMIT 1;
""".strip()

            rows_fb = _run_sql_safe(sql_fb)
            if not (isinstance(rows_fb, dict) and "_error" in rows_fb) and rows_fb:
                texto = _monta_texto_sku_cliente_fallback(sku, cliente, rows_fb[0])
                return {"output": texto, "sql": sql_fb, "rows": rows_fb}

            return {
                "output": f"Tentei consultar o SKU {sku} no cliente {cliente}, mas não encontrei dados.",
                "sql": sql_full,
                "rows": rows_full,
            }

        # ===== caso geral: usa agente =====
        agent = _route_agent(prompt)
        try:
            agent_res = agent.invoke({"input": prompt})
            raw_out = agent_res.get("output", "")
        except Exception:
            chain_raw = query_chain.invoke({"question": f"Gere apenas a consulta SQL para: {prompt}"})
            raw_out = str(chain_raw)

        sql_candidate = _only_sql(raw_out).strip()

        if not sql_candidate.lower().startswith("select"):
            texto_final = raw_out.strip() if raw_out else "Não consegui gerar resposta estruturada."
            return {"output": texto_final, "sql": None, "rows": []}

        rows_sample = _run_sql_safe(sql_candidate)
        texto_final = _summarize_result(prompt, rows_sample)

        return {"output": texto_final, "sql": sql_candidate, "rows": rows_sample}

    return run_query
