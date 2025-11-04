# agent.py
# -*- coding: utf-8 -*-

import os
import re
from typing import Optional, Dict, Any

from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain

DB_URI = "sqlite:///db/base.db"

# SKU tipo A1234
_SKU_RX = re.compile(r"\b([A-Z]\d{3,8})\b", re.I)
_STOP_TOKENS = [
    " em ", " no ", " na ", " de ", " do ", " da ",
    " para ", " por ", " com ", " que ", " e ", " ou ",
    " onde ", " quando "
]
_STOP_PUNCT = r"[\,\.\?\:\;\!\|/()\[\]\n\r\t]"


def _extract_sku_and_client(prompt: str):
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
        if rest.startswith('"'):
            m = re.search(r'^"([^"]+)"', rest)
            if m:
                cliente = m.group(1).strip()
        elif rest.startswith("'"):
            m = re.search(r"^'([^']+)'", rest)
            if m:
                cliente = m.group(1).strip()
        else:
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


def _only_sql(text: str) -> str:
    if not text:
        return ""
    txt = text.strip().strip("`").strip()
    if txt.lower().startswith("select"):
        return txt
    m = re.search(r"(?is)\bselect\b.+", txt, re.DOTALL)
    return m.group(0).strip() if m else txt


def _is_description_question(prompt: str) -> bool:
    p = (prompt or "").lower()
    return ("descrição" in p) or ("descricao" in p) or ("description" in p)


def _get_description_by_sku(db: SQLDatabase, sku: str) -> Optional[str]:
    # 1) tenta item_master
    try:
        sql1 = f'''
        SELECT "ITEM DESCRIPTION" AS desc_txt
        FROM item_master
        WHERE "ITEM" = '{sku}'
        LIMIT 1;
        '''.strip()
        rows = db.run(sql1)
        if rows and rows[0].get("desc_txt"):
            return rows[0]["desc_txt"]
    except Exception:
        pass

    # 2) tenta classificacao_items
    try:
        sql2 = f'''
        SELECT "ITEM DESCRIPTION" AS desc_txt
        FROM classificacao_items
        WHERE "ITEM" = '{sku}'
        LIMIT 1;
        '''.strip()
        rows = db.run(sql2)
        if rows and rows[0].get("desc_txt"):
            return rows[0]["desc_txt"]
    except Exception:
        pass

    return None


def _summarize_rows(rows: Any) -> str:
    if isinstance(rows, dict) and "_error" in rows:
        return "Houve erro ao executar a consulta. Veja a SQL gerada abaixo."
    if not rows:
        return "Nenhum dado encontrado para a consulta."
    if isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict):
        partes = [f"{k}: {v}" for k, v in rows[0].items()]
        return "Encontrei 1 registro: " + "; ".join(partes)
    if isinstance(rows, list):
        cols = list(rows[0].keys()) if rows and isinstance(rows[0], dict) else []
        return f"Encontrei {len(rows)} registros. Colunas principais: {', '.join(cols)}."
    return "Consulta concluída."


def get_agent(open_api_key: Optional[str] = None):
    api_key = open_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Faltou a OPENAI_API_KEY no ambiente.")

    db = SQLDatabase.from_uri(DB_URI)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    sql_chain = create_sql_query_chain(llm, db, k=3)

    def _run_sql_safe(sql: str):
        try:
            return db.run(sql)
        except Exception as e:
            return {"_error": str(e), "_sql": sql}

    def run_query(prompt: str) -> Dict[str, Any]:
        sku, cliente = _extract_sku_and_client(prompt)

        # 1) descrição do item → NÃO mostrar SQL se não achar
        if sku and _is_description_question(prompt):
            desc = _get_description_by_sku(db, sku)
            if desc:
                return {
                    "output": f"A descrição do item {sku} é: {desc}",
                    "sql": None,
                    "rows": []
                }
            else:
                return {
                    "output": (
                        f"Não encontrei a descrição do item {sku} nas tabelas disponíveis. "
                        "Verifique se esse SKU está no ITEM MASTER da base."
                    ),
                    "sql": None,
                    "rows": []
                }

        # 2) SKU + cliente → tenta estoque / classe / retail
        if sku and cliente:
            sql = f'''
            SELECT
              s."OHI CY"                  AS ohi_cy,
              s."OHI Var%"                AS ohi_var,
              st."Status POS Master 2025" AS status,
              rw."RETAIL"                 AS retail
            FROM summary_country s
            LEFT JOIN status_sku     st ON st."SKU" = s."Item"
            LEFT JOIN relatorio_week rw ON rw."SKU" = s."Item"
            LEFT JOIN classificacao_clientes cc
                ON cc."Nome Fictício" = s."Client DC Group"
            WHERE s."Item" = '{sku}'
              AND (cc."Nome Fictício" = '{cliente}' OR s."Client DC Group" = '{cliente}')
            LIMIT 1;
            '''.strip()
            rows = _run_sql_safe(sql)
            if not (isinstance(rows, dict) and "_error" in rows) and rows:
                r = rows[0]
                status = (r.get("status") or "").upper()
                classe = "TLP" if "TLP" in status else "NTLP"
                retail = r.get("retail")
                retail_fmt = f"{retail:.2f}".replace(".", ",") if retail else "0,00"
                return {
                    "output": (
                        f"Para o SKU {sku} no cliente {cliente}: está classificado como {classe}. "
                        f"Retail Price: {retail_fmt}. Estoque atual: {r.get('ohi_cy')}, "
                        f"variação: {r.get('ohi_var')}."
                    ),
                    "sql": sql,
                    "rows": rows,
                }

        # 3) caso geral → gera SQL
        raw_sql = sql_chain.invoke({
            "question": f"Gere apenas o SELECT SQL para SQLite com base na pergunta: {prompt}"
        })
        sql = _only_sql(raw_sql).strip()

        if not sql.lower().startswith("select"):
            return {
                "output": "Não consegui gerar uma consulta SQL válida.",
                "sql": None,
                "rows": [],
            }

        rows = _run_sql_safe(sql)
        texto = _summarize_rows(rows)

        return {
            "output": texto,
            "sql": sql,
            "rows": rows,
        }

    return run_query
