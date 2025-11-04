# agent.py
# -*- coding: utf-8 -*-

import os
import re
from typing import Optional, Dict, Any

from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain

DB_URI = "sqlite:///db/base.db"

SKU_RX = re.compile(r"\b([A-Z]\d{3,8})\b", re.I)


def _extract_sku(prompt: str) -> Optional[str]:
    m = SKU_RX.search(prompt or "")
    return m.group(1).upper() if m else None


def _is_description_question(prompt: str) -> bool:
    p = (prompt or "").lower()
    return ("descrição" in p) or ("descricao" in p) or ("description" in p)


def get_agent(open_api_key: Optional[str] = None):
    api_key = open_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Faltou a OPENAI_API_KEY no ambiente.")

    db = SQLDatabase.from_uri(DB_URI)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    sql_chain = create_sql_query_chain(llm, db, k=3)

    def _get_desc_from_tables(sku: str) -> Optional[str]:
        # tenta item_master
        try:
            rows = db.run(f'''
                SELECT "ITEM DESCRIPTION" AS desc_txt
                FROM item_master
                WHERE "ITEM" = '{sku}'
                LIMIT 1;
            ''')
            if rows and rows[0].get("desc_txt"):
                return rows[0]["desc_txt"]
        except Exception:
            pass

        # tenta classificacao_items
        try:
            rows = db.run(f'''
                SELECT "ITEM DESCRIPTION" AS desc_txt
                FROM classificacao_items
                WHERE "ITEM" = '{sku}'
                LIMIT 1;
            ''')
            if rows and rows[0].get("desc_txt"):
                return rows[0]["desc_txt"]
        except Exception:
            pass

        return None

    def _only_sql(text: str) -> str:
        if not text:
            return ""
        txt = text.strip().strip("`").strip()
        if txt.lower().startswith("select"):
            return txt
        m = re.search(r"(?is)\bselect\b.+", txt)
        return m.group(0).strip() if m else txt

    def _run_sql_safe(sql: str):
        try:
            return db.run(sql)
        except Exception as e:
            return {"_error": str(e), "_sql": sql}

    def _summarize(rows: Any) -> str:
        if isinstance(rows, dict) and "_error" in rows:
            return "Houve erro ao executar a consulta. Veja a SQL gerada."
        if not rows:
            return "Nenhum dado encontrado."
        if isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict):
            partes = [f"{k}: {v}" for k, v in rows[0].items()]
            return "Encontrei 1 registro: " + "; ".join(partes)
        if isinstance(rows, list):
            cols = list(rows[0].keys()) if rows and isinstance(rows[0], dict) else []
            return f"Encontrei {len(rows)} registros. Colunas: {', '.join(cols)}."
        return "Consulta concluída."

    def run_query(prompt: str) -> Dict[str, Any]:
        sku = _extract_sku(prompt)

        # 1) PERGUNTA DE DESCRIÇÃO → nunca devolve SQL
        if sku and _is_description_question(prompt):
            desc = _get_desc_from_tables(sku)
            if desc:
                return {
                    "output": f"A descrição do item {sku} é: {desc}",
                    "sql": None,
                    "rows": []
                }
            else:
                return {
                    "output": f"Não encontrei a descrição do item {sku} nas tabelas disponíveis.",
                    "sql": None,
                    "rows": []
                }

        # 2) CASO GERAL → gera SQL
        raw = sql_chain.invoke({
            "question": f"Gere apenas o SELECT SQL para SQLite, sem explicação, para: {prompt}"
        })
        sql = _only_sql(raw).strip()
        if not sql.lower().startswith("select"):
            return {
                "output": "Não consegui gerar uma consulta SQL válida.",
                "sql": None,
                "rows": []
            }

        rows = _run_sql_safe(sql)
        texto = _summarize(rows)

        return {
            "output": texto,
            "sql": sql,
            "rows": rows
        }

    return run_query
