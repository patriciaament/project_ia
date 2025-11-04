# agent.py
# -*- coding: utf-8 -*-

import os
import re
from typing import Optional, Dict, Any

from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain

DB_URI = "sqlite:///db/base.db"

# pega códigos tipo A8350, A5460...
SKU_RX = re.compile(r"\b([A-Z]\d{3,8})\b", re.I)


def _get_api_key(passed: Optional[str]) -> str:
    key = passed or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("Faltou a OPENAI_API_KEY no ambiente.")
    return key


def _only_sql(text: str) -> str:
    if not text:
        return ""
    t = text.strip().strip("`").strip()
    if t.lower().startswith("select"):
        return t
    m = re.search(r"(?is)\bselect\b.+", t)
    if m:
        return m.group(0).strip()
    return t


def _extract_item(prompt: str) -> Optional[str]:
    m = SKU_RX.search(prompt or "")
    return m.group(1).upper() if m else None


def _looks_like_item_question(prompt: str) -> bool:
    p = (prompt or "").lower()
    gatilhos = [
        "descrição do item",
        "descricao do item",
        "item description",
        "o que você pode me dizer do item",
        "o que voce pode me dizer do item",
        "informações do item",
        "informacoes do item",
        "dados do item",
    ]
    return any(g in p for g in gatilhos)


def _format_item_answer(row: Dict[str, Any], code: str) -> str:
    desc = (
        row.get("ITEM DESCRIPTION")
        or row.get("Item Description")
        or row.get("item_description")
        or ""
    )
    lvl1 = row.get("Level_1") or row.get("LEVEL_1") or ""
    lvl2 = row.get("Level_2") or row.get("LEVEL_2") or ""
    lvl3 = row.get("Level_3") or row.get("LEVEL_3") or ""
    lvl4 = row.get("Level_4") or row.get("LEVEL_4") or ""

    partes = [f"O item {code} foi encontrado."]
    if desc:
        partes.append(f"Descrição: {desc}.")
    if any([lvl1, lvl2, lvl3, lvl4]):
        partes.append(
            "Classificação: "
            + " > ".join([x for x in [lvl1, lvl2, lvl3, lvl4] if x])
            + "."
        )
    return " ".join(partes)


def _summarize_rows(rows: Any) -> str:
    if isinstance(rows, dict) and rows.get("_error"):
        return "A SQL foi gerada, mas o banco devolveu erro. Veja a SQL para depurar."
    if not rows:
        return "Consulta concluída, mas não houve registros para esse filtro."
    if isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict):
        linha = "; ".join(f"{k}: {v}" for k, v in rows[0].items())
        return f"Encontrei 1 registro: {linha}"
    if isinstance(rows, list):
        cols = list(rows[0].keys()) if rows and isinstance(rows[0], dict) else []
        return (
            f"Encontrei {len(rows)} registros. "
            f"Colunas principais: {', '.join(cols)}. "
            f"Veja a SQL gerada para detalhes."
        )
    return "Consulta concluída."


def get_agent(open_api_key: Optional[str] = None):
    api_key = _get_api_key(open_api_key)

    # conexão única
    db = SQLDatabase.from_uri(DB_URI)

    # LLM único
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

    # cadeia genérica de SQL (para perguntas que não são de item)
    sql_chain = create_sql_query_chain(llm, db, k=3)

    def run_query(user_prompt: str) -> Dict[str, Any]:
        # --------------------------------------------------
        # 1. CASO ESPECIAL: descrição / dados de ITEM
        # --------------------------------------------------
        item_code = _extract_item(user_prompt)
        if item_code and _looks_like_item_question(user_prompt):
            # vamos tentar 2 nomes de tabela:
            # 1) item_master
            # 2) "ITEM MASTER"
            sql_try_1 = f"""
            SELECT
                "ITEM",
                "ITEM DESCRIPTION",
                "Level_1",
                "Level_2",
                "Level_3",
                "Level_4"
            FROM item_master
            WHERE "ITEM" = '{item_code}'
            LIMIT 1;
            """.strip()

            try:
                rows = db.run(sql_try_1)
                table_used = "item_master"
            except Exception:
                # tenta o nome com espaço, que é o que veio do Sheets
                sql_try_2 = f"""
                SELECT
                    "ITEM",
                    "ITEM DESCRIPTION",
                    "Level_1",
                    "Level_2",
                    "Level_3",
                    "Level_4"
                FROM "ITEM MASTER"
                WHERE "ITEM" = '{item_code}'
                LIMIT 1;
                """.strip()
                try:
                    rows = db.run(sql_try_2)
                    table_used = '"ITEM MASTER"'
                    sql_try_1 = sql_try_2  # para devolver a que funcionou
                except Exception as e2:
                    return {
                        "output": (
                            f"Tentei buscar o item {item_code}, mas o banco não encontrou "
                            f"as tabelas item_master nem \"ITEM MASTER\".\n"
                            f"Erro original: {e2}"
                        ),
                        "sql": sql_try_1,
                        "rows": [],
                    }

            if rows:
                texto = _format_item_answer(rows[0], item_code)
            else:
                texto = f"O item {item_code} não foi encontrado na tabela {table_used}."

            return {
                "output": texto,
                "sql": sql_try_1,
                "rows": rows,
            }

        # --------------------------------------------------
        # 2. CASO GERAL: deixar o LLM gerar a SQL
        # --------------------------------------------------
        try:
            raw_sql = sql_chain.invoke({"question": user_prompt})
        except Exception as e:
            return {
                "output": f"Não consegui gerar SQL automaticamente: {e}",
                "sql": None,
                "rows": [],
            }

        sql_final = _only_sql(raw_sql)

        if not sql_final.lower().startswith("select"):
            return {
                "output": sql_final,
                "sql": None,
                "rows": [],
            }

        try:
            rows = db.run(sql_final)
        except Exception as e:
            return {
                "output": f"A SQL foi gerada mas o banco devolveu erro: {e}",
                "sql": sql_final,
                "rows": [],
            }

        texto = _summarize_rows(rows)
        return {
            "output": texto,
            "sql": sql_final,
            "rows": rows,
        }

    return run_query
