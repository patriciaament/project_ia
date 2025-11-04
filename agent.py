# agent.py
# -*- coding: utf-8 -*-

import os
import re
from typing import Optional, Dict, Any, List

from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain


# ----------------------------------------------------------------------
# CONFIG BÁSICA
# ----------------------------------------------------------------------
DB_URI = "sqlite:///db/base.db"

# item / sku padrão seu: A8350, A5460 etc
SKU_RX = re.compile(r"\b([A-Z]\d{3,8})\b", re.I)


# ----------------------------------------------------------------------
# HELPERs
# ----------------------------------------------------------------------
def _get_api_key(passed_key: Optional[str]) -> str:
    key = passed_key or os.getenv("OPENAI_API_KEY")
    if not key:
        # deixa explodir cedo pra ficar claro no Streamlit
        raise ValueError("Faltou a OPENAI_API_KEY no ambiente (.env ou secrets.toml).")
    return key


def _only_sql(text: str) -> str:
    """
    LangChain às vezes devolve texto + SQL. Aqui eu puxo só o SELECT.
    """
    if not text:
        return ""
    t = text.strip().strip("`").strip()
    if t.lower().startswith("select"):
        return t
    m = re.search(r"(?is)\bselect\b.+", t)
    if m:
        return m.group(0).strip()
    return t


def _extract_item_from_prompt(prompt: str) -> Optional[str]:
    """
    Pega o primeiro código no formato A1234 da pergunta.
    """
    m = SKU_RX.search(prompt or "")
    return m.group(1).upper() if m else None


def _looks_like_item_question(prompt: str) -> bool:
    """
    Detecta perguntas do tipo:
      - qual a descrição do item A8350?
      - o que você pode me dizer do item A8350
      - qual o item description do item A8350
    """
    p = (prompt or "").lower()
    gatilhos = [
        "descrição do item",
        "descricao do item",
        "item description",
        "o que você pode me dizer do item",
        "o que voce pode me dizer do item",
        "dados do item",
        "informações do item",
        "informacoes do item",
    ]
    return any(g in p for g in gatilhos)


def _format_item_answer(row: Dict[str, Any], code: str) -> str:
    desc = row.get("ITEM DESCRIPTION") or row.get("item description") or ""
    lvl1 = row.get("Level_1") or ""
    lvl2 = row.get("Level_2") or ""
    lvl3 = row.get("Level_3") or ""
    lvl4 = row.get("Level_4") or ""

    partes = [f"O item {code} existe na base."]
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
        return "A SQL foi gerada, mas houve erro ao executar no banco. Veja a SQL gerada para depurar."
    if not rows:
        return "Consulta concluída, mas não encontrei registros para esse filtro."
    if isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict):
        # 1 linha -> devolve os campos
        cols = "; ".join(f"{k}: {v}" for k, v in rows[0].items())
        return f"Encontrei 1 registro: {cols}"
    if isinstance(rows, list):
        cols = list(rows[0].keys()) if rows and isinstance(rows[0], dict) else []
        return (
            f"Encontrei {len(rows)} registros. "
            f"Colunas principais: {', '.join(cols)}. "
            f"Veja a SQL gerada para detalhes."
        )
    return "Consulta concluída."


# ----------------------------------------------------------------------
# FUNÇÃO PRINCIPAL
# ----------------------------------------------------------------------
def get_agent(open_api_key: Optional[str] = None):
    """
    devolve a função run_query(prompt) que o Streamlit chama
    """
    api_key = _get_api_key(open_api_key)

    # conexão única
    db = SQLDatabase.from_uri(DB_URI)

    # LLM única
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key,
    )

    # chain genérica de SQL
    sql_chain = create_sql_query_chain(llm, db, k=3)

    def run_query(user_prompt: str) -> Dict[str, Any]:
        # ------------------------------------------------------------------
        # 1) CASO ESPECIAL: perguntas de ITEM → vou direto na item_master
        # ------------------------------------------------------------------
        item_code = _extract_item_from_prompt(user_prompt)
        if item_code and _looks_like_item_question(user_prompt):
            # nome da tabela conforme o seu SQLite
            sql_item = f"""
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
                rows = db.run(sql_item)
            except Exception as e:
                return {
                    "output": f"Tentei buscar o item {item_code}, mas o banco retornou erro: {e}",
                    "sql": sql_item,
                    "rows": [],
                }

            if rows:
                texto = _format_item_answer(rows[0], item_code)
            else:
                texto = f"Não encontrei o item {item_code} na tabela item_master."

            return {
                "output": texto,
                "sql": sql_item,
                "rows": rows,
            }

        # ------------------------------------------------------------------
        # 2) CASO GERAL → LLM gera SQL, eu rodo, devolvo resumo
        # ------------------------------------------------------------------
        try:
            raw_sql = sql_chain.invoke({"question": user_prompt})
        except Exception as e:
            return {
                "output": f"Não consegui gerar a SQL automaticamente: {e}",
                "sql": None,
                "rows": [],
            }

        sql_final = _only_sql(raw_sql)

        # se mesmo assim não veio SELECT, devolve o texto
        if not sql_final.lower().startswith("select"):
            return {
                "output": sql_final,
                "sql": None,
                "rows": [],
            }

        # executa
        try:
            rows = db.run(sql_final)
        except Exception as e:
            return {
                "output": f"A SQL foi gerada mas deu erro ao executar: {e}",
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
