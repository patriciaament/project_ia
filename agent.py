# agent.py
# -*- coding: utf-8 -*-

import os
import re
from typing import Optional, Dict, Any, List

from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain

DB_URI = "sqlite:///db/base.db"

SKU_RX = re.compile(r"\b([A-Z]\d{3,8})\b", re.I)


# --------------------------------------------------------
# util: pega key
# --------------------------------------------------------
def _get_api_key(passed: Optional[str]) -> str:
    key = passed or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("Faltou a OPENAI_API_KEY no ambiente.")
    return key


# --------------------------------------------------------
# util: extrai só SELECT
# --------------------------------------------------------
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


# --------------------------------------------------------
# util: vê se é pergunta de item
# --------------------------------------------------------
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
        "qual a descrição do item",
        "qual a descricao do item",
    ]
    return any(g in p for g in gatilhos)


# --------------------------------------------------------
# NOVO: descobrir o nome da tabela de item no sqlite
# --------------------------------------------------------
def _find_item_table(db: SQLDatabase) -> Optional[str]:
    """
    lê sqlite_master e tenta achar a tabela que veio do seu Google Sheets de itens.
    """
    try:
        rows = db.run("SELECT name FROM sqlite_master WHERE type='table';")
    except Exception:
        return None

    if not rows:
        return None

    # vira lista de nomes
    table_names = [r["name"] if isinstance(r, dict) else r[0] for r in rows]

    # 1) preferidos, na ordem
    preferidos = [
        "item_master",
        "ITEM MASTER",
        "ITEM_MASTER",
        "ITEMMASTER",
    ]
    lower_map = {name.lower(): name for name in table_names}
    for pref in preferidos:
        if pref.lower() in lower_map:
            return lower_map[pref.lower()]

    # 2) se não achar, pega a primeira que contenha "item"
    candidatos = [
        name
        for name in table_names
        if "item" in name.lower()
    ]
    if candidatos:
        # pega o mais curto (normalmente é o mais limpo)
        candidatos.sort(key=len)
        return candidatos[0]

    return None


# --------------------------------------------------------
# formata resposta do item
# --------------------------------------------------------
def _format_item_answer(row: Dict[str, Any], code: str) -> str:
    desc = (
        row.get("ITEM DESCRIPTION")
        or row.get("Item Description")
        or row.get("item_description")
        or row.get("ITEM_DESCRIPTION")
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


# --------------------------------------------------------
# resumo genérico
# --------------------------------------------------------
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


# --------------------------------------------------------
# função principal
# --------------------------------------------------------
def get_agent(open_api_key: Optional[str] = None):
    api_key = _get_api_key(open_api_key)

    # conexão global
    db = SQLDatabase.from_uri(DB_URI)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    sql_chain = create_sql_query_chain(llm, db, k=3)

    # descobre o nome real da tabela de item
    discovered_item_table = _find_item_table(db)

    def run_query(user_prompt: str) -> Dict[str, Any]:
        # 1) caso especial de item
        item_code = _extract_item(user_prompt)
        if item_code and _looks_like_item_question(user_prompt):
            if not discovered_item_table:
                # não achou nenhuma tabela de item → devolve lista de tabelas
                try:
                    rows = db.run("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = [r["name"] for r in rows]
                except Exception:
                    tables = []
                return {
                    "output": (
                        f"Tentei buscar o item {item_code}, mas não consegui identificar "
                        f"qual tabela do banco contém os itens (ex.: 'item_master', 'ITEM MASTER'). "
                        f"Tabelas encontradas: {tables}"
                    ),
                    "sql": None,
                    "rows": [],
                }

            # monta a SQL usando o nome que descobrimos
            sql_item = f"""
            SELECT
                "ITEM",
                "ITEM DESCRIPTION",
                "Level_1",
                "Level_2",
                "Level_3",
                "Level_4"
            FROM "{discovered_item_table}"
            WHERE "ITEM" = '{item_code}'
            LIMIT 1;
            """.strip()

            try:
                rows = db.run(sql_item)
            except Exception as e:
                return {
                    "output": (
                        f"Tentei consultar o item {item_code} na tabela '{discovered_item_table}', "
                        f"mas o banco retornou erro: {e}"
                    ),
                    "sql": sql_item,
                    "rows": [],
                }

            if rows:
                texto = _format_item_answer(rows[0], item_code)
            else:
                texto = (
                    f"Consegui encontrar a tabela de itens ('{discovered_item_table}'), "
                    f"mas o item {item_code} não está nela."
                )

            return {
                "output": texto,
                "sql": sql_item,
                "rows": rows,
            }

        # 2) caso geral → gera SQL com LLM
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
