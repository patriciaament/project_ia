# agent.py
# -*- coding: utf-8 -*-
import os
import re
from typing import Optional, Dict, Any, List

from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain

DB_URI = "sqlite:///db/base.db"

# sku tipo A8350
SKU_RX = re.compile(r"\b([A-Z]\d{3,8})\b", re.I)


def normalize(name: str) -> str:
    """deixa minúsculo e tira espaço/underscore pra comparar nomes de tabela."""
    return re.sub(r"[\s_]+", "", name or "").lower()


def build_table_index(db: SQLDatabase) -> Dict[str, str]:
    """
    Lê TODAS as tabelas do sqlite e cria um índice normalizado -> nome real.
    Assim conseguimos achar 'ITEM MASTER', 'item_master', 'Item Master' etc.
    """
    engine = db._engine
    rows = engine.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    index = {}
    for (tbl,) in rows:
        index[normalize(tbl)] = tbl
    return index


# lista de apelidos que queremos resolver
WANTED_TABLES = {
    "item_master": [
        "item master",
        "item_master",
        "itemmaster",
        "item  master",
    ],
    "summary_by_country": [
        "summary by country",
        "summary_by_country",
        "summarycountry",
        "summary_country",
    ],
    "pos_by_week": [
        "pos by week",
        "pos_by_week",
        "posweek",
    ],
    "status_skus": [
        "status skus",
        "status_skus",
        "statussku",
        "status_sku",
    ],
    "relatorio_week": [
        "relatorio week 2025",
        "relatorio_week_2025",
        "relatorio week",
        "relatorio_week",
    ],
    "classificacao_clientes": [
        "classificacao clientes",
        "classificação clientes",
        "classificacao_clientes",
        "classificacao",
    ],
}


def resolve_tables(db: SQLDatabase) -> Dict[str, str]:
    """
    devolve um dict com o nome lógico -> nome real no sqlite
    ex.: resolved["item_master"] == "ITEM MASTER"
    """
    index = build_table_index(db)
    resolved: Dict[str, str] = {}
    for logical, candidates in WANTED_TABLES.items():
        found = None
        for cand in candidates:
            key = normalize(cand)
            if key in index:
                found = index[key]
                break
        if found:
            resolved[logical] = found
    return resolved


def extract_sku(prompt: str) -> Optional[str]:
    m = SKU_RX.search(prompt or "")
    if m:
        return m.group(1).upper()
    return None


def get_agent(open_api_key: Optional[str] = None):
    api_key = open_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Faltou a OPENAI_API_KEY no ambiente.")

    # 1) conecta no banco e descobre os nomes reais das tabelas
    db = SQLDatabase.from_uri(DB_URI)
    resolved = resolve_tables(db)

    # 2) LLM para os outros casos (quando não for um "traz a descrição do item X")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    fallback_chain = create_sql_query_chain(llm, db, k=3)

    # helpers ----------------------------
    def run_sql_safe(sql: str):
        try:
            return db.run(sql)
        except Exception as e:
            return {"_error": str(e), "_sql": sql}

    def summarize_rows(rows: Any) -> str:
        if isinstance(rows, dict) and "_error" in rows:
            return "Tentei consultar, mas o banco retornou erro. Veja a SQL gerada abaixo."
        if not rows:
            return "Consulta concluída, mas não encontrei registros para esse filtro."
        if isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict):
            r = rows[0]
            # pega algumas colunas mais comuns
            desc = r.get("ITEM DESCRIPTION") or r.get("Item Description") or r.get("descricao") or r.get("description")
            if desc:
                return f"Descrição do item: {desc}"
            # se não tiver uma coluna de descrição clara, mostra tudo
            pairs = "; ".join(f"{k}: {v}" for k, v in r.items())
            return f"Encontrei 1 linha com os seguintes dados: {pairs}"
        # mais de uma linha
        return f"Encontrei {len(rows)} linhas. Veja a SQL gerada abaixo."

    # 3) função principal chamada pelo Streamlit
    def run_query(user_prompt: str) -> Dict[str, Any]:
        prompt_low = (user_prompt or "").lower()
        sku = extract_sku(user_prompt)

        # caso 1: pergunta claramente sobre descrição / item
        if sku and any(x in prompt_low for x in ["descrição", "descricao", "item description", "o que pode me dizer", "dados do item", "info do item"]):
            item_tbl = resolved.get("item_master")
            if not item_tbl:
                # banco não tem essa tabela de jeito nenhum
                return {
                    "output": "Tentei buscar o item, mas o banco não tem uma tabela equivalente ao ITEM MASTER. Confira se ela foi importada para o SQLite.",
                    "sql": None,
                    "rows": [],
                }

            # montamos a query usando o NOME REAL da tabela
            sql = f'''
SELECT
    "ITEM",
    "ITEM DESCRIPTION",
    "Level_1",
    "Level_2",
    "Level_3",
    "Level_4"
FROM "{item_tbl}"
WHERE "ITEM" = '{sku}'
LIMIT 1;
'''.strip()

            rows = run_sql_safe(sql)
            text = summarize_rows(rows)
            return {
                "output": text,
                "sql": sql,
                "rows": rows,
            }

        # caso 2: qualquer outra pergunta → deixa o LLM gerar SQL
        try:
            raw = fallback_chain.invoke({"question": user_prompt})
            sql = (raw or "").strip().strip("`")
            if not sql.lower().startswith("select"):
                return {
                    "output": raw,
                    "sql": None,
                    "rows": [],
                }
            rows = run_sql_safe(sql)
            text = summarize_rows(rows)
            return {
                "output": text,
                "sql": sql,
                "rows": rows,
            }
        except Exception as e:
            return {
                "output": f"Não consegui gerar uma consulta automática: {e}",
                "sql": None,
                "rows": [],
            }

    return run_query
