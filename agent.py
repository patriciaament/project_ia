# agent.py
# -*- coding: utf-8 -*-
import os
import re
from typing import List, Dict, Callable, Optional

from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent

DB_URI = "sqlite:///db/base.db"

def make_sql_agent(db_uri: str, include_tables: List[str], name: str, api_key: str) -> Callable[[str], Dict]:
    """Cria um agente ‘fechado’ às include_tables e retorna um runner run(prompt)->dict."""
    db  = SQLDatabase.from_uri(db_uri, include_tables=include_tables)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

    base_ctx = f"""
Você é um gerador de SQL para SQLite. Responda APENAS com a consulta SQL (apenas SELECT).
Você só pode usar as tabelas: {', '.join(include_tables)}.
Aspas duplas em colunas com espaço/acentos (ex.: s."Client DC Group"). Nada de explicações.
"""
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        prefix=base_ctx,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=4,
        max_execution_time=25,
        early_stopping_method="generate",
    )

    def run(prompt: str) -> Dict:
        res = agent.invoke({"input": prompt})
        sql = (res.get("output") or "").strip().strip("`").replace("```sql","").replace("```","")
        try:
            rows = db.run(sql)
            return {"agent": name, "sql": sql, "rows": rows, "output": f"OK - {name}"}
        except Exception as e:
            return {"agent": name, "sql": sql, "rows": [], "output": f"Erro ao executar SQL: {e}"}

    return run

def get_agent(open_api_key: Optional[str] = None):
    """
    Constrói e retorna um roteador de agentes (função run_router(prompt)->dict).
    Só inicializa LLMs depois de ter a API key.
    """
    # 1) resolve a key (env, secrets, arg)
    api_key = (
        open_api_key
        or os.getenv("OPENAI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        # tenta streamlit.secrets se existir
        try:
            import streamlit as st  # type: ignore
            api_key = st.secrets.get("OPENAI_API_KEY")  # pyright: ignore
        except Exception:
            pass
    if not api_key:
        raise ValueError("OPENAI_API_KEY não encontrada. Defina no ambiente/Secrets ou passe para get_agent().")

    # 2) constrói os agentes especializados AQUI (não no topo do arquivo)
    agent_summary       = make_sql_agent(DB_URI, ["summary_country"], name="AG_SUMMARY",       api_key=api_key)
    agent_posweek       = make_sql_agent(DB_URI, ["pos_week"],         name="AG_POSWEEK",       api_key=api_key)
    agent_item          = make_sql_agent(DB_URI, ["item_master"],      name="AG_ITEM",          api_key=api_key)
    agent_status        = make_sql_agent(DB_URI, ["status_sku"],       name="AG_STATUS",        api_key=api_key)
    agent_relweek       = make_sql_agent(DB_URI, ["relatorio_week"],   name="AG_RELWEEK",       api_key=api_key)
    agent_summary_item  = make_sql_agent(DB_URI, ["summary_country","item_master"], name="AG_SUMMARY_ITEM", api_key=api_key)

    ROUTES = [
        (lambda p: re.search(r"\bretail|preço|price\b", p, re.I),                  agent_relweek),
        (lambda p: re.search(r"\b(status|tlp|ntlp)\b", p, re.I),                  agent_status),
        (lambda p: re.search(r"\bpos\b.*\bsemana|\bweek\b", p, re.I),             agent_posweek),
        (lambda p: re.search(r"\bdescrição|description|marca|level_\d\b", p, re.I), agent_item),
        (lambda p: re.search(r"\bpos\b.*(4 semanas|l4w)|\bmarca|linha\b", p, re.I), agent_summary_item),
        (lambda p: True, agent_summary),  # fallback
    ]

    def run_router(prompt: str) -> Dict:
        p = (prompt or "").strip()
        for cond, ag in ROUTES:
            if cond(p):
                return ag(p)
        return agent_summary(p)

    return run_router
