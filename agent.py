# agents.py
# pip install langchain langchain-openai langchain-community

import os, re
from typing import List, Dict, Callable
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # ou passe direto no ChatOpenAI

# ---- fábrica de agentes SQL “fechados” por tabela ----
def make_sql_agent(db_uri: str, include_tables: List[str], name: str) -> Callable[[str], Dict]:
    """
    Cria um agente que só enxerga 'include_tables'.
    Retorna uma função run(prompt) -> dict(output=..., sql=..., rows=...)
    """
    db = SQLDatabase.from_uri(db_uri, include_tables=include_tables)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)

    base_ctx = f"""
Você é um gerador de SQL para SQLite. Responda APENAS com a consulta SQL (apenas SELECT).
Você só pode usar as tabelas: {', '.join(include_tables)}.
Aspas duplas em colunas com espaço/acentos (ex.: s."Client DC Group").
Nada de explicações, somente SQL válida.
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
        # 1) pede a SQL pura
        res = agent.invoke({"input": prompt})
        sql = (res.get("output") or "").strip().strip("`")
        # 2) executa no próprio DB “fechado”
        try:
            rows = db.run(sql)
            return {"agent": name, "sql": sql, "rows": rows, "output": f"OK - {name}"}
        except Exception as e:
            return {"agent": name, "sql": sql, "rows": [], "output": f"Erro: {e}"}

    return run

# ---- defina seus agentes especializados ----
DB_URI = "sqlite:///db/base.db"

agent_summary  = make_sql_agent(DB_URI, ["summary_country"],         name="AG_SUMMARY")
agent_posweek  = make_sql_agent(DB_URI, ["pos_week"],                 name="AG_POSWEEK")
agent_item     = make_sql_agent(DB_URI, ["item_master"],              name="AG_ITEM")
agent_status   = make_sql_agent(DB_URI, ["status_sku"],               name="AG_STATUS")
agent_relweek  = make_sql_agent(DB_URI, ["relatorio_week"],           name="AG_RELWEEK")
# se precisar de join entre 2 tabelas específicas, crie um agente “dual”
agent_summary_item = make_sql_agent(DB_URI, ["summary_country","item_master"], name="AG_SUMMARY_ITEM")

# ---- router determinístico e simples ----
ROUTES = [
    # (regex/condição, agente)
    (lambda p: re.search(r"\bretail|preço|price\b", p, re.I), agent_relweek),
    (lambda p: re.search(r"\b(status|tlp|ntlp)\b", p, re.I),   agent_status),
    (lambda p: re.search(r"\bpos\b.*\bsemana|\bweek\b", p, re.I), agent_posweek),
    (lambda p: re.search(r"\bdescrição|description|marca|level_\d\b", p, re.I), agent_item),
    # join típico: POS L4W de uma linha/marca => precisa summary + item_master
    (lambda p: re.search(r"\bpos\b.*(4 semanas|l4w)|\bmarca|linha\b", p, re.I), agent_summary_item),
    # fallback geral pra métricas acumuladas
    (lambda p: True, agent_summary),
]

def run_router(prompt: str) -> Dict:
    p = prompt.strip()
    for cond, ag in ROUTES:
        if cond(p):
            return ag(p)
    # nunca chega aqui por causa do fallback True
