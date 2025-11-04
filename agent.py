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
# CONFIGURAÇÃO
# ======================================================

DB_URI = "sqlite:///db/base.db"

# nomes reais das tuas tabelas no SQLite
KNOWN_TABLES = [
    "Summary By Country",
    "POS by Week",
    "Status SKUs",
    "ITEM MASTER",
    "Relatório Week 2025",
    "Classificação Clientes",
]

# SKU do tipo A8350, A5460 etc.
SKU_RX = re.compile(r"\b([A-Z]\d{3,8})\b", re.I)


# ======================================================
# HELPERS
# ======================================================

def _extract_sku(prompt: str) -> Optional[str]:
    m = SKU_RX.search(prompt or "")
    return m.group(1).upper() if m else None


def _only_sql(text: str) -> str:
    if not text:
        return ""
    txt = text.strip().strip("`").strip()
    if txt.lower().startswith("select"):
        return txt
    m = re.search(r"(?is)\bselect\b.+", txt, re.DOTALL)
    return m.group(0).strip() if m else txt


def _summarize(rows: Any) -> str:
    if isinstance(rows, dict) and "_error" in rows:
        return "Tentei executar a consulta, mas o banco retornou erro. Veja a SQL gerada."
    if not rows:
        return "Consulta concluída, mas não encontrei linhas para essa pergunta."
    if isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict):
        partes = [f"{k}: {v}" for k, v in rows[0].items()]
        return "Encontrei 1 registro: " + "; ".join(partes) + "."
    if isinstance(rows, list):
        cols = list(rows[0].keys()) if rows and isinstance(rows[0], dict) else []
        return f"Encontrei {len(rows)} linhas. Colunas: {', '.join(cols)}."
    return "Consulta concluída."


# ======================================================
# AGENTE
# ======================================================

def get_agent(open_api_key: Optional[str] = None):
    api_key = open_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Faltou a OPENAI_API_KEY no ambiente.")

    # conexão “grande” (enxerga tudo)
    db = SQLDatabase.from_uri(DB_URI)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)

    # fallback para quando o agente quebrar
    query_chain = create_sql_query_chain(llm, db, k=3)

    # utilitário para rodar SQL
    def run_sql_safe(sql: str):
        try:
            return db.run(sql)
        except Exception as e:
            return {"_error": str(e), "_sql": sql}

    # checar se a tabela existe exatamente com esse nome
    def table_exists(name: str) -> bool:
        check_sql = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{name}';"
        try:
            res = db.run(check_sql)
            return bool(res)
        except Exception:
            return False

    # --------------------------------------------------
    # 1. handler especial: “qual a descrição do item AX...?”
    # --------------------------------------------------
    def try_item_description(prompt: str):
        sku = _extract_sku(prompt)
        if not sku:
            return None

        # frases que indicam que o usuário quer descrição mesmo
        wants_desc = any(
            k in (prompt or "").lower()
            for k in [
                "descrição do item",
                "descricao do item",
                "item description",
                "me diga do item",
                "o que você pode me dizer do item",
                "o que voce pode me dizer do item",
            ]
        )
        if not wants_desc:
            return None

        # tua tabela de itens
        item_table = "ITEM MASTER"
        if not table_exists(item_table):
            # se por acaso o nome no sqlite ficou diferente, mostra isso
            return {
                "output": (
                    f"Tentei buscar na tabela '{item_table}', mas ela não existe no SQLite. "
                    f"Verifique o nome da tabela quando gerou o .db."
                ),
                "sql": None,
                "rows": [],
            }

        # aqui usamos os nomes de colunas iguais ao do seu print
        sql = f'''
SELECT
  "ITEM",
  "ITEM DESCRIPTION",
  "Level_1",
  "Level_2",
  "Level_3",
  "Level_4"
FROM "{item_table}"
WHERE "ITEM" = '{sku}'
LIMIT 1;
'''.strip()

        rows = run_sql_safe(sql)

        if isinstance(rows, dict) and "_error" in rows:
            # deu erro de coluna → talvez o SQLite tenha salvo sem espaço (acontece às vezes)
            # então tentamos sem espaço na coluna de descrição
            sql2 = f'''
SELECT
  "ITEM",
  "Level_1",
  "Level_2",
  "Level_3",
  "Level_4"
FROM "{item_table}"
WHERE "ITEM" = '{sku}'
LIMIT 1;
'''.strip()
            rows2 = run_sql_safe(sql2)
            if isinstance(rows2, dict) and "_error" in rows2:
                return {
                    "output": "Tentei buscar o item no 'ITEM MASTER', mas o banco não aceitou as colunas que eu pedi.",
                    "sql": sql,
                    "rows": rows,
                }
            if rows2:
                r = rows2[0]
                levels = [r.get("Level_1"), r.get("Level_2"), r.get("Level_3"), r.get("Level_4")]
                levels = [x for x in levels if x]
                txt = f"Encontrei o item {sku} na tabela '{item_table}', mas não havia a coluna de descrição. "
                if levels:
                    txt += "Classificação: " + " > ".join(levels) + "."
                return {"output": txt, "sql": sql2, "rows": rows2}
            return {
                "output": f"Não encontrei o item {sku} na tabela '{item_table}'.",
                "sql": sql2,
                "rows": [],
            }

        # se chegou aqui, deu certo e tem a coluna ITEM DESCRIPTION
        if rows:
            r = rows[0]
            desc = r.get("ITEM DESCRIPTION") or "sem descrição cadastrada"
            levels = [r.get("Level_1"), r.get("Level_2"), r.get("Level_3"), r.get("Level_4")]
            levels = [x for x in levels if x]
            txt = f"O item {sku} tem a descrição: {desc}."
            if levels:
                txt += " Classificação: " + " > ".join(levels) + "."
            return {"output": txt, "sql": sql, "rows": rows}

        return {
            "output": f"Não encontrei o item {sku} na tabela '{item_table}'.",
            "sql": sql,
            "rows": [],
        }

    # --------------------------------------------------
    # 2. agentes especializados (AGORA com os nomes certos)
    # --------------------------------------------------
    def make_agent_for(tables: List[str]):
        # todos esses nomes têm espaço, então usamos include_tables com o nome exato
        sub_db = SQLDatabase.from_uri(DB_URI, include_tables=tables)
        toolkit = SQLDatabaseToolkit(db=sub_db, llm=llm)
        base_ctx = f"""
        Você é um gerador de SQL para SQLite.
        Gere APENAS SELECTs.
        Use SOMENTE estas tabelas: {', '.join('\"'+t+'\"' for t in tables)}.
        Não invente nomes de tabela.
        """
        memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)
        return create_sql_agent(
            llm=llm,
            toolkit=toolkit,
            verbose=False,
            handle_parsing_errors=True,
            prefix=base_ctx,
            memory=memory,
            max_iterations=3,
            max_execution_time=15,
        )

    agent_summary = make_agent_for(["Summary By Country"])
    agent_pos = make_agent_for(["POS by Week"])
    agent_status = make_agent_for(["Status SKUs"])
    agent_item = make_agent_for(["ITEM MASTER"])
    agent_relweek = make_agent_for(["Relatório Week 2025"])
    agent_clientes = make_agent_for(["Classificação Clientes"])
    agent_big = make_agent_for(KNOWN_TABLES)

    # roteamento simples
    def route(prompt: str):
        p = prompt.lower()
        if "descri" in p and _extract_sku(prompt):
            return agent_item
        if any(k in p for k in ["pos", "semana", "wk", "últimas", "ultimas"]):
            return agent_pos
        if any(k in p for k in ["status", "tlp", "ntlp", "sku"]):
            return agent_status
        if any(k in p for k in ["estoque", "retail"]):
            return agent_relweek
        if any(k in p for k in ["cliente", "canal"]):
            return agent_clientes
        if any(k in p for k in ["resumo", "country", "visão geral", "visao geral"]):
            return agent_summary
        return agent_big

    # --------------------------------------------------
    # função que o Streamlit chama
    # --------------------------------------------------
    def run_query(user_prompt: str) -> Dict[str, Any]:
        # 1) primeiro tenta o caminho “descrição do item”
        desc_res = try_item_description(user_prompt)
        if desc_res is not None:
            return desc_res

        # 2) senão, passa para o agente
        agent = route(user_prompt)
        try:
            ares = agent.invoke({"input": user_prompt})
            raw = ares.get("output", "")
        except Exception:
            # fallback total
            raw = query_chain.invoke({"question": f"Gere um SELECT para: {user_prompt}"})

        sql_candidate = _only_sql(str(raw))
        if not sql_candidate.lower().startswith("select"):
            return {"output": str(raw), "sql": None, "rows": []}

        rows = run_sql_safe(sql_candidate)
        txt = _summarize(rows)
        return {"output": txt, "sql": sql_candidate, "rows": rows}

    return run_query
