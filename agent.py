# agent.py
# -*- coding: utf-8 -*-
import os
import re
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from langchain.memory import ConversationBufferWindowMemory


def get_agent(open_api_key: str | None):
    """
    Mantém sua interface original.
    - Prioriza a chave passada por parâmetro.
    - Se não vier, tenta pegar do ambiente (OPENAI_API_KEY).
    - Se nada existir, dá um erro amigável (sem quebrar import).
    """
    api_key = open_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Faltou a OpenAI API key. Passe via parâmetro em get_agent('<SUA_CHAVE>') "
            "ou defina a variável de ambiente OPENAI_API_KEY."
        )

    # DB (igual ao seu)
    db = SQLDatabase.from_uri("sqlite:///db/base.db")

    # LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key
    )

    # Toolkit
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    # Contexto original (use r-string pra evitar escapes acidentais)
    BASE_CONTEXT = r"""
Você é um assistente especializado em análise de dados que gera consultas SQL
baseadas na base SQLite conectada e traduzir perguntas em consultas SQL.
[... mantenha aqui o seu dicionário/descrição das tabelas exatamente como já estava ...]
A REGRA MAIS IMPORTANTE:
Ao gerar o 'Action Input' para a ferramenta 'sql_db_query' ou 'sql_db_query_checker',
o SQL DEVE ser EXATAMENTE a string da consulta SQL pura (sem markdown, sem explicações).
"""

    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True
    )

    # Agente (mantendo sua criação original)
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        handle_parsing_errors=True,
        prefix=BASE_CONTEXT,
        memory=memory,
        # Se sua versão do langchain suportar, você pode testar:
        # use_query_checker=True,
        # agent_type="openai-tools",
    )

    # --- Sanitizador opcional: se a saída vier com prosa + SQL, extrai só o SELECT e executa ---
    _SQL_BLOCK = re.compile(r"(?is)\bselect\b.+", re.DOTALL)

    def _only_sql(text: str) -> str:
        text = (text or "").strip().strip("`").strip()
        if text.lower().startswith("select"):
            return text
        m = _SQL_BLOCK.search(text)
        return m.group(0).strip() if m else text

    def run_query(user_prompt: str):
        """
        Mantém seu contrato original.
        1) Tenta usar o agente normalmente.
        2) Se a saída final vier com texto + SQL, extraímos o SELECT e executamos direto no DB.
        """
        res = agent_executor.invoke({"input": user_prompt})
        out = res.get("output", "")

        # Se a saída parece conter SQL, tentamos rodar direto no DB (robustez contra parsing errors)
        if isinstance(out, str) and "select" in out.lower():
            sql = _only_sql(out)
            if sql.lower().startswith("select"):
                try:
                    rows = db.run(sql)
                    return {"sql": sql, "rows": rows, "agent_output": out}
                except Exception as e:
                    return {"sql": sql, "error": str(e), "agent_output": out}

        return res

    return run_query
