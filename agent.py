import os
import re
import pandas as pd
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from langchain.memory import ConversationBufferWindowMemory

# ========================
# CONFIGURAÇÕES INICIAIS
# ========================

# Carrega o .env (onde fica sua chave da OpenAI)
load_dotenv()

# URI do banco local SQLite
DB_URI = "sqlite:///db/base.db"

# Obtém a chave da OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Defina OPENAI_API_KEY no ambiente (.env).")


# ========================
# FUNÇÃO PARA CRIAR AGENTE SQL
# ========================
def make_sql_agent(db_uri, include_tables, name, api_key):
    """Cria um agente SQL restrito a certas tabelas"""
    db = SQLDatabase.from_uri(db_uri, include_tables=include_tables)

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key,
    )

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    base_ctx = f"""
    Você é um analista de dados especialista em SQL e negócios.
    Gere consultas SQL válidas e otimizadas (SQLite).
    Use SOMENTE as tabelas: {", ".join(include_tables)}.
    Retorne apenas SQL bem formatado e explique o resultado
    em português claro e objetivo — tipo “É TLP, o preço é 16,99”.
    """

    memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history", return_messages=True)

    agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        prefix=base_ctx,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=3,
        max_execution_time=15,
        memory=memory,
    )

    return agent


# ========================
# AGENTES SEPARADOS
# ========================
agent_summary = make_sql_agent(DB_URI, ["summary_country"], "AG_SUMMARY", OPENAI_API_KEY)
agent_posweek = make_sql_agent(DB_URI, ["pos_week"], "AG_POSWEEK", OPENAI_API_KEY)
agent_item = make_sql_agent(DB_URI, ["item_master"], "AG_ITEM", OPENAI_API_KEY)
agent_status = make_sql_agent(DB_URI, ["status_sku"], "AG_STATUS", OPENAI_API_KEY)
agent_relweek = make_sql_agent(DB_URI, ["relatorio_week"], "AG_RELWEEK", OPENAI_API_KEY)
agent_summary_item = make_sql_agent(DB_URI, ["summary_country", "item_master"], "AG_SUMMARY_ITEM", OPENAI_API_KEY)


# ========================
# ROTEADOR DE AGENTES
# ========================
def route_prompt(prompt: str):
    """Seleciona o agente com base no conteúdo do prompt"""
    prompt = prompt.lower()

    if re.search(r"retail|price|preço", prompt):
        return agent_relweek
    elif re.search(r"status|tlp|ntlp", prompt):
        return agent_status
    elif re.search(r"pos|venda|semana", prompt):
        return agent_posweek
    elif re.search(r"descrição|marca|level", prompt):
        return agent_item
    elif re.search(r"estoque|ohi|variação", prompt):
        return agent_summary_item
    else:
        return agent_summary


# ========================
# EXECUÇÃO E INTERPRETAÇÃO
# ========================
def get_agent_response(prompt: str):
    """Executa a query SQL, obtém resultado e explica"""
    agent = route_prompt(prompt)

    try:
        # 1️⃣ Gera SQL a partir do prompt
        result = agent.invoke({"input": prompt})
        sql = result["output"]

        # 2️⃣ Executa a consulta no banco
        db = SQLDatabase.from_uri(DB_URI)
        df = pd.read_sql_query(sql, db._engine)

        # 3️⃣ Cria explicação natural do resultado
        llm_explainer = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.5,
            api_key=OPENAI_API_KEY,
        )

        explanation_prompt = f"""
        Explique o resultado dessa consulta SQL em linguagem natural e concisa.
        A consulta gerada foi:
        {sql}

        Os primeiros resultados da tabela:
        {df.head(10).to_markdown(index=False)}

        Pergunta original do usuário:
        "{prompt}"

        Seja analítico, direto e claro. Exemplo:
        “O SKU A7171 é TLP e o preço sugerido é 16,99.”
        """

        explanation = llm_explainer.invoke(explanation_prompt)
        return sql, df, explanation.content

    except Exception as e:
        return None, None, f"⚠️ Erro ao processar consulta: {str(e)}"
