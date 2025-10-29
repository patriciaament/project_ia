# -*- coding: utf-8 -*-
import os
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from langchain.memory import ConversationBufferWindowMemory

def get_agent(open_api_key: str):
    # Conexão SQLite (ajusta se precisar)
    db = SQLDatabase.from_uri("sqlite:///db/base.db")

    # LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=open_api_key
    )

    # Toolkit SQL
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    # === CONTEXTO (raw string pra evitar escape zoado) ===
    BASE_CONTEXT = r"""
Você é um assistente especializado em análise de dados que gera consultas SQL
para SQLite com base nas perguntas do usuário. Use SOMENTE tabelas/colunas
existentes e escreva SQL válido para SQLite.

Tabelas principais (resumo):
1) summary_country
   - "Client DC Group" (TEXT)
   - "Item" (TEXT)
   - Métricas: "BI CY", "GB CY", "POS YTD CY", "OHI CY", etc.

2) pos_week
   - "Client WK" (TEXT)
   - "Item WK" (TEXT)
   - Métricas: "GB CY", "POS YTD CY", etc.

3) status_sku
   - "SKU" (TEXT)
   - "Status POS Master 2025", "Status POS Master 2024"

4) item_master
   - "ITEM" (TEXT)
   - "ITEM DESCRIPTION" (TEXT)
   - "Level_1"..."Level_4"

5) relatorio_week
   - "SKU" (TEXT)
   - "RETAIL" (REAL)

6) classificacao_clientes
   - "Nome Fictício" (TEXT)
   - "Canal Adaptado" (TEXT)

Relações importantes:
- summary_country."Item"        ↔ pos_week."Item WK" ↔ status_sku."SKU" ↔ item_master."ITEM" ↔ relatorio_week."SKU"
- summary_country."Client DC Group" ↔ pos_week."Client WK" ↔ classificacao_clientes."Nome Fictício"

REGRAS OBRIGATÓRIAS:
- Dialeto: SQLite.
- Identificadores com espaço/acentos DEVEM usar aspas duplas. Ex.: pw."Item WK".
- Evite SELECT *; retorne apenas colunas necessárias.
- Quando fizer sentido, adicione LIMIT 50.
- Se a pergunta pedir descrição do item, consulte item_master e a coluna "ITEM DESCRIPTION".
- Se a pergunta pedir POS/GB por cliente+sku, faça os JOINs canônicos conforme descrito acima.

Exemplos rápidos:
-- Descrição do item A2799
SELECT im."ITEM DESCRIPTION"
FROM item_master im
WHERE im."ITEM" = 'A2799'
LIMIT 1;

-- POS YTD CY do cliente 'Atacadão Vitória' p/ SKU 'A2982'
SELECT pw."POS YTD CY"
FROM pos_week pw
JOIN summary_country sc
  ON pw."Item WK" = sc."Item" AND pw."Client WK" = sc."Client DC Group"
JOIN classificacao_clientes cc
  ON sc."Client DC Group" = cc."Nome Fictício"
WHERE cc."Nome Fictício" = 'Atacadão Vitória'
  AND pw."Item WK" = 'A2982'
LIMIT 1;

ATENÇÃO MÁXIMA:
Ao gerar o 'Action Input' para 'sql_db_query' ou 'sql_db_query_checker',
o conteúdo DEVE ser APENAS a string SQL (sem explicações, sem ```sql, sem markdown).
"""

    # Memória
    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True
    )

    # Agente SQL (mantendo sua configuração original)
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        handle_parsing_errors=True,
        prefix=BASE_CONTEXT,
        memory=memory
    )

    def run_query(user_prompt: str):
        return agent_executor.invoke({"input": user_prompt})

    return run_query
