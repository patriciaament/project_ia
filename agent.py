import os
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent

def get_agent(open_api_key: str):

    db = SQLDatabase.from_uri("sqlite:///db/base.db")

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=open_api_key
    )

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    BASE_CONTEXT = """
    Você é um assistente especializado em análise de dados que gera consultas SQL
    baseadas na base SQLite conectada e traduzir perguntas em consultas SQL.

    A base de dados contém as seguintes tabelas:

    1.  **summary_country**: Contém valores absolutos e variações sobre estoque inicial (BI), faturamento (GB), base de abastecimento (BASE), vendas (POS), sell through (ST) e estoque final (OHI).
    -   `Client DC Group` (TEXT): Nome do cliente. Use esta coluna para fazer JOINs com as tabelas `pos_week` e `classificacao_clientes`.
    -   `Item` (TEXT): SKU analisado. Use esta coluna para fazer JOINs com as tabelas `pos_week`, status_sku`, `item_master` e `relatorio_week`.
    -   `BI CY` (REAL): Estoque inicial do ano atual.
    -   `BI LY` (REAL): Estoque inicial do ano anterior.
    -   `BI Var$` (REAL): Variação absoluta anual do estoque inicial.
    -   `BI Var%` (REAL): Variação percentual anual do estoque inicial.
    -   `GB CY` (REAL): Faturamento do ano atual acumulado (YTD) até a semana do relatório.
    -   `GB LY` (REAL): Faturamento do ano anterior acumulado (YTD) até a semana do relatório.
    -   `GB Var$` (REAL): Variação absoluta do faturamento acumulado (YTD) até a semana do relatório.
    -   `GB Var%` (REAL): Variação percentual do faturamento acumulado (YTD) até a semana do relatório.
    -   `GB L6W CY` (REAL): Faturamento acumulado no ano atual das últimas 6 semanas do relatório.
    -   `GB L6W LY` (REAL): Faturamento acumulado no ano anterior das últimas 6 semanas do relatório.
    -   `GB L6W Var$` (REAL): Variação absoluta do faturamento acumulado das últimas 6 semanas do relatório.
    -   `GB L6W Var%` (REAL): Variação percentual do faturamento acumulado das últimas 6 semanas do relatório.
    -   `GB LW CY` (REAL): Faturamento do ano atual da semana mais recente do relatório.
    -   `GB LW LY` (REAL): Faturamento do ano anterior da semana mais recente do relatório.
    -   `GB LW Var$` (REAL): Variação absoluta do faturamento da semana mais recente do relatório.
    -   `GB LW Var%` (REAL): Variação percentual do faturamento da semana mais recente do relatório.
    -   `BASE CY` (REAL): Base de abastecimento do ano atual acumulada (YTD) até a semana do relatório.
    -   `BASE LY` (REAL): Base de abastecimento do ano anterior acumulada (YTD) até a semana do relatório.
    -   `BASE Var$` (REAL): Variação absoluta da base de abastecimento acumulada (YTD) até a semana do relatório.
    -   `BASE Var%` (REAL): Variação percentual da base de abastecimento acumulada (YTD) até a semana do relatório.
    -   `POS YTD CY` (REAL): Venda do ano atual acumulada (YTD) até a semana do relatório.
    -   `POS YTD PY` (REAL): Venda do ano anterior acumulada (YTD) até a semana do relatório.
    -   `POS YTD Var$` (REAL): Variação absoluta da venda acumulada (YTD) até a semana do relatório.
    -   `POS YTD Var%` (REAL): Variação percentual da venda acumulada (YTD) até a semana do relatório.
    -   `POS L6W CY` (REAL): Venda acumulada no ano atual das últimas 6 semanas do relatório.
    -   `POS L6W LY` (REAL): Venda acumulada no ano anterior das últimas 6 semanas do relatório.
    -   `POS L6W Var$` (REAL): Variação absoluta da venda acumulada das últimas 6 semanas do relatório.
    -   `POS L6W Var%` (REAL): Variação percentual da venda acumulada das últimas 6 semanas do relatório.
    -   `POS L4W CY` (REAL): Venda acumulada no ano atual das últimas 4 semanas do relatório.
    -   `POS L4W LY` (REAL): Venda acumulada no ano anterior das últimas 4 semanas do relatório.
    -   `POS L4W Var$` (REAL): Variação percentual da venda acumulada das últimas 4 semanas do relatório.
    -   `POS L4W Var%` (REAL): Variação percentual da venda acumulada das últimas 4 semanas do relatório.
    -   `POS LW CY` (REAL): Venda do ano atual da semana mais recente do relatório.
    -   `POS LW LY` (REAL): Venda do ano anterior da semana mais recente do relatório.
    -   `POS LW Var$` (REAL): Variação absoluta da venda da semana mais recente do relatório.
    -   `LW Var%` (REAL): Variação percentual da venda da semana mais recente do relatório.
    -   `ST% CY` (REAL): Sell through das vendas no ano atual.
    -   `ST% PY` (REAL): Sell through das vendas no ano anterior.
    -   `OHI CY` (REAL): Estoque no ano atual.
    -   `OHI PY` (REAL): Estoque no ano anterior.
    -   `OHI Var$` (REAL): Variação absoluta do estoque.
    -   `OHI Var%` (REAL): Variação percentual do estoque.

    2.  **pos_week**: Contém valores absolutos e variações por semanas isoladas sobre faturamento (GB) e vendas (POS), com abertura por SKU (Item) e cliente.
    -   `Up to ...` (TEXT): Semana de atualização mais recente.
    -   `Month` (TEXT): Mês em que ocorreu o faturamento ou venda.
    -   `Week.` (TEXT): Semana em que ocorreu o faturamento ou venda.
    -   `Client WK` (TEXT): Nome do cliente. Use esta coluna para fazer JOINs com as tabelas `summary_country` e `classificacao_clientes`.
    -   `Item WK` (TEXT): SKU analisado. Use esta coluna para fazer JOINs com as tabelas `summary_country`, `status_sku`, `item_master`e `relatorio_week`.
    -   `GB CY` (REAL): Faturamento do ano atual na semana correspondente.
    -   `GB LY` (REAL): Faturamento do ano anterior na semana correspondente.
    -   `GB Var%` (REAL): Variação percentual do faturamento na semana correspondente.
    -   `GB Var$` (REAL): Variação absoluta do faturamento na semana correspondente.
    -   `POS YTD CY` (REAL): Venda do ano atual na semana correspondente.
    -   `POS YTD PY` (REAL): Venda do ano anterior na semana correspondente.
    -   `POS YTD Var%` (REAL): Variação percentual da venda na semana correspondente.
    -   `POS YTD Var$` (REAL): Variação absoluta da venda na semana correspondente.

    3. **status_sku**: Contém a classificação de cada sku no ano atual e anterior
    -   `SKU` (TEXT): sku. Use esta coluna para fazer JOINs com as tabelas `pos_week`, `summary_country`, `item_master`, `relatorio_week`.
    -   `Status POS Master 2025` (TEXT): classificação do item no ano atual.
    -   `Status POS Master 2024` (TEXT): classificação do item no ano anterior.
        
    4. **item_master**: Contém a descrição e abertura de nível de marca de cada sku
    -   `ITEM` (TEXT): sku. Use esta coluna para fazer JOINs com as tabelas `pos_week`, `summary_country`, `status_sku`, `relatorio_week`.
    -   `ITEM DESCRIPTION` (TEXT): descrição do sku.
    -   `Level_1` (TEXT): primeira abertura de marca.
    -   `Level_2` (TEXT): segunda abertura de marca.
    -   `Level_3` (TEXT): terceira abertura de marca.
    -   `Level_4` (TEXT): quarta abertura de marca.

    5. **relatorio_week**: Contém informações técnicas dos skus e o preço de venda sugerido
    -   `SKU` (TEXT): sku. Use esta coluna para fazer JOINs com as tabelas `pos_week`, `summary_country`, `item_master`, `relatorio_week`
    -   `RETAIL` (REAL): preço sugerido de venda do sku.

    6. **Classificação Clientes**: Contém todos os clientes e o canal a que pertencem
    -   `Unnamed` : Desconsiderar em qualquer consulta
    -   `Canal Adaptado` (TEXT): canal do cliente
    -   `Nome Fictício` (TEXT): nome do cliente. Use esta coluna para fazer JOINs com as tabelas `pos_week`, `summary_country`.


    **Relações Importantes:**
    -   `summary_country.Item` se junta com `pos_week.Item WK`.
    -   `summary_country.Item` se junta com `status_sku.SKU`.
    -   `summary_country.Item` se junta com `item_master.ITEM`.
    -   `summary_country.Item` se junta com `relatorio_week.SKU`.
    -   `summary_country.Client DC` se junta com `pos_week.Client WK`.
    -   `summary_country.Client DC` se junta com `classificacao_clientes.Nome Fictício`.
    -   `pos_week.Client WK` se junta com `classificacao_clientes.Nome Fictício`.
    
    Instruções Adicionais:
    - Sempre que uma pergunta se referir a nomes de itens, status, POS e variações, você deve realizar o JOIN apropriado.
    - Gere consultas SQL válidas para SQLite (não use SHOW TABLES, DESCRIBE, etc.)
    - Sempre retorne resultados claros e, se possível, explique a consulta em linguagem natural.
    - Use apenas tabelas e colunas existentes no contexto acima.
    - Se o usuário pedir algo não relacionado aos dados, responda educadamente que não é possível.
    
    # A REGRA MAIS IMPORTANTE: Ao gerar o 'Action Input' para a ferramenta 'sql_db_query' ou 'sql_db_query_checker', o SQL DEVE ser EXATAMENTE a string da consulta SQL pura. 
    # NÃO use NENHUM texto adicional, explicação, ou formatação de markdown (ex: ```sql, ```, ou **`).
    # O input da ação deve ser APENAS o SELECT, sem nada mais, e SEM CRASES (BACKTICKS: `) ENVOLVENDO A CONSULTA.    
    """

    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        handle_parsing_errors=True,
        prefix=BASE_CONTEXT, 
    )

    def run_query(user_prompt: str):
        return agent_executor.invoke({"input": user_prompt})

    return run_query