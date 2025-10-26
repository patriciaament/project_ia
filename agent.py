# agent.py
# -*- coding: utf-8 -*-
import os
import re
from typing import Optional, Dict, Any

from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from langchain.memory import ConversationBufferWindowMemory


def get_agent(open_api_key: Optional[str]):
    """
    Mantém sua interface original.
    - Prioriza a chave passada por parâmetro.
    - Se não vier, tenta a variável de ambiente OPENAI_API_KEY.
    """
    api_key = open_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Faltou a OpenAI API key. Passe via get_agent('<SUA_CHAVE>') "
            "ou defina OPENAI_API_KEY no ambiente."
        )

    # Conexão SQLite
    db = SQLDatabase.from_uri("sqlite:///db/base.db")

    # LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=api_key,
    )

    # Toolkit SQL
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    # ======== CONTEXTO REFORÇADO (raw string) ========
    BASE_CONTEXT = r"""
Você é um gerador de SQL para **SQLite**. Sua resposta ao acionar a ferramenta deve ser
APENAS a string SQL (sem markdown, sem explicações, sem ```sql).

REGRAS DURAS (NÃO QUEBRE):
- Use APENAS estes nomes de tabela, exatamente assim (sem pluralizar/abreviar):
  summary_country, pos_week, status_sku, item_master, relatorio_week, classificacao_clientes
- Sempre use **aspas duplas** em colunas com espaço/acentos: s."POS YTD Var%", pw."Item WK", s."Client DC Group".
- Evite SELECT *; traga só colunas necessárias. Quando listar muitos registros, adicione LIMIT.
- Se precisar de **variação absoluta de POS na LW**:
  1) Primeiro tente s."POS LW Var$".
  2) Se a coluna não existir, calcule:  (s."POS LW CY" - s."POS LW LY")  como "POS LW Var$".
- Para **variação % agregada** (YTD), calcule com SOMA:
    ROUND((SUM(s."POS YTD CY") - SUM(s."POS YTD PY")) * 100.0 / NULLIF(SUM(s."POS YTD PY"), 0), 2)
  e NÃO como média de percentuais.
- "NTLP" = categoria/linha em item_master (geralmente im."Level_2").
- "Barbie" = marca/linha em item_master (geralmente im."Level_1").
- "TLP" = classe em status_sku (st."Status POS Master 2025" = 'TLP').
- "L4W" = últimas 4 semanas → use s."POS L4W CY" quando disponível.
- Se a pergunta pedir **top/bottom**, retorne **UMA única consulta SQL** usando `UNION ALL` com labels (‘TOP 5’, ‘BOTTOM 5’).
- Não invente nomes de tabelas/colunas. Se ficar em dúvida, use os JOINs canônicos abaixo.

Aliases e chaves (recomendado):
- s = summary_country
  - chaves: s."Item", s."Client DC Group"
- pw = pos_week
  - chaves: pw."Item WK", pw."Client WK"
- im = item_master
  - chaves: im."ITEM", im."ITEM DESCRIPTION", im."Level_1", im."Level_2", im."Level_3", im."Level_4"
- st = status_sku
  - chaves: st."SKU", st."Status POS Master 2025"
- cc = classificacao_clientes
  - chaves: cc."Nome Fictício", cc."Canal Adaptado"

JOINs canônicos:
- pw."Item WK" = s."Item"
- pw."Client WK" = s."Client DC Group"
- s."Item" = im."ITEM"
- s."Item" = st."SKU"
- s."Client DC Group" = cc."Nome Fictício"

Exemplos (few-shots):

-- (1) Descrição do item:
SELECT im."ITEM DESCRIPTION"
FROM item_master im
WHERE im."ITEM" = 'A2799'
LIMIT 1;

-- (2) POS YTD CY por cliente e SKU (JOIN canônico):
SELECT pw."POS YTD CY"
FROM pos_week pw
JOIN summary_country s
  ON pw."Item WK" = s."Item" AND pw."Client WK" = s."Client DC Group"
JOIN classificacao_clientes cc
  ON s."Client DC Group" = cc."Nome Fictício"
WHERE cc."Nome Fictício" = 'Atacadão Vitória'
  AND pw."Item WK" = 'A2982'
LIMIT 1;

-- (3) NTLP + Barbie (variação absoluta na LW, agregada):
SELECT
  SUM(COALESCE(s."POS LW Var$", s."POS LW CY" - s."POS LW LY")) AS "POS LW Var$_total"
FROM summary_country s
JOIN item_master im ON im."ITEM" = s."Item"
WHERE im."Level_1" = 'Barbie'
  AND im."Level_2" = 'NTLP';

-- (4) TLP (variação % YTD agregada correta):
SELECT
  ROUND(
    (SUM(s."POS YTD CY") - SUM(s."POS YTD PY")) * 100.0
    / NULLIF(SUM(s."POS YTD PY"), 0), 2
  ) AS "POS YTD Var%_agregado"
FROM summary_country s
JOIN status_sku st ON st."SKU" = s."Item"
WHERE st."Status POS Master 2025" = 'TLP';

-- (5) POS das últimas 4 semanas (L4W) por CLIENTE + MARCA:
SELECT
  SUM(s."POS L4W CY") AS "POS L4W_total"
FROM summary_country s
JOIN classificacao_clientes cc ON cc."Nome Fictício" = s."Client DC Group"
JOIN item_master im           ON im."ITEM"           = s."Item"
WHERE cc."Nome Fictício" = 'Atacadão Vitória'
  AND im."Level_1"      = 'Hot Wheels';

-- (6) TOP 5 e BOTTOM 5 SKUs (contribuição no L4W) numa única query:
SELECT 'TOP 5' AS faixa, s."Item" AS sku, im."ITEM DESCRIPTION" AS descricao, s."POS L4W CY" AS pos_l4w
FROM summary_country s
JOIN classificacao_clientes cc ON cc."Nome Fictício" = s."Client DC Group"
JOIN item_master im           ON im."ITEM"           = s."Item"
WHERE cc."Nome Fictício" = 'Atacadão Vitória'
  AND im."Level_1"      = 'Hot Wheels'
ORDER BY s."POS L4W CY" DESC
LIMIT 5
UNION ALL
SELECT 'BOTTOM 5' AS faixa, s."Item" AS sku, im."ITEM DESCRIPTION" AS descricao, s."POS L4W CY" AS pos_l4w
FROM summary_country s
JOIN classificacao_clientes cc ON cc."Nome Fictício" = s."Client DC Group"
JOIN item_master im           ON im."ITEM"           = s."Item"
WHERE cc."Nome Fictício" = 'Atacadão Vitória'
  AND im."Level_1"      = 'Hot Wheels'
ORDER BY s."POS L4W CY" ASC
LIMIT 5;

A REGRA MAIS IMPORTANTE:
Ao gerar o 'Action Input' para 'sql_db_query' ou 'sql_db_query_checker',
o input deve ser EXATAMENTE a string da consulta SQL pura (apenas SELECT).
"""

    # Memória
    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True
    )

    # Agente SQL
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        handle_parsing_errors=True,
        prefix=BASE_CONTEXT,
        memory=memory,
        # Se disponível na sua versão do langchain, pode testar:
        # agent_type="openai-tools",
        # use_query_checker=True,
        # max_iterations=4,
        # max_execution_time=40,
    )

    # ================== UTILITÁRIOS DE ROBUSTEZ ==================

    # regex pra extrair só o SELECT se vier texto junto
    _SQL_BLOCK = re.compile(r"(?is)\bselect\b.+", re.DOTALL)

    def _only_sql(text: str) -> str:
        text = (text or "").strip().strip("`").strip()
        if text.lower().startswith("select"):
            return text
        m = _SQL_BLOCK.search(text)
        return m.group(0).strip() if m else text

    # correções bobas de pluralização que o LLM às vezes inventa
    def _fix_common_typos(sql: str) -> str:
        return (
            sql.replace("summary_countrys", "summary_country")
               .replace("summary_countries", "summary_country")
               .replace("summary_countrie", "summary_country")
        )

    # ================== API DE EXECUÇÃO ==================
    def run_query(user_prompt: str) -> Dict[str, Any]:
        """
        Executa o agente. Se a saída vier com prosa + SQL, sanitiza e roda direto no DB.
        Retorna:
          - {"sql": "...", "rows": [...]} em caso de sucesso
          - {"sql": "...", "error": "..."} em erro de execução
          - ou o dicionário original do agente (quando não houver SQL claro)
        """
        res = agent_executor.invoke({"input": user_prompt})
        out = res.get("output", "")

        if isinstance(out, str) and "select" in out.lower():
            sql = _only_sql(out)
            if sql.lower().startswith("select"):
                sql = _fix_common_typos(sql)
                try:
                    rows = db.run(sql)
                    return {"sql": sql, "rows": rows}
                except Exception as e:
                    return {"sql": sql, "error": str(e)}

        return res  # mantém retorno do agente quando não é SQL

    return run_query
