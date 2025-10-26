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
from langchain.chains import create_sql_query_chain


# ========================== UTILS ==========================

_SKU_RX = re.compile(r"\b([A-Z]\d{3,6})\b", re.I)  # ex.: A7171, A2982 etc.

def _extract_sku_and_client(prompt: str):
    """
    Tenta capturar SKU (ex.: A7171) e um possível nome de cliente do prompt.
    Heurística simples: pega o 1º SKU e o trecho entre 'cliente' e próximo '?'/'/'/fim.
    """
    sku = None
    m = _SKU_RX.search(prompt or "")
    if m:
        sku = m.group(1).upper()

    cliente = None
    p = (prompt or "").lower()
    i = p.find("cliente")
    if i >= 0:
        frag = prompt[i:]  # original-case
        # corta em ?, . ou quebra de linha
        j = re.search(r"[?\n\r/]", frag)
        cliente = frag[len("cliente"): j.start()] if j else frag[len("cliente"):]
        cliente = cliente.strip(" :.-").strip()
        # se sobrar vazio, ignora
        if cliente and len(cliente) < 2:
            cliente = None

    return sku, cliente


def _fmt_decimal_brl(x: float, casas=2) -> str:
    try:
        s = f"{float(x):.{casas}f}"
    except Exception:
        return str(x)
    return s.replace(".", ",")


# ========================== AGENT FACTORY ==========================

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

    # ======== CONTEXTO (igual ao anterior, já reforçado) ========
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
- Para top/bottom, use uma única consulta com UNION ALL (com labels).
- Não invente nomes de tabelas/colunas. Use os JOINs canônicos abaixo.

Aliases e chaves:
- s = summary_country  (s."Item", s."Client DC Group")
- pw = pos_week        (pw."Item WK", pw."Client WK")
- im = item_master     (im."ITEM", im."ITEM DESCRIPTION", im."Level_1", im."Level_2", im."Level_3", im."Level_4")
- st = status_sku      (st."SKU", st."Status POS Master 2025")
- rw = relatorio_week  (rw."SKU", rw."RETAIL")
- cc = classificacao_clientes (cc."Nome Fictício", cc."Canal Adaptado")

JOINs canônicos:
- pw."Item WK" = s."Item"
- pw."Client WK" = s."Client DC Group"
- s."Item" = im."ITEM"
- s."Item" = st."SKU"
- s."Item" = rw."SKU"
- s."Client DC Group" = cc."Nome Fictício"

Exemplos (few-shots):
-- NTLP + Barbie (variação absoluta LW, agregada):
SELECT
  SUM(COALESCE(s."POS LW Var$", s."POS LW CY" - s."POS LW LY")) AS "POS LW Var$_total"
FROM summary_country s
JOIN item_master im ON im."ITEM" = s."Item"
WHERE im."Level_1" = 'Barbie'
  AND im."Level_2" = 'NTLP';

-- TLP (variação % YTD agregada correta):
SELECT
  ROUND(
    (SUM(s."POS YTD CY") - SUM(s."POS YTD PY")) * 100.0
    / NULLIF(SUM(s."POS YTD PY"), 0), 2
  ) AS "POS YTD Var%_agregado"
FROM summary_country s
JOIN status_sku st ON st."SKU" = s."Item"
WHERE st."Status POS Master 2025" = 'TLP';

-- POS das últimas 4 semanas (L4W) por CLIENTE + MARCA:
SELECT
  SUM(s."POS L4W CY") AS "POS L4W_total"
FROM summary_country s
JOIN classificacao_clientes cc ON cc."Nome Fictício" = s."Client DC Group"
JOIN item_master im           ON im."ITEM"           = s."Item"
WHERE cc."Nome Fictício" = 'Atacadão Vitória'
  AND im."Level_1"      = 'Hot Wheels';

-- TOP 5 e BOTTOM 5 SKUs (contribuição no L4W) numa única query:
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

    # Agente SQL (com limites)
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        handle_parsing_errors=True,
        prefix=BASE_CONTEXT,
        memory=memory,
        max_iterations=4,
        max_execution_time=35,
    )

    # fallback one-shot
    query_chain = create_sql_query_chain(llm, db, k=5)

    # =============== Sanitizadores de SQL ===============

    _SQL_BLOCK = re.compile(r"(?is)\bselect\b.+", re.DOTALL)

    def _only_sql(text: str) -> str:
        text = (text or "").strip().strip("`").strip()
        if text.lower().startswith("select"):
            return text
        m = _SQL_BLOCK.search(text)
        return m.group(0).strip() if m else text

    def _fix_common_typos(sql: str) -> str:
        return (
            sql.replace("summary_countrys", "summary_country")
               .replace("summary_countries", "summary_country")
               .replace("summary_countrie", "summary_country")
        )

    def _run_sql(sql: str) -> Dict[str, Any]:
        sql = _fix_common_typos(sql)
        try:
            rows = db.run(sql)
            return {"sql": sql, "rows": rows}
        except Exception as e:
            return {"sql": sql, "error": str(e)}

    # =============== Resposta determinística p/ “SKU + cliente” ===============

    def _maybe_answer_stock_status_retail(prompt: str) -> Optional[Dict[str, Any]]:
        """
        Se detectar um SKU + cliente e a pergunta falar de estoque/status/retail,
        roda uma SQL estável e devolve frase no formato pedido.
        """
        lower = (prompt or "").lower()
        gatilhos = any(k in lower for k in [
            "estoque", "stock", "ohi", "variação", "tlp", "ntlp", "retail", "preço", "price"
        ])
        if not gatilhos:
            return None

        sku, cliente = _extract_sku_and_client(prompt)
        if not sku or not cliente:
            return None  # deixa o agente tocar

        # SQL determinística
        sql = f"""
SELECT
  s."OHI CY"                       AS ohi_cy,
  s."OHI Var%"                     AS ohi_var_pct,
  st."Status POS Master 2025"      AS status_2025,
  rw."RETAIL"                      AS retail_price
FROM summary_country s
LEFT JOIN status_sku         st ON st."SKU" = s."Item"
LEFT JOIN relatorio_week     rw ON rw."SKU" = s."Item"
LEFT JOIN classificacao_clientes cc ON cc."Nome Fictício" = s."Client DC Group"
WHERE s."Item" = '{sku}'
  AND (cc."Nome Fictício" = '{cliente}' OR s."Client DC Group" = '{cliente}')
LIMIT 1;
""".strip()

        resp = _run_sql(sql)
        if "rows" not in resp or not resp["rows"]:
            # não achou — devolve o SQL e erro soft
            return {
                "sql": sql,
                "error": "SKU/cliente não encontrado na base para esta consulta."
            }

        row = resp["rows"][0]
        # Normaliza chaves vindas do driver
        def _g(k):  # pega campo com tolerância a case/alias do driver
            for kk in row.keys():
                if kk.lower() == k.lower():
                    return row[kk]
            return None

        ohi = _g("ohi_cy")
        varp = _g("ohi_var_pct")
        status = (_g("status_2025") or "").strip().upper()
        retail = _g("retail_price")

        # Decide TLP/NTLP
        classe = "TLP" if status == "TLP" else "NTLP"  # default NTLP se vier outra coisa

        retail_str = _fmt_decimal_brl(retail, 2) if retail is not None else "0,00"
        # Frase enxuta no padrão pedido:
        frase = f"É {classe}, tem Retail Price de {retail_str}."

        # (Se quiser também incluir estoque e var%):
        # if ohi is not None and varp is not None:
        #     frase += f" Estoque: {_fmt_decimal_brl(ohi, 0)} (variação { _fmt_decimal_brl(varp, 0) }%)."

        return {
            "sql": sql,
            "rows": resp["rows"],
            "text": frase
        }

    # =============== API principal ===============

    def run_query(user_prompt: str) -> Dict[str, Any]:
        # 1) Tenta resposta determinística para perguntas “SKU + cliente”
        det = _maybe_answer_stock_status_retail(user_prompt)
        if det is not None:
            return det

        # 2) Tenta o agente
        try:
            res = agent_executor.invoke({"input": user_prompt})
        except Exception:
            # 3) fallback one-shot
            raw = query_chain.invoke({"question": f"{BASE_CONTEXT}\n\nPergunta do usuário:\n{user_prompt}"})
            sql = _only_sql(raw)
            return _run_sql(sql)

        out = res.get("output", "")
        blocked = isinstance(out, str) and (
            "iteration limit" in out.lower() or "time limit" in out.lower()
        )

        if blocked or not isinstance(out, str) or "select" not in out.lower():
            # fallback one-shot
            raw = query_chain.invoke({"question": f"{BASE_CONTEXT}\n\nPergunta do usuário:\n{user_prompt}"})
            sql = _only_sql(raw)
            return _run_sql(sql)

        sql = _only_sql(out)
        if not sql.lower().startswith("select"):
            raw = query_chain.invoke({"question": f"{BASE_CONTEXT}\n\nPergunta do usuário:\n{user_prompt}"})
            sql = _only_sql(raw)
            return _run_sql(sql)

        return _run_sql(sql)

    return run_query
