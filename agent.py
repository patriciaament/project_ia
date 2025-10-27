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

_SKU_RX = re.compile(r"\b([A-Z]\d{3,8})\b", re.I)  # ex.: A7171

_STOP_TOKENS = [
    " em ", " no ", " na ", " de ", " do ", " da ", " para ", " por ", " com ",
    " que ", " e ", " ou ", " onde ", " quando ",
]
_STOP_PUNCT = r"[\,\.\?\:\;\!\|/()\[\]\n\r\t]"

def _extract_sku_and_client(prompt: str):
    """
    Captura SKU e cliente com heurísticas robustas:
    - SKU: 1ª ocorrência tipo A7171
    - Cliente: após 'cliente', preferindo ENTRE ASPAS. Se sem aspas, corta por stopwords/pontuação.
    """
    sku = None
    m = _SKU_RX.search(prompt or "")
    if m:
        sku = m.group(1).upper()

    cliente = None
    p = prompt or ""
    p_low = p.lower()

    idx = p_low.find("cliente")
    if idx >= 0:
        rest = p[idx + len("cliente"):].lstrip()

        # 1) Tenta aspas duplas
        if rest.startswith('"'):
            m = re.search(r'^"([^"]+)"', rest)
            if m:
                cliente = m.group(1).strip()

        # 2) Tenta aspas simples
        if not cliente and rest.startswith("'"):
            m = re.search(r"^'([^']+)'", rest)
            if m:
                cliente = m.group(1).strip()

        # 3) Sem aspas: corta por stopwords/pontuação
        if not cliente:
            m = re.search(_STOP_PUNCT, rest)
            cut = rest[: m.start()] if m else rest

            cut_low = " " + cut.lower() + " "
            min_pos = None
            for tok in _STOP_TOKENS:
                pos = cut_low.find(tok)
                if pos != -1:
                    pos = pos - 1
                    if min_pos is None or pos < min_pos:
                        min_pos = pos
            if min_pos is not None:
                cut = cut[:min_pos]

            cliente = cut.strip(" :.-").strip()

        if cliente:
            parts = cliente.split()
            if len(parts) > 6:
                cliente = " ".join(parts[:6]).strip()

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

    # ======== CONTEXTO ========
    BASE_CONTEXT = r"""
Você é um gerador de SQL para **SQLite**. Sua resposta ao acionar a ferramenta deve ser
APENAS a string SQL (sem markdown, sem explicações, sem ```sql).

REGRAS DURAS:
- Use APENAS estes nomes de tabela: summary_country, pos_week, status_sku, item_master, relatorio_week, classificacao_clientes
- Aspas duplas em colunas com espaço/acentos: s."POS YTD Var%", pw."Item WK", s."Client DC Group".
- Evite SELECT *; traga só colunas necessárias.
- Variação POS LW: prefira s."POS LW Var$"; senão (s."POS LW CY" - s."POS LW LY").
- Agregação de % (YTD): ROUND((SUM(s."POS YTD CY") - SUM(s."POS YTD PY"))*100.0/NULLIF(SUM(s."POS YTD PY"),0),2)
- JOINs canônicos:
  s."Item" = st."SKU" = im."ITEM" = rw."SKU"; s."Client DC Group" = cc."Nome Fictício"; pw."Item WK" = s."Item"; pw."Client WK" = s."Client DC Group"
A REGRA MAIS IMPORTANTE:
O input da ação deve ser EXATAMENTE a string da consulta SQL pura (apenas SELECT).
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

    def _run_sql(sql: str):
        sql = _fix_common_typos(sql)
        return db.run(sql)

    # =============== Resposta determinística p/ “SKU + cliente” ===============

    def _maybe_answer_stock_status_retail(prompt: str) -> Optional[Dict[str, Any]]:
        """
        Se detectar um SKU + cliente e a pergunta falar de estoque/status/retail,
        roda uma SQL estável e devolve SOMENTE a frase curta:
        'É TLP, tem Retail Price de 16,99.'
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

        try:
            rows = _run_sql(sql)
        except Exception as e:
            return {"output": f"Erro ao consultar a base: {e}"}

        if not rows:
            return {"output": "Não encontrei esse SKU/cliente na base."}

        row = rows[0]

        def _g(k):
            for kk in row.keys():
                if kk.lower() == k.lower():
                    return row[kk]
            return None

        status = (_g("status_2025") or "").strip().upper()
        classe = "TLP" if "TLP" in status else "NTLP"

        retail = _g("retail_price")
        retail_str = _fmt_decimal_brl(retail, 2) if retail is not None else "0,00"

        # >>>>>>>>>>>> SOMENTE a frase curta pedida:
        frase = f"É {classe}, tem Retail Price de {retail_str}."

        # devolve só 'output' (sem rows) pra não ativar renderização de texto longo no app
        return {"output": frase}

    # =============== API principal ===============

    def run_query(user_prompt: str) -> Dict[str, Any]:
        # 1) Tenta resposta determinística “SKU + cliente”
        det = _maybe_answer_stock_status_retail(user_prompt)
        if det is not None:
            return det  # ex.: {"output": "É TLP, tem Retail Price de 16,99."}

        # 2) Tenta o agente
        try:
            res = agent_executor.invoke({"input": user_prompt})
        except Exception:
            # 3) fallback one-shot
            raw = query_chain.invoke({"question": f"{BASE_CONTEXT}\n\nPergunta do usuário:\n{user_prompt}"})
            sql = _only_sql(raw)
            # Em fallback, evita devolver rows pra não poluir UI:
            return {"output": f"SQL gerada:\n{sql}"}

        out = res.get("output", "")
        blocked = isinstance(out, str) and (
            "iteration limit" in out.lower() or "time limit" in out.lower()
        )

        if blocked or not isinstance(out, str):
            raw = query_chain.invoke({"question": f"{BASE_CONTEXT}\n\nPergunta do usuário:\n{user_prompt}"})
            sql = _only_sql(raw)
            return {"output": f"SQL gerada:\n{sql}"}

        if "select" in out.lower():
            sql = _only_sql(out)
            return {"output": f"SQL gerada:\n{sql}"}

        # Se o agente já montou uma frase, devolve ela
        return {"output": str(out).strip() or "(sem saída)"}

    return run_query
