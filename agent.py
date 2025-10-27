# agent.py
# -*- coding: utf-8 -*-
import os, re
from typing import Optional, Dict, Any

from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import create_sql_query_chain

# ===== utils =====
_SKU_RX = re.compile(r"\b([A-Z]\d{3,8})\b", re.I)
_STOP_TOKENS = [" em "," no "," na "," de "," do "," da "," para "," por "," com "," que "," e "," ou "," onde "," quando "]
_STOP_PUNCT = r"[\,\.\?\:\;\!\|/()\[\]\n\r\t]"

def _extract_sku_and_client(prompt: str):
    sku = None
    m = _SKU_RX.search(prompt or "")
    if m: sku = m.group(1).upper()

    cliente = None
    p = prompt or ""; p_low = p.lower()
    idx = p_low.find("cliente")
    if idx >= 0:
        rest = p[idx+len("cliente"):].lstrip()
        if rest.startswith('"'):
            m = re.search(r'^"([^"]+)"', rest); 
            if m: cliente = m.group(1).strip()
        if not cliente and rest.startswith("'"):
            m = re.search(r"^'([^']+)'", rest); 
            if m: cliente = m.group(1).strip()
        if not cliente:
            m = re.search(_STOP_PUNCT, rest)
            cut = rest[:m.start()] if m else rest
            cut_low = " "+cut.lower()+" "; min_pos = None
            for tok in _STOP_TOKENS:
                pos = cut_low.find(tok)
                if pos != -1:
                    pos = pos - 1
                    if min_pos is None or pos < min_pos: min_pos = pos
            if min_pos is not None: cut = cut[:min_pos]
            cliente = cut.strip(" :.-").strip()
        if cliente:
            parts = cliente.split()
            if len(parts) > 6: cliente = " ".join(parts[:6]).strip()
    return sku, cliente

def _fmt_brl(x) -> str:
    try:
        return f"{float(x):.2f}".replace(".", ",")
    except Exception:
        return "0,00"

# ===== factory =====
def get_agent(open_api_key: Optional[str]):
    api_key = open_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Defina OPENAI_API_KEY no ambiente ou passe a chave para get_agent().")

    db = SQLDatabase.from_uri("sqlite:///db/base.db")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    BASE_CONTEXT = r"""
Você é um gerador de SQL para SQLite. Ao acionar a ferramenta, retorne APENAS a string SQL (apenas SELECT).
Use as tabelas: summary_country, pos_week, status_sku, item_master, relatorio_week, classificacao_clientes.
Aspas duplas em colunas com espaço/acentos (ex.: s."Client DC Group"). JOINs canônicos conforme documentação.
"""

    memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)
    agent_executor = create_sql_agent(
        llm=llm, toolkit=toolkit, verbose=True, handle_parsing_errors=True,
        prefix=BASE_CONTEXT, memory=memory, max_iterations=4, max_execution_time=35,
    )
    query_chain = create_sql_query_chain(llm, db, k=5)

    def _run_sql(sql: str):
        return db.run(sql)

    def _maybe_answer_short(prompt: str) -> Optional[Dict[str, Any]]:
        low = (prompt or "").lower()
        if not any(k in low for k in ["estoque","stock","ohi","variação","tlp","ntlp","retail","preço","price"]):
            return None
        sku, cliente = _extract_sku_and_client(prompt)
        if not sku or not cliente:
            return None

        sql = f'''
SELECT
  st."Status POS Master 2025" AS status_2025,
  rw."RETAIL"                 AS retail_price
FROM summary_country s
LEFT JOIN status_sku         st ON st."SKU" = s."Item"
LEFT JOIN relatorio_week     rw ON rw."SKU" = s."Item"
LEFT JOIN classificacao_clientes cc ON cc."Nome Fictício" = s."Client DC Group"
WHERE s."Item" = '{sku}'
  AND (cc."Nome Fictício" = '{cliente}' OR s."Client DC Group" = '{cliente}')
LIMIT 1;
'''.strip()

        try:
            rows = _run_sql(sql)
        except Exception as e:
            return {"output": f"Erro ao consultar a base: {e}"}

        if not rows:
            return {"output": "Não encontrei esse SKU/cliente na base."}

        row = rows[0]
        # pega campos ignorando case
        def g(k):
            for kk in row.keys():
                if kk.lower() == k.lower():
                    return row[kk]
            return None

        status = (g("status_2025") or "").upper()
        classe = "TLP" if "TLP" in status else "NTLP"
        retail = _fmt_brl(g("retail_price"))
        return {"output": f"É {classe}, tem Retail Price de {retail}."}  # <<< só frase curta

    # ===== API =====
    def run_query(user_prompt: str) -> Dict[str, Any]:
        # 1) rota curta (SKU+cliente)
        short = _maybe_answer_short(user_prompt)
        if short is not None:
            return short

        # 2) tenta agente; se vier SQL, retorna apenas a SQL como texto (sem rows)
        try:
            res = agent_executor.invoke({"input": user_prompt})
        except Exception:
            raw = query_chain.invoke({"question": f"{BASE_CONTEXT}\n\nPergunta do usuário:\n{user_prompt}"})
            sql = str(raw).strip().strip("`")
            return {"output": f"SQL gerada:\n{sql}"}

        out = str(res.get("output", "")).strip()
        if out.lower().startswith("select"):
            return {"output": f"SQL gerada:\n{out}"}
        if out:
            return {"output": out}
        return {"output": "Não consegui gerar resposta curta."}

    return run_query
