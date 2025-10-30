
# -*- coding: utf-8 -*-
import re
from typing import Optional, Dict, Any
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import create_sql_query_chain

# ============================
# CONFIGURAÇÃO GERAL
# ============================
DB_URI = "sqlite:///db/base.db"

_SKU_RX = re.compile(r"\b([A-Z]\d{3,8})\b", re.I)
_STOP_TOKENS = [" em "," no "," na "," de "," do "," da "," para "," por "," com "," que "," e "," ou "," onde "," quando "]
_STOP_PUNCT = r"[\,\.\?\:\;\!\|/()\[\]\n\r\t]"

# ============================
# FUNÇÕES DE SUPORTE
# ============================
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

def _fmt_decimal_brl(x: float, casas=2) -> str:
    try: s = f"{float(x):.{casas}f}"
    except Exception: return str(x)
    return s.replace(".", ",")

# ============================
# CRIAÇÃO DE AGENTES
# ============================
def _make_sql_agent(tables, openai_key):
    db = SQLDatabase.from_uri(DB_URI, include_tables=tables)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_key)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    BASE_CONTEXT = f"""
Você é um especialista SQL focado em tabelas específicas.
Gere apenas SELECTs válidos para SQLite.
Tabelas disponíveis: {', '.join(tables)}.
"""

    memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)

    return create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=False,
        handle_parsing_errors=True,
        prefix=BASE_CONTEXT,
        memory=memory,
        max_iterations=3,
        max_execution_time=25,
    )

# Agentes especializados
agent_summary = _make_sql_agent(["summary_country"], "summary")
agent_posweek = _make_sql_agent(["pos_week"], "posweek")
agent_item = _make_sql_agent(["classificacao_items"], "item")
agent_status = _make_sql_agent(["status_sku"], "status")
agent_relweek = _make_sql_agent(["relatorio_week"], "relweek")
agent_clientes = _make_sql_agent(["classificacao_clientes"], "clientes")
agent_misto = _make_sql_agent(["summary_country", "classificacao_items"], "misto")

# ============================
# ROTEAMENTO INTELIGENTE
# ============================
def _route_agent(prompt: str):
    prompt_low = prompt.lower()
    if any(x in prompt_low for x in ["venda", "semana", "pos", "lw", "ytd"]):
        return agent_posweek
    if any(x in prompt_low for x in ["tlp", "ntlp", "status"]):
        return agent_status
    if any(x in prompt_low for x in ["estoque", "ohi", "variação", "retail", "preço"]):
        return agent_relweek
    if any(x in prompt_low for x in ["marca", "descrição", "level", "sku", "item"]):
        return agent_item
    if any(x in prompt_low for x in ["resumo", "country", "mundo"]):
        return agent_summary
    return agent_misto

# ============================
# EXECUÇÃO / CONSULTA
# ============================
def get_agent(openai_key: Optional[str] = None):
    api_key = openai_key
    db = SQLDatabase.from_uri(DB_URI)
    query_chain = create_sql_query_chain(ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key), db, k=3)

    _SQL_BLOCK = re.compile(r"(?is)\bselect\b.+", re.DOTALL)
    def _only_sql(text: str) -> str:
        text = (text or "").strip().strip("`").strip()
        if text.lower().startswith("select"): return text
        m = _SQL_BLOCK.search(text); return m.group(0).strip() if m else text

    def run_query(prompt: str) -> Dict[str, Any]:
        agent = _route_agent(prompt)

        # Casos diretos de SKU + cliente
        sku, cliente = _extract_sku_and_client(prompt)
        if sku and cliente:
            sql = f'''
SELECT
  s."OHI CY" AS ohi_cy,
  s."OHI Var%" AS ohi_var,
  st."Status POS Master 2025" AS status,
  rw."RETAIL" AS retail
FROM summary_country s
LEFT JOIN status_sku st ON st."SKU" = s."Item"
LEFT JOIN relatorio_week rw ON rw."SKU" = s."Item"
LEFT JOIN classificacao_clientes cc ON cc."Nome Fictício" = s."Client DC Group"
WHERE s."Item" = '{sku}'
  AND (cc."Nome Fictício" = '{cliente}' OR s."Client DC Group" = '{cliente}')
LIMIT 1;
'''.strip()
            try:
                rows = db.run(sql)
                if not rows: return {"output": "SKU/cliente não encontrado."}
                row = rows[0]
                status = (row.get("status") or "").upper()
                classe = "TLP" if "TLP" in status else "NTLP"
                retail = row.get("retail")
                retail_fmt = _fmt_decimal_brl(retail, 2)
                return {"output": f"É {classe}, tem Retail Price de {retail_fmt}."}
            except Exception as e:
                return {"output": f"Erro ao consultar: {e}"}

        # Caso geral → via agente
        try:
            res = agent.invoke({"input": prompt})
            out = res.get("output", "")
            if isinstance(out, str) and "select" in out.lower():
                return {"output": f"SQL gerada:\n{_only_sql(out)}"}
            if isinstance(out, str) and out.strip():
                return {"output": out.strip()}
        except Exception:
            raw = query_chain.invoke({"question": f"Gere SQL válida para: {prompt}"})
            return {"output": f"SQL gerada:\n{_only_sql(raw)}"}

        return {"output": "Não consegui gerar resposta."}

    return run_query
