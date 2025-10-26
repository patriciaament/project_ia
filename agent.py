# agent_v2.py
# IA SQL robusta: gera o SQL em 1 passo (sem loop), executa, e retorna {sql, rows|error}

import os
import re
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain

# -------------------------
# CONFIG
# -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_URI = os.getenv("DB_URI", "sqlite:///db/base.db")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    raise ValueError("Defina OPENAI_API_KEY no ambiente (.env).")

# LLM
llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0,
    api_key=OPENAI_API_KEY,
)

# DB (limita as tabelas pra dar grounding forte)
db = SQLDatabase.from_uri(
    DB_URI,
    include_tables=[
        "summary_country",
        "pos_week",
        "status_sku",
        "item_master",
        "relatorio_week",
        "classificacao_clientes",
    ],
    sample_rows_in_table_info=3,  # mostra exemplos p/ o LLM acertar colunas
)

# -------------------------
# CONTEXTO (raw string)
# -------------------------
BASE_CONTEXT = r"""
Você é um gerador de **SQL para SQLite**. Responda **APENAS com a string SQL** (sem markdown, sem explicações).
Sempre cite identificadores com espaço/acentos usando **aspas duplas**: pw."Item WK", sc."Client DC Group", etc.
Evite SELECT *; traga só o necessário. Quando listar muitas linhas, use LIMIT 50.

Tabelas & relações (aliases recomendados):
- summary_country (sc)
  - chaves: sc."Item", sc."Client DC Group"
- pos_week (pw)
  - chaves: pw."Item WK", pw."Client WK"
- item_master (im)
  - chaves: im."ITEM", im."ITEM DESCRIPTION", im."Level_1", im."Level_2", im."Level_3", im."Level_4"
- classificacao_clientes (cc)
  - chaves: cc."Nome Fictício", cc."Canal Adaptado"

JOINs canônicos:
- pw."Item WK" = sc."Item"
- pw."Client WK" = sc."Client DC Group"
- sc."Client DC Group" = cc."Nome Fictício"
- im."ITEM" pode se unir a sc."Item" e pw."Item WK"

Conceitos do negócio:
- **LW** = "Last Week" (semana mais recente) — use as colunas *LW* quando aparecerem (ex.: "GB LW CY", "POS LW CY", "POS LW Var$" etc), se existirem.
- **Variação absoluta** = coluna com sufixo **Var$**.
- **NTLP** = categoria/linha de produto. Considere que isso costuma morar em `item_master` (ex.: "Level_2" ou similar).
- **Barbie** = marca/linha. Costuma morar em `item_master` (ex.: "Level_1" = 'Barbie').

Exemplos de padrões úteis:
1) Descrição do item:
SELECT im."ITEM DESCRIPTION"
FROM item_master im
WHERE im."ITEM" = 'A2799'
LIMIT 1;

2) POS YTD CY por cliente e SKU (JOIN canônico):
SELECT pw."POS YTD CY"
FROM pos_week pw
JOIN summary_country sc
  ON pw."Item WK" = sc."Item" AND pw."Client WK" = sc."Client DC Group"
JOIN classificacao_clientes cc
  ON sc."Client DC Group" = cc."Nome Fictício"
WHERE cc."Nome Fictício" = 'Atacadão Vitória'
  AND pw."Item WK" = 'A2982'
LIMIT 1;

Dicas:
- Se a pergunta envolver **NTLP** e **Barbie**, filtre em `item_master`, unindo `im."ITEM"` a `pw."Item WK"` (e/ou `sc."Item"`).
- Para **variação absoluta de POS na LW**, procure colunas como **"POS LW Var$"**; se não houver, calcule como ( "POS LW CY" - "POS LW LY" ), quando existir.
- Sempre gere **UMA** query válida de **SELECT** para SQLite, sem comentários e sem texto adicional.
"""

# -------------------------
# Cadeia: gera SQL (1 passo)
# -------------------------
make_sql = create_sql_query_chain(llm, db, k=5)  # k define amostras de colunas no prompt interno

_SQL_RE = re.compile(r"(?is)\bselect\b.+", re.DOTALL)

def _sanitize_sql(text: str) -> str:
    """Extrai só o SELECT, garante que não vem markdown/prosa, e injeta LIMIT se faltar."""
    text = (text or "").strip().strip("`").strip()
    m = _SQL_RE.search(text)
    if not m:
        return text
    sql = m.group(0).strip().rstrip(";")
    # LIMIT auto (se a query for potencialmente grande)
    if " limit " not in sql.lower():
        sql = f"{sql} LIMIT 200"
    return sql + ";"

def _is_select(sql: str) -> bool:
    return sql.lower().lstrip().startswith("select")

# -------------------------
# API pública
# -------------------------
def run_query(question: str) -> Dict[str, Any]:
    """
    Usa LLM para gerar o SQL em 1 tiro e executa.
    Retorna: {"sql": "...", "rows": [...] } ou {"sql": "...", "error": "..."}
    """
    prompt = f"{BASE_CONTEXT}\n\nPergunta do usuário:\n{question}"
    try:
        raw_sql = make_sql.invoke({"question": prompt})
    except Exception as e:
        return {"error": f"Falha ao gerar SQL: {e}"}

    sql = _sanitize_sql(raw_sql)
    if not _is_select(sql):
        return {"sql": raw_sql, "error": "Modelo não gerou um SELECT válido. Refine a pergunta."}

    try:
        rows = db.run(sql)
        return {"sql": sql, "rows": rows}
    except Exception as e:
        return {"sql": sql, "error": str(e)}
