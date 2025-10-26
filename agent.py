# app_sql_ai.py
# ------------------------------------------------------------
# App de IA para consultas SQL (SQLite) com geração + execução
# robusta (sem OUTPUT_PARSING_FAILURE). Pronto pra usar no backend
# ou colar numa API (FastAPI/Flask).
# ------------------------------------------------------------

import os
import re
import logging
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain

# ---------- logging básico ----------
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
log = logging.getLogger("sql-ai")

# ---------- env ----------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # defina no .env
DB_URI = os.getenv("DB_URI", "sqlite:///db/base.db")  # troque aqui se quiser
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # pode trocar p/ gpt-4o

if not OPENAI_API_KEY:
    raise ValueError("🚨 Falta a OPENAI_API_KEY no .env")

# ---------- LLM ----------
llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0,
    api_key=OPENAI_API_KEY,
)

# ---------- DB ----------
# Dica: inclua só as tabelas que realmente vai usar p/ dar mais grounding
INCLUDE_TABLES = [
    "summary_country",          # considere criar views *_v em snake_case no banco
    "pos_week",
    "status_sku",
    "item_master",
    "relatorio_week",
    "classificacao_clientes",
]

db = SQLDatabase.from_uri(
    DB_URI,
    include_tables=INCLUDE_TABLES,
    sample_rows_in_table_info=3,   # 2-5 amostras por tabela ajuda MUITO a acertar nomes/joins
)

# ---------- Prompt base (contexto) ----------
# Mantive tua descrição rica das tabelas, mas resumi aqui pra caber.
# Se preferir, jogue esse bloco num .txt e carregue.
BASE_CONTEXT = r"""
Você é um assistente de dados que gera consultas **SQLite** a partir de perguntas em linguagem natural.
Trabalhe SOMENTE com as tabelas listadas na documentação do esquema retornada pelo banco.
IMPORTANTE: várias colunas têm espaços/acentos. No SQLite, **identificadores com espaço/acentos DEVEM ser citados com ASPAS DUPLAS**:
ex.: pw."Item WK", sc."Client DC Group".

Aliases recomendados:
- summary_country -> sc
- pos_week -> pw
- classificacao_clientes -> cc
- item_master -> im
- status_sku -> ss
- relatorio_week -> rw

Relações canônicas:
- sc."Item" ↔ pw."Item WK"
- sc."Client DC Group" ↔ pw."Client WK"
- sc."Client DC Group" ↔ cc."Nome Fictício"
- im."ITEM" ↔ sc."Item" / pw."Item WK" / ss."SKU" / rw."SKU"

Regras de output (OBRIGATÓRIO):
- Gere **somente** SQL válido para **SQLite** (sem crases/backticks, sem DESCRIBE/SHOW).
- Não use SELECT * (retorne só colunas necessárias).
- Use LIMIT quando fizer sentido (ex.: amostrar 50 linhas).
- NUNCA retorne explicações junto com a query. O output deve ser APENAS a string SQL.

Few-shots (exemplos):
-- Q: "Qual a descrição do item A2799?"
SELECT im."ITEM DESCRIPTION"
FROM item_master im
WHERE im."ITEM" = 'A2799'
LIMIT 1;

-- Q: "Qual é o 'POS YTD CY' do cliente 'Atacadão Vitória' para o SKU 'A2982'?"
SELECT pw."POS YTD CY"
FROM pos_week pw
JOIN summary_country sc
  ON pw."Item WK" = sc."Item" AND pw."Client WK" = sc."Client DC Group"
JOIN classificacao_clientes cc
  ON sc."Client DC Group" = cc."Nome Fictício"
WHERE cc."Nome Fictício" = 'Atacadão Vitória'
  AND pw."Item WK" = 'A2982'
LIMIT 1;
"""

# ---------- Cadeia de geração de SQL (function-calling por baixo) ----------
# Isso evita o OUTPUT_PARSING_FAILURE, pois o chain retorna só a string SQL.
make_sql = create_sql_query_chain(llm, db, k=5)  # k=colunas de preview p/ grounding

# ---------- Helpers ----------
SELECT_RE = re.compile(r"^\s*select\b", re.IGNORECASE)

def _ensure_select_only(sql: str) -> str:
    """Bloqueia comandos não-SELECT (segurança básica)."""
    if not SELECT_RE.match(sql or ""):
        raise ValueError("Consulta não é SELECT. Bloqueado por segurança.")
    return sql

def _ensure_limit(sql: str, default_limit: int = 200) -> str:
    """Se não houver LIMIT, adiciona um p/ evitar dumps gigantes."""
    if re.search(r"\blimit\b\s+\d+", sql, flags=re.IGNORECASE):
        return sql
    # não injeta LIMIT dentro de subqueries; isso é simples e cobre 99% dos casos
    return f"{sql.rstrip().rstrip(';')} LIMIT {default_limit};"

# ---------- API pública ----------
def run_query(question: str) -> Dict[str, Any]:
    """
    Recebe a pergunta em linguagem natural, gera SQL e executa.
    Retorna {sql, rows} ou {sql, error}.
    """
    # 1) gera SQL cru
    prompt = f"{BASE_CONTEXT}\n\nPergunta do usuário: {question}"
    sql: str = make_sql.invoke({"question": prompt})

    # limpeza simples de quebras ou blocos triple-backtick que o LLM possa colocar
    sql = sql.strip().strip("`").strip()
    # segurança
    try:
        sql = _ensure_select_only(sql)
        sql = _ensure_limit(sql)
    except Exception as e:
        return {"sql": sql, "error": f"Validação de segurança: {e}"}

    log.info(f"SQL gerado:\n{sql}")
    # 2) executa
    try:
        rows = db.run(sql)
        return {"sql": sql, "rows": rows}
    except Exception as e:
        # retorna erro + SQL p/ você depurar rápido
        return {"sql": sql, "error": str(e)}

# ---------- exemplo de uso local ----------
if __name__ == "__main__":
    # Perguntas exemplo:
    qs = [
        "Qual a descrição do item A2799?",
        "Qual é o 'POS YTD CY' do cliente 'Atacadão Vitória' para o SKU 'A2982'?",
        "Top 5 itens por 'POS YTD CY' no cliente 'Atacadão Vitória'",
    ]
    for q in qs:
        print("\nQ:", q)
        print(run_query(q))
