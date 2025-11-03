# agent.py
# -*- coding: utf-8 -*-

import os
import re
from typing import Optional, Dict, Any, List

from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.chains import create_sql_query_chain

DB_URI = "sqlite:///db/base.db"

# SKU tipo A2799, A7171 etc
_SKU_RX = re.compile(r"\b([A-Z]\d{3,8})\b", re.I)
_STOP_TOKENS = [
    " em ", " no ", " na ", " de ", " do ", " da ",
    " para ", " por ", " com ", " que ", " e ", " ou ",
    " onde ", " quando "
]
_STOP_PUNCT = r"[\,\.\?\:\;\!\|/()\[\]\n\r\t]"


# ---------------------------------------------------------
# util: extrair sku e cliente de uma frase
# ---------------------------------------------------------
def _extract_sku_and_client(prompt: str):
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

        if rest.startswith('"'):
            m = re.search(r'^"([^"]+)"', rest)
            if m:
                cliente = m.group(1).strip()

        if not cliente and rest.startswith("'"):
            m = re.search(r"^'([^']+)'", rest)
            if m:
                cliente = m.group(1).strip()

        if not cliente:
            m = re.search(_STOP_PUNCT, rest)
            cut = rest[:m.start()] if m else rest
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


def _only_sql(text: str) -> str:
    if not text:
        return ""
    txt = text.strip().strip("`").strip()
    if txt.lower().startswith("select"):
        return txt
    m = re.search(r"(?is)\bselect\b.+", txt, re.DOTALL)
    if m:
        return m.group(0).strip()
    return txt


# ---------------------------------------------------------
# nova parte: pegar descrição direto do banco
# ---------------------------------------------------------
def _get_description_by_sku(db: SQLDatabase, sku: str) -> Optional[str]:
    """
    tenta achar a descrição do SKU em tabelas conhecidas.
    volta só o texto da descrição ou None.
    """
    # 1) tenta item_master
    sql_item_master = f"""
SELECT
  "ITEM DESCRIPTION" AS desc_txt
FROM item_master
WHERE "ITEM" = '{sku}'
LIMIT 1;
""".strip()
    try:
        rows = db.run(sql_item_master)
        if rows and rows[0].get("desc_txt"):
            return rows[0]["desc_txt"]
    except Exception:
        pass

    # 2) tenta classificacao_items (caso exista com outro nome)
    sql_classif = f"""
SELECT
  "ITEM DESCRIPTION" AS desc_txt
FROM classificacao_items
WHERE "ITEM" = '{sku}'
LIMIT 1;
""".strip()
    try:
        rows = db.run(sql_classif)
        if rows and rows[0].get("desc_txt"):
            return rows[0]["desc_txt"]
    except Exception:
        pass

    # se nada deu certo
    return None


# ---------------------------------------------------------
# detectar se a pergunta é de descrição
# ---------------------------------------------------------
def _is_description_question(prompt: str) -> bool:
    p = (prompt or "").lower()
    return ("descrição" in p) or ("descricao" in p) or ("description" in p)


# ---------------------------------------------------------
# pick de tabelas (rota simples)
# ---------------------------------------------------------
def _pick_tables(prompt: str) -> List[str]:
    p = prompt.lower()
    if any(x in p for x in ["pos", "semana", "lw", "últimas", "ultimas", "ytd"]):
        return ["pos_week"]
    if any(x in p for x in ["tlp", "ntlp", "status", "classificação", "classificacao"]):
        return ["status_sku"]
    if any(x in p for x in ["estoque", "ohi", "retail", "preço", "preco", "sugerido"]):
        return ["summary_country", "relatorio_week", "status_sku", "classificacao_clientes"]
    if any(x in p for x in ["descrição", "descricao", "item", "sku", "marca", "level"]):
        return ["item_master", "classificacao_items", "status_sku"]
    if any(x in p for x in ["cliente", "canal", "rede"]):
        return ["classificacao_clientes", "summary_country"]
    return [
        "summary_country",
        "pos_week",
        "status_sku",
        "relatorio_week",
        "item_master",
        "classificacao_items",
        "classificacao_clientes",
    ]


# ---------------------------------------------------------
# resumo padrão pra quando não for descrição
# ---------------------------------------------------------
def _summarize_rows(rows: Any) -> str:
    if isinstance(rows, dict) and "_error" in rows:
        return "Gerei a SQL, mas houve erro ao executar no banco. Veja a consulta."
    if not rows:
        return "Não encontrei dados para esse filtro."
    if isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict):
        partes = [f"{k}: {v}" for k, v in rows[0].items()]
        return "Encontrei 1 registro. " + "; ".join(partes)
    if isinstance(rows, list):
        cols = list(rows[0].keys()) if rows and isinstance(rows[0], dict) else []
        return f"Encontrei {len(rows)} linhas. Colunas principais: {', '.join(cols)}."
    return "Consulta concluída."


# ---------------------------------------------------------
# função principal
# ---------------------------------------------------------
def get_agent(open_api_key: Optional[str] = None):
    api_key = open_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Faltou a OPENAI_API_KEY no ambiente.")

    db = SQLDatabase.from_uri(DB_URI)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    sql_chain = create_sql_query_chain(llm, db, k=3)

    def _run_sql_safe(sql: str):
        try:
            return db.run(sql)
        except Exception as e:
            return {"_error": str(e), "_sql": sql}

    def run_query(prompt: str) -> Dict[str, Any]:
        # 1) caso especial: pergunta de descrição + SKU
        sku, cliente = _extract_sku_and_client(prompt)
        if sku and _is_description_question(prompt):
            desc = _get_description_by_sku(db, sku)
            if desc:
                return {
                    "output": f"A descrição do item {sku} é: {desc}",
                    "sql": None,
                    "rows": []
                }
            # se não achou descrição, cai pro fluxo normal

        # 2) caso especial: sku + cliente (estoque / retail / tlp-ntlp)
        if sku and cliente:
            sql = f"""
SELECT
  s."OHI CY"                  AS ohi_cy,
  s."OHI Var%"                AS ohi_var,
  st."Status POS Master 2025" AS status,
  rw."RETAIL"                 AS retail
FROM summary_country s
LEFT JOIN status_sku     st ON st."SKU" = COALESCE(s."Item", s."SKU", s."Item Code")
LEFT JOIN relatorio_week rw ON rw."SKU" = COALESCE(s."Item", s."SKU", s."Item Code")
LEFT JOIN classificacao_clientes cc
    ON cc."Nome Fictício" = s."Client DC Group"
WHERE COALESCE(s."Item", s."SKU", s."Item Code") = '{sku}'
  AND (cc."Nome Fictício" = '{cliente}' OR s."Client DC Group" = '{cliente}')
LIMIT 1;
""".strip()
            rows = _run_sql_safe(sql)
            if not (isinstance(rows, dict) and "_error" in rows) and rows:
                r = rows[0]
                status = (r.get("status") or "").upper()
                classe = "TLP" if "TLP" in status else "NTLP"
                retail = _fmt_decimal_brl(r.get("retail") or 0, 2)
                return {
                    "output": (
                        f"Para o SKU {sku} no cliente {cliente}: é {classe}, "
                        f"Retail Price {retail}, estoque {r.get('ohi_cy')}, variação {r.get('ohi_var')}."
                    ),
                    "sql": sql,
                    "rows": rows,
                }
            # fallback só por sku
            sql_fb = f"""
SELECT
  st."Status POS Master 2025" AS status,
  rw."RETAIL"                 AS retail
FROM status_sku st
LEFT JOIN relatorio_week rw ON rw."SKU" = st."SKU"
WHERE st."SKU" = '{sku}'
LIMIT 1;
""".strip()
            rows_fb = _run_sql_safe(sql_fb)
            if not (isinstance(rows_fb, dict) and "_error" in rows_fb) and rows_fb:
                r = rows_fb[0]
                status = (r.get("status") or "").upper()
                classe = "TLP" if "TLP" in status else "NTLP"
                retail = _fmt_decimal_brl(r.get("retail") or 0, 2)
                return {
                    "output": f"Para o SKU {sku}: é {classe} e o Retail Price é {retail}.",
                    "sql": sql_fb,
                    "rows": rows_fb,
                }

        # 3) fluxo normal → LLM gera SQL
        tables = _pick_tables(prompt)
        question = (
            "Gere apenas um SELECT válido para SQLite, usando SOMENTE estas tabelas "
            f"{', '.join(tables)}. "
            "Não explique, não use texto extra. Pergunta do usuário: "
            f"{prompt}"
        )
        raw_sql = sql_chain.invoke({"question": question})
        sql = _only_sql(raw_sql).strip()

        if not sql.lower().startswith("select"):
            return {
                "output": raw_sql if raw_sql else "Não consegui gerar SQL.",
                "sql": None,
                "rows": [],
            }

        rows = _run_sql_safe(sql)
        texto = _summarize_rows(rows)

        return {
            "output": texto,
            "sql": sql,
            "rows": rows,
        }

    return run_query
