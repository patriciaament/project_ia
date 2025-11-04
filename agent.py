@@ -12,13 +12,9 @@
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import create_sql_query_chain

# ======================================================
# CONFIG GERAL
# ======================================================

DB_URI = "sqlite:///db/base.db"

# SKU: começa com letra, depois 3-8 dígitos, tipo A7171
# SKU do tipo A8350, A5460 etc.
_SKU_RX = re.compile(r"\b([A-Z]\d{3,8})\b", re.I)

_STOP_TOKENS = [
@@ -29,42 +25,28 @@
_STOP_PUNCT = r"[\,\.\?\:\;\!\|/()\[\]\n\r\t]"


# ======================================================
# FUNÇÕES UTILITÁRIAS
# ======================================================

# =============== helpers básicos =============== #
def _extract_sku(prompt: str) -> Optional[str]:
    m = _SKU_RX.search(prompt or "")
    if m:
        return m.group(1).upper()
    return None
    return m.group(1).upper() if m else None


def _extract_sku_and_client(prompt: str):
    """ainda usamos esse para o caso SKU + cliente"""
    sku = _extract_sku(prompt)

    cliente = None
    p = prompt or ""
    p_low = p.lower()

    idx = p_low.find("cliente")
    if idx >= 0:
        rest = p[idx + len("cliente"):].lstrip()

        # cliente "Nome"
        if rest.startswith('"'):
            m = re.search(r'^"([^"]+)"', rest)
            if m:
                cliente = m.group(1).strip()

        # cliente 'Nome'
        if not cliente and rest.startswith("'"):
            m = re.search(r"^'([^']+)'", rest)
            if m:
                cliente = m.group(1).strip()

        # cliente sem aspas
        if not cliente:
            m = re.search(_STOP_PUNCT, rest)
            cut = rest[:m.start()] if m else rest
@@ -79,12 +61,10 @@ def _extract_sku_and_client(prompt: str):
            if min_pos is not None:
                cut = cut[:min_pos]
            cliente = cut.strip(" :.-").strip()

        if cliente:
            parts = cliente.split()
            if len(parts) > 6:
                cliente = " ".join(parts[:6]).strip()

    return sku, cliente


@@ -103,36 +83,29 @@ def _only_sql(text: str) -> str:
    if txt.lower().startswith("select"):
        return txt
    m = re.search(r"(?is)\bselect\b.+", txt, re.DOTALL)
    if m:
        return m.group(0).strip()
    return txt
    return m.group(0).strip() if m else txt


def _summarize_result(pergunta: str, rows: Any) -> str:
    if isinstance(rows, dict) and "_error" in rows:
        return ("Tentei consultar os dados, mas houve um erro ao executar a query no banco. "
                "Veja a SQL gerada no painel.")
        return "Tentei consultar, mas houve erro ao executar a SQL. Veja a consulta gerada."
    if not rows:
        return "Não encontrei dados relevantes pra essa pergunta no banco."
        return "Não encontrei dados para essa pergunta."
    if isinstance(rows, list) and len(rows) == 1 and isinstance(rows[0], dict):
        row = rows[0]
        partes = [f"{k}: {v}" for k, v in row.items()]
        return "Encontrei 1 registro: " + "; ".join(partes)
        partes = [f"{k}: {v}" for k, v in rows[0].items()]
        return "Encontrei 1 registro: " + "; ".join(partes) + "."
    if isinstance(rows, list):
        cols = list(rows[0].keys()) if rows and isinstance(rows[0], dict) else []
        return (f"Encontrei {len(rows)} linhas que atendem à consulta. "
                f"Colunas principais: {', '.join(cols)}.")
        return f"Encontrei {len(rows)} linhas. Colunas principais: {', '.join(cols)}."
    return "Consulta concluída."


# ======================================================
# get_agent (principal)
# ======================================================

# =============== agent principal =============== #
def get_agent(open_api_key: Optional[str] = None):

    api_key = open_api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Faltou a OPENAI_API_KEY no ambiente (.env ou secrets).")
        raise ValueError("Faltou a OPENAI_API_KEY no ambiente.")

    db_main = SQLDatabase.from_uri(DB_URI)
    llm_main = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
@@ -144,14 +117,22 @@ def _run_sql_safe(sql: str):
        except Exception as e:
            return {"_error": str(e), "_sql": sql}

    # ------------- agents menores -------------
    def _table_exists(name: str) -> bool:
        q = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{name}';"
        try:
            res = db_main.run(q)
            return bool(res)
        except Exception:
            return False

    # cria agentes especializados
    def _make_sql_agent(tables: List[str]):
        sub_db = SQLDatabase.from_uri(DB_URI, include_tables=tables)
        toolkit = SQLDatabaseToolkit(db=sub_db, llm=llm_main)
        base_ctx = f"""
        Você é um gerador de SQL para SQLite.
        Gere APENAS SELECTs válidos para as tabelas: {', '.join(tables)}.
        Não coloque explicação junto. Apenas o SELECT.
        Gere APENAS SELECTs para as tabelas: {', '.join(tables)}.
        Não coloque explicações junto.
        """
        memory = ConversationBufferWindowMemory(k=3, memory_key="chat_history", return_messages=True)
        return create_sql_agent(
@@ -181,103 +162,101 @@ def _make_sql_agent(tables: List[str]):
        "item_master",
    ])

    # ---------- roteador ---------- #
    def _route_agent(prompt: str):
        p = prompt.lower()
        if any(x in p for x in ["pos", "semana", "lw", "4 semanas", "ytd"]):
        if any(x in p for x in ["pos", "semana", "lw", "ytd"]):
            return agent_posweek
        if any(x in p for x in ["tlp", "ntlp", "status", "classificação", "classificacao"]):
            return agent_status
        if any(x in p for x in ["estoque", "ohi", "retail", "preço", "preco"]):
            return agent_relweek
        if any(x in p for x in ["descrição", "description", "item description", "level", "sku", "item"]):
        if any(x in p for x in ["descrição", "descricao", "description", "item description", "level", "sku", "item"]):
            return agent_item
        if any(x in p for x in ["cliente", "canal", "rede"]):
            return agent_clientes
        if any(x in p for x in ["resumo", "country", "visão geral", "visao geral"]):
            return agent_summary
        return agent_misto

    # --------------------------------------------------
    # HOOK 1: perguntas de descrição de item
    # --------------------------------------------------
    # ---------- NOVO: pegador de descrição que tenta vários nomes de tabela ---------- #
    def _maybe_answer_item_description(prompt: str):
        p = (prompt or "").lower()
        wants_description = any(
            k in p for k in [
                "descrição do item",
                "descricao do item",
                "item description",
                "o que você pode me dizer do item",
                "me diga do item",
                "descreva o item",
            ]
        )
        wants_description = any(k in p for k in [
            "descrição do item",
            "descricao do item",
            "item description",
            "o que você pode me dizer do item",
            "o que voce pode me dizer do item",
            "me diga do item",
        ])
        sku = _extract_sku(prompt)

        if not wants_description or not sku:
            return None  # não é esse caso
            return None

        # ordem de tentativas de tabela
        candidate_tables = [
            "item_master",        # nome mais provável
            "ITEM_MASTER",        # caso tenha sido criado em caps
            "Item_Master",        # variação
            "classificacao_items" # aquela que você já mostrou
        ]

        # colunas que queremos SE existirem
        wanted_cols = ['"ITEM"', '"ITEM DESCRIPTION"', '"Level_1"', '"Level_2"', '"Level_3"', '"Level_4"']

        # 1) tenta na tabela item_master (que é a que você mostrou)
        sql_item_master = f'''
SELECT "ITEM", "ITEM DESCRIPTION", "Level_1", "Level_2", "Level_3", "Level_4"
FROM item_master
        for tbl in candidate_tables:
            if not _table_exists(tbl):
                continue

            # primeiro tenta com ITEM DESCRIPTION
            sql_full = f'''
SELECT {", ".join(wanted_cols)}
FROM {tbl}
WHERE "ITEM" = '{sku}'
LIMIT 1;
'''.strip()
            rows = _run_sql_safe(sql_full)

        rows = _run_sql_safe(sql_item_master)
        if not (isinstance(rows, dict) and "_error" in rows) and rows:
            r = rows[0]
            desc = r.get("ITEM DESCRIPTION") or r.get("item description") or "sem descrição cadastrada"
            l1 = r.get("Level_1") or r.get("level_1")
            l2 = r.get("Level_2") or r.get("level_2")
            l3 = r.get("Level_3") or r.get("level_3")
            l4 = r.get("Level_4") or r.get("level_4")
            texto = f"O item {sku} tem a descrição: {desc}."
            niveis = [l for l in [l1, l2, l3, l4] if l]
            if niveis:
                texto += " Ele está classificado nos níveis: " + " > ".join(niveis) + "."
            return {
                "output": texto,
                "sql": sql_item_master,
                "rows": rows,
            }

        # 2) fallback na classificacao_items (caso o SQLite esteja com esse nome)
        sql_classif = f'''
SELECT "ITEM", "ITEM DESCRIPTION", "Level_1", "Level_2", "Level_3", "Level_4"
FROM classificacao_items
            # se deu erro porque a coluna não existe, faz um select reduzido
            if isinstance(rows, dict) and "_error" in rows and "no such column" in rows["_error"].lower():
                sql_reduced = f'''
SELECT "ITEM", "Level_1", "Level_2", "Level_3", "Level_4"
FROM {tbl}
WHERE "ITEM" = '{sku}'
LIMIT 1;
'''.strip()
        rows2 = _run_sql_safe(sql_classif)
        if not (isinstance(rows2, dict) and "_error" in rows2) and rows2:
            r = rows2[0]
            desc = r.get("ITEM DESCRIPTION") or "sem descrição cadastrada"
            l1 = r.get("Level_1")
            l2 = r.get("Level_2")
            l3 = r.get("Level_3")
            l4 = r.get("Level_4")
            texto = f"O item {sku} tem a descrição: {desc}."
            niveis = [l for l in [l1, l2, l3, l4] if l]
            if niveis:
                texto += " Ele está classificado nos níveis: " + " > ".join(niveis) + "."
            return {
                "output": texto,
                "sql": sql_classif,
                "rows": rows2,
            }

        # 3) se nem item_master nem classificacao_items funcionarem, diz claro
                rows2 = _run_sql_safe(sql_reduced)
                if not (isinstance(rows2, dict) and "_error" in rows2) and rows2:
                    r = rows2[0]
                    levels = [r.get("Level_1"), r.get("Level_2"), r.get("Level_3"), r.get("Level_4")]
                    levels = [x for x in levels if x]
                    texto = f"Encontrei o item {sku} na tabela {tbl}, mas ela não tem coluna de descrição."
                    if levels:
                        texto += " Classificação: " + " > ".join(levels) + "."
                    return {"output": texto, "sql": sql_reduced, "rows": rows2}
                # se nem o reduzido funcionou, passa pra próxima tabela
                continue

            # se não deu erro e veio linha, pronto
            if not (isinstance(rows, dict) and "_error" in rows) and rows:
                r = rows[0]
                desc = r.get("ITEM DESCRIPTION") or "sem descrição cadastrada"
                levels = [r.get("Level_1"), r.get("Level_2"), r.get("Level_3"), r.get("Level_4")]
                levels = [x for x in levels if x]
                texto = f"O item {sku} tem a descrição: {desc}."
                if levels:
                    texto += " Classificação: " + " > ".join(levels) + "."
                return {"output": texto, "sql": sql_full, "rows": rows}

        # se nenhuma tabela ajudou
        return {
            "output": f"Não encontrei a descrição do item {sku} nas tabelas disponíveis.",
            "sql": sql_item_master,
            "rows": rows,
            "output": f"Não encontrei a descrição do item {sku} nas tabelas que tenho acesso.",
            "sql": None,
            "rows": [],
        }

    # --------------------------------------------------
    # HOOK 2: SKU + cliente (estoque / retail / TLP)
    # --------------------------------------------------
    # ---------- SKU + cliente (o de antes) ---------- #
    def _monta_texto_sku_cliente_full(sku: str, cliente: str, row: Dict[str, Any]) -> str:
        status_val = (row.get("status") or "").upper()
        classe = "TLP" if "TLP" in status_val else "NTLP"
@@ -286,8 +265,8 @@ def _monta_texto_sku_cliente_full(sku: str, cliente: str, row: Dict[str, Any]) -
        ohi_cy = row.get("ohi_cy")
        ohi_var = row.get("ohi_var")
        return (
            f"Para o SKU {sku} no cliente {cliente}: está classificado como {classe}. "
            f"Retail Price: {retail_fmt}. Estoque (OHI CY): {ohi_cy}, variação: {ohi_var}."
            f"Para o SKU {sku} no cliente {cliente}: classificado como {classe}, "
            f"retail price {retail_fmt}, estoque {ohi_cy}, variação {ohi_var}."
        )

    def _monta_texto_sku_cliente_fallback(sku: str, cliente: str, row: Dict[str, Any]) -> str:
@@ -296,20 +275,18 @@ def _monta_texto_sku_cliente_fallback(sku: str, cliente: str, row: Dict[str, Any
        retail_val = row.get("retail")
        retail_fmt = _fmt_decimal_brl(retail_val, 2)
        return (
            f"Para o SKU {sku}: está classificado como {classe} e o retail price é {retail_fmt}. "
            f"Não consegui confirmar os dados específicos do cliente {cliente}."
            f"Para o SKU {sku}: classificado como {classe} e retail price {retail_fmt}. "
            f"Não consegui trazer os dados específicos do cliente {cliente}."
        )

    # --------------------------------------------------
    # FUNÇÃO QUE O FRONT CHAMA
    # --------------------------------------------------
    # ---------- função que o Streamlit chama ---------- #
    def run_query(prompt: str) -> Dict[str, Any]:
        # 1) primeiro: se for claramente pedir descrição de SKU → tratamos aqui
        # 1) caso “qual a descrição do item X?”
        desc_res = _maybe_answer_item_description(prompt)
        if desc_res is not None:
            return desc_res

        # 2) depois: caso especial SKU + cliente (estoque / retail)
        # 2) caso SKU + cliente
        sku, cliente = _extract_sku_and_client(prompt)
        if sku and cliente:
            sql_full = f'''
@@ -328,8 +305,11 @@ def run_query(prompt: str) -> Dict[str, Any]:
'''.strip()
            rows_full = _run_sql_safe(sql_full)
            if not (isinstance(rows_full, dict) and "_error" in rows_full) and rows_full:
                texto = _monta_texto_sku_cliente_full(sku, cliente, rows_full[0])
                return {"output": texto, "sql": sql_full, "rows": rows_full}
                return {
                    "output": _monta_texto_sku_cliente_full(sku, cliente, rows_full[0]),
                    "sql": sql_full,
                    "rows": rows_full,
                }

            sql_fb = f'''
SELECT
@@ -342,16 +322,19 @@ def run_query(prompt: str) -> Dict[str, Any]:
'''.strip()
            rows_fb = _run_sql_safe(sql_fb)
            if not (isinstance(rows_fb, dict) and "_error" in rows_fb) and rows_fb:
                texto = _monta_texto_sku_cliente_fallback(sku, cliente, rows_fb[0])
                return {"output": texto, "sql": sql_fb, "rows": rows_fb}
                return {
                    "output": _monta_texto_sku_cliente_fallback(sku, cliente, rows_fb[0]),
                    "sql": sql_fb,
                    "rows": rows_fb,
                }

            return {
                "output": f"Tentei recuperar dados do SKU {sku} para o cliente {cliente}, mas não consegui.",
                "output": f"Tentei recuperar o SKU {sku} para o cliente {cliente}, mas não consegui.",
                "sql": sql_full,
                "rows": rows_full,
            }

        # 3) caso geral → roteia pra um agente
        # 3) caso geral → agente
        agent = _route_agent(prompt)
        try:
            agent_res = agent.invoke({"input": prompt})
@@ -366,7 +349,6 @@ def run_query(prompt: str) -> Dict[str, Any]:

        rows_sample = _run_sql_safe(sql_candidate)
        texto_final = _summarize_result(prompt, rows_sample)

        return {
            "output": texto_final,
            "sql": sql_candidate,
