# -*- coding: utf-8 -*-
import streamlit as st
import hmac
from agent import get_agent  # <- usa agent.py já corrigido
import os

# ============================
# CONFIG GERAL DA PÁGINA
# ============================
st.set_page_config(page_title="IA para Insights de Negócio")
st.title("🤖 IA para Insights de Negócio")


# ============================
# AUTENTICAÇÃO (sua lógica atual)
# ============================
def check_password():
    def password_entered():
        # compara senha digitada com st.secrets["auth"]["app_password"]
        if hmac.compare_digest(
            st.session_state["password"],
            st.secrets["auth"]["app_password"]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    # já autenticado?
    if st.session_state.get("password_correct", False):
        return True

    # pede senha
    st.text_input(
        "Senha de Acesso",
        type="password",
        on_change=password_entered,
        key="password"
    )

    # erro se digitou errado
    if "password_correct" in st.session_state:
        st.error("Senha incorreta. Tente novamente.")

    st.stop()


if not check_password():
    st.stop()


# ============================
# PEGAR A CHAVE OPENAI
# ============================
def load_api_key() -> str:
    """
    1. Tenta pegar de st.secrets["openai"]["api_key"]
    2. Se não tiver, tenta OPENAI_API_KEY do ambiente
    3. Se nada, para execução
    """
    api_key = None

    # seu padrão atual: st.secrets["openai"]["api_key"]
    if "openai" in st.secrets and "api_key" in st.secrets["openai"]:
        api_key = st.secrets["openai"]["api_key"]

    # fallback: variável de ambiente
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        st.error(
            "❌ OPENAI_API_KEY não encontrada. "
            "Adicione em st.secrets['openai']['api_key'] ou na variável de ambiente."
        )
        st.stop()

    return api_key


# ============================
# INICIALIZAR O AGENTE NA SESSÃO
# ============================
def init_session_agent():
    """
    Cria o 'agente' somente 1x por sessão.
    Atenção: get_agent() retorna uma FUNÇÃO run_query(prompt) → dict.
    A gente guarda essa função em st.session_state['run_query'].
    """
    if "run_query" not in st.session_state:
        openai_key = load_api_key()
        st.session_state["run_query"] = get_agent(openai_key=openai_key)
        # agora st.session_state["run_query"] é chamável:
        # st.session_state["run_query"]("minha pergunta")

    return st.session_state["run_query"]


run_query = init_session_agent()  # função que executa as perguntas


# ============================
# HISTÓRICO DE MENSAGENS
# ============================
if "messages" not in st.session_state:
    st.session_state.messages = []

# renderiza histórico no chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        # se existir SQL salva anteriormente, mostra expansor
        if message["role"] == "assistant" and "sql_query" in message:
            with st.expander("Ver SQL gerada"):
                st.code(message["sql_query"], language="sql")


# ============================
# INPUT DO USUÁRIO
# ============================
if user_prompt := st.chat_input("Digite sua pergunta sobre os dados:"):

    # loga mensagem do usuário
    st.session_state.messages.append({
        "role": "user",
        "content": user_prompt
    })

    # mostra a pergunta imediatamente
    with st.chat_message("user"):
        st.write(user_prompt)

    # IA respondendo...
    with st.chat_message("assistant"):
        with st.spinner("Consultando base e analisando..."):
            try:
                # CHAVE: agora é run_query(prompt), não mais agent_executor(...)
                response = run_query(user_prompt)  # <- retorna dict {"output": "..."} ou {"output": "...", etc}

                final_answer = response.get("output", "(sem retorno)")
                st.write(final_answer)

                # tentativa de extrair SQL (se a resposta veio no formato "SQL gerada:\nSELECT ...")
                sql_content = None
                if final_answer.lower().startswith("sql gerada"):
                    # pega só o trecho depois de "SQL gerada:" pra mostrar bonitinho
                    parts = final_answer.split("SQL gerada:", 1)
                    if len(parts) > 1:
                        sql_content = parts[1].strip()

                # monta bloco da IA para guardar no histórico
                assistant_message = {
                    "role": "assistant",
                    "content": final_answer
                }

                # se a gente conseguiu isolar SQL, salva também
                if sql_content:
                    assistant_message["sql_query"] = sql_content
                    with st.expander("Ver SQL gerada"):
                        st.code(sql_content, language="sql")

                # coloca resposta no histórico
                st.session_state.messages.append(assistant_message)

            except Exception as e:
                error_message = f"Erro ao processar a consulta: {e}"
                st.error(error_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })
