# app.py
# -*- coding: utf-8 -*-
import hmac
import streamlit as st
from agent import get_agent

st.set_page_config(page_title="IA para Insights de Negócio")
st.title("🤖 IA para Insights de Negócio")


# ---------------------------------
# Autenticação simples via senha
# ---------------------------------
def check_password():
    def password_entered():
        if hmac.compare_digest(
            st.session_state["password"],
            st.secrets["auth"]["app_password"]
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input(
        "Senha de Acesso",
        type="password",
        on_change=password_entered,
        key="password"
    )
    if "password_correct" in st.session_state:
        st.error("Senha incorreta. Tente novamente.")

    st.stop()


if not check_password():
    st.stop()


# ---------------------------------
# Inicializa o 'agente orquestrador'
# ---------------------------------
@st.cache_resource
def initialize_agent():
    # pega chave segura do .streamlit/secrets.toml
    openai_key = st.secrets["openai"]["api_key"]
    return get_agent(open_api_key=openai_key)

run_query = initialize_agent()


# ---------------------------------
# Estado da conversa
# ---------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # cada msg: {"role": "user"/"assistant", "content": "...", "sql_query": "..."?}


# ---------------------------------
# Render histórico (chat replay)
# ---------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

        # se for resposta do assistant e tiver SQL, mostra no expander
        if message["role"] == "assistant" and "sql_query" in message and message["sql_query"]:
            with st.expander("Ver SQL gerada"):
                st.code(message["sql_query"], language="sql")


# ---------------------------------
# Caixa de input do usuário
# ---------------------------------
if user_prompt := st.chat_input("Digite sua pergunta sobre os dados:"):
    # salva pergunta no histórico
    st.session_state.messages.append({
        "role": "user",
        "content": user_prompt
    })

    # render pergunta imediatamente
    with st.chat_message("user"):
        st.write(user_prompt)

    # gera resposta
    with st.chat_message("assistant"):
        with st.spinner("Consultando dados e gerando análise..."):
            try:
                # chama nossa função unificada
                response = run_query(user_prompt)

                # pega texto final amigável
                final_answer = response.get("output", "(sem retorno)")
                st.write(final_answer)

                # pega SQL pra auditoria
                sql_content = response.get("sql")
                if sql_content:
                    with st.expander("Ver SQL gerada"):
                        st.code(sql_content, language="sql")

                # salva resposta do assistant no histórico
                assistant_message = {
                    "role": "assistant",
                    "content": final_answer
                }
                if sql_content:
                    assistant_message["sql_query"] = sql_content

                st.session_state.messages.append(assistant_message)

            except Exception as e:
                error_message = f"Erro ao processar a consulta: {e}"
                st.error(error_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })
