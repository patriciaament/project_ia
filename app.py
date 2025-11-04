import streamlit as st
from agent import get_agent
import hmac

st.set_page_config(page_title="IA para Insights de Negócio")
st.title("🤖 IA para Insights de Negócio")


def check_password():
    def password_entered():
        if hmac.compare_digest(st.session_state["password"], st.secrets["auth"]["app_password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input("Senha de Acesso", type="password", on_change=password_entered, key="password")
    if "password_correct" in st.session_state:
        st.error("Senha incorreta. Tente novamente.")
    st.stop()


if not check_password():
    st.stop()

# ⚠️ sem cache pra pegar SEMPRE o agent.py novo
openai_key = st.secrets["openai"]["api_key"]
run_query = get_agent(open_api_key=openai_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        sql_msg = message.get("sql")
        if sql_msg:  # só mostra se tiver mesmo
            with st.expander("Ver SQL gerada"):
                st.code(sql_msg, language="sql")

if user_prompt := st.chat_input("Digite sua pergunta sobre os dados:"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando..."):
            try:
                resp = run_query(user_prompt)
                texto = resp.get("output", "")
                sql = resp.get("sql")

                st.write(texto)

                if sql:  # de novo: só mostra se vier
                    with st.expander("Ver SQL gerada"):
                        st.code(sql, language="sql")

                st.session_state.messages.append(
                    {"role": "assistant", "content": texto, "sql": sql}
                )
            except Exception as e:
                st.error(f"Erro ao processar: {e}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Erro ao processar: {e}"}
                )
