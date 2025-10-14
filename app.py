import streamlit as st
from agent import get_agent
import hmac 

st.set_page_config(page_title="IA para Insights de Negócio", layout="wide")
st.title("🤖 IA para Consultas SQL")


def check_password():

    def password_entered():
        if hmac.compare_digest(st.session_state["password"], st.secrets["auth"]["app_password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"] 
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.text_input(
        "Senha de Acesso", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("Senha incorreta. Tente novamente.")

    st.stop()


if not check_password():
    st.stop()

@st.cache_resource
def initialize_agent():
    return get_agent()

agent_executor = initialize_agent()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        if message["role"] == "assistant" and "sql_query" in message:
            with st.expander("Ver SQL gerada"):
                st.code(message["sql_query"], language="sql")

if user_prompt := st.chat_input("Digite sua pergunta sobre os dados:"):
    
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    
    with st.chat_message("user"):
        st.write(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Gerando e executando a consulta..."):
            
            try:
                response = agent_executor(user_prompt)
                
                final_answer = response["output"]
                st.success("Consulta executada com sucesso!")
                st.write(final_answer)

                sql_content = None
                try:
                    sql_content = response["intermediate_steps"][-1][0].tool_input
                    
                except Exception:
                    pass
                
                assistant_message = {
                    "role": "assistant", 
                    "content": final_answer
                }
                
                if sql_content:
                    assistant_message["sql_query"] = sql_content
                    with st.expander("Ver SQL gerada"):
                        st.code(sql_content, language="sql")
                
                st.session_state.messages.append(assistant_message)


            except Exception as e:
                error_message = f"Erro ao processar a consulta: {e}"
                st.error(error_message)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_message
                })
