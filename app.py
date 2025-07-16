import streamlit as st
import requests

API_BASE = "http://0.0.0.0:8000"

st.set_page_config(page_title="RAG Chatbot", layout="centered")
st.title("RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

try:
    if question := st.chat_input("Enter your question"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        context = "Context:" + "\n".join([f"{m['role'].capitalize()}:{m['content']}" for m in st.session_state.messages])
        last_query = st.session_state.messages[-1]['content']
        response = requests.post(
            url=f"{API_BASE}/chat",
            json={"query": last_query}
        )
        response.raise_for_status()

        full_response = response.json()
        with st.chat_message("assistant"):
            st.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
except Exception as e:
    st.error(f"Error occurred: {e}")