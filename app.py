import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv("API_BASE")

st.set_page_config(page_title="Legal Chatbot", layout="centered")
st.title("💬 Legal Chatbot")

with st.sidebar:
    st.header("🛠 Option")
    if st.button("🆕 New chat"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

try:
    if question := st.chat_input("Enter your question here"):
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        chat_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

        with st.chat_message("assistant"):
            stream_placeholder = st.empty()
            full_response = ""

            response = requests.post(
            url=f"{API_BASE}/chat/stream",
            json={"question": question, "chat_history": chat_history},
            stream=True
            )
                
            for chunk in response:
                full_response += chunk.decode("utf-8")
                stream_placeholder.markdown(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

except Exception as e:
    st.error(f"Error occurred: {e}")
