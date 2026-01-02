import streamlit as st
import requests

API_URL = "http://localhost:8000/chat"

st.set_page_config(page_title="AiiT SmartChat", layout="centered")
st.title("🤖 AiiT SmartChat")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Escreve a tua pergunta...")

if user_input:
    st.chat_message("user").write(user_input)
    response = requests.post(API_URL, json={"message": user_input}).json()["response"]
    st.chat_message("assistant").write(response)

    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": response})
