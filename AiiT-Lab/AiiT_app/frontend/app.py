# frontend/app.py
import streamlit as st
import requests
from transformers import pipeline

API_URL = "http://localhost:8000"

st.title("AiiT Services – Plataforma Profissional de IA")

# --- Login JWT ---
token = st.text_input("JWT Token", type="password")
if token:
    res = requests.get(f"{API_URL}/me", headers={"Authorization": f"Bearer {token}"})
    if res.status_code == 200:
        user = res.json()
        st.success(f"Bem-vindo, {user['email']}!")
    else:
        st.error("Token inválido!")

# --- Pagamentos ---
plan = st.selectbox("Escolha o plano:", ["basic", "pro", "enterprise"])
if st.button("Pagar"):
    if token:
        res = requests.post(f"{API_URL}/pay/{plan}", headers={"Authorization": f"Bearer {token}"})
        if res.status_code == 200:
            st.markdown(f"[Clique para pagar]({res.json()['checkout_url']})")
        else:
            st.error("Erro ao gerar checkout")
    else:
        st.warning("Faça login primeiro!")

# --- Modelos de IA ---
st.header("Executar Modelos")
task = st.selectbox("Escolha a tarefa:", ["Texto", "Imagem", "Áudio"])
if task == "Texto":
    prompt = st.text_area("Prompt")
    if st.button("Executar Texto"):
        generator = pipeline("text-generation")
        result = generator(prompt, max_length=150)[0]["generated_text"]
        st.write(result)
