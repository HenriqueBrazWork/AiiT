import streamlit as st
from transformers import pipeline

# Forçar tema claro com CSS customizado
st.markdown("""
    <style>
        .main {
            background-color: white;
            color: black;
        }
        .css-18e3th9 {
            background-color: white;
        }
        .css-1d391kg {
            background-color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Carregar pipelines
sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
classification_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")
chatbot = pipeline("text-generation", model="gpt2")

st.title("🧠 App Inteligente com IA")

menu = st.sidebar.radio("Escolha uma função:", ["🔍 Análise de Sentimento", "🏷️ Classificação de Texto", "📝 Geração de Resumo", "💬 Resposta Automática"])

user_input = st.text_area("Digite seu texto aqui:")

if st.button("Executar"):
    if not user_input.strip():
        st.warning("Por favor, insira um texto.")
    else:
        if menu == "🔍 Análise de Sentimento":
            result = sentiment_model(user_input)
            st.json(result)

        elif menu == "🏷️ Classificação de Texto":
            result = classification_model(user_input)
            st.json(result)

        elif menu == "📝 Geração de Resumo":
            prompt = "summarize: " + user_input
            result = summarizer(prompt, max_length=60, min_length=20, do_sample=False)
            st.success(result[0]['summary_text'])

        elif menu == "💬 Resposta Automática":
            result = chatbot(user_input, max_length=60, num_return_sequences=1)
            st.success(result[0]['generated_text'])

st.sidebar.markdown("---")
st.sidebar.info("💡 Feito com modelos BERT, T5 e GPT.")
