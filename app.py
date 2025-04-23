import streamlit as st
import torch
from transformers import pipeline
from PIL import Image

# Determine device
device = 0 if torch.cuda.is_available() else -1

# Load pipelines
@st.cache_resource
def load_pipelines():
    return {
        "sentiment": pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device=device),
        "summarizer": pipeline("summarization", model="facebook/bart-large-cnn", device=device),
        "image_clf": pipeline("image-classification", model="microsoft/resnet-50", device=device),
        "asr": pipeline("automatic-speech-recognition", model="facebook/wav2vec2-large-960h-lv60-self", device=device),
        "sentiment_en": pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device),
        "translator": pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-pt"), device=device),
        "qa": pipeline("question-answering", model="deepset/roberta-base-squad2", device=device)
    }

pipes = load_pipelines()

st.title("💼 Exemplos de Soluções de Mercado com IA")

task = st.sidebar.selectbox("Selecionar solução de mercado:", [
    "1. Análise de Sentimentos (E-commerce)",
    "2. Resumo de Artigo (Mídia)",
    "3. Classificação de Imagens (Indústria)",
    "4. Transcrição e Sentimento (Call Center)",
    "5. Tradução de Descrição de Produto",
    "6. Perguntas & Respostas (FAQ)"
])

if task.startswith("1"):
    st.header("Análise de Sentimentos de Reviews")
    reviews = st.text_area("Insira as avaliações (uma por linha):")
    if st.button("Analisar Sentimentos"):
        texts = [r for r in reviews.split("\n") if r.strip()]
        with st.spinner("Analisando..."):
            results = [pipes["sentiment"](text)[0] for text in texts]
        for text, res in zip(texts, results):
            st.write(f"> **{text}** → {res['label']} ({res['score']:.2f})")

elif task.startswith("2"):
    st.header("Resumo Automático de Artigo")
    article = st.text_area("Cole o texto do artigo:")
    if st.button("Gerar Resumo"):
        with st.spinner("Sumarizando..."):
            summary = pipes["summarizer"](article, max_length=60, min_length=20, do_sample=False)[0]["summary_text"]
        st.success(summary)

elif task.startswith("3"):
    st.header("Classificação de Imagens para Controle de Qualidade")
    uploaded_image = st.file_uploader("Envie uma imagem da peça:", type=["jpg","png","jpeg"])
    if uploaded_image:
        img = Image.open(uploaded_image)
        st.image(img, use_column_width=True)
        if st.button("Classificar Imagem"):
            with st.spinner("Classificando..."):
                preds = pipes["image_clf"](img)
            st.json(preds)

elif task.startswith("4"):
    st.header("Transcrição e Sentimento de Áudio (Call Center)")
    uploaded_audio = st.file_uploader("Envie um áudio de chamada:", type=["wav","mp3"])
    if uploaded_audio:
        st.audio(uploaded_audio)
        if st.button("Transcrever + Analisar Sentimento"):
            with st.spinner("Processando áudio..."):
                transcript = pipes["asr"](uploaded_audio)["text"]
                sentiment = pipes["sentiment_en"](transcript)[0]
            st.subheader("Transcrição")
            st.write(transcript)
            st.subheader("Sentimento")
            st.write(f"{sentiment['label']} ({sentiment['score']:.2f})")

elif task.startswith("5"):
    st.header("Tradução de Descrição de Produto (EN→PT)")
    desc = st.text_area("Descrição em Inglês:")
    if st.button("Traduzir"):
        with st.spinner("Traduzindo..."):
            tr = pipes["translator"](desc)[0]["translation_text"]
        st.success(tr)

elif task.startswith("6"):
    st.header("Sistema de Perguntas & Respostas (FAQ)")
    context = st.text_area("Contexto / Documento:")
    question = st.text_input("Pergunta:")
    if st.button("Responder"):
        with st.spinner("Buscando resposta..."):
            ans = pipes["qa"](question=question, context=context)["answer"]
        st.subheader("Resposta")
        st.write(ans)
