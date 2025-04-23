import streamlit as st
from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
# Carregar os modelos de IA relevantes para cada serviço
sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
classification_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")
chatbot = pipeline("text-generation", model="gpt2")
image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224-in21k")
audio_classifier = pipeline("audio-classification", model="facebook/wav2vec2-large-xlsr-53")
speech_to_text = pipeline("automatic-speech-recognition", model="facebook/wav2vec2-large-xlsr-53")
object_detection = pipeline("object-detection", model="facebook/detectron2")
question_answering = pipeline("question-answering", model="deepset/roberta-base-squad2")
translation = pipeline("translation_en_to_fr", model="t5-small")

# Título da App
st.title("💡 Aplicação de Serviços de IA e Robótica")

# Menu lateral para selecionar o serviço
menu = st.sidebar.radio("Escolha um serviço:", [
    "Análise de Sentimentos",
    "Classificação de Texto",
    "Resumos Automáticos",
    "Desenvolvimento de Chatbots",
    "Classificação de Imagens",
    "Análise de Áudio",
    "Transcrição de Fala",
    "Detecção de Objetos",
    "Resposta a Perguntas",
    "Tradução Automática"
])

# Funções específicas para cada serviço
if menu == "Análise de Sentimentos":
    st.write("""
        **Análise de Sentimentos**: Descubra como as pessoas estão se sentindo sobre um tópico ou conteúdo.
    """)
    user_input = st.text_area("Digite o texto para análise de sentimentos:")
    if st.button("Analisar Sentimento"):
        result = sentiment_model(user_input)
        st.write(f"Resultado da Análise de Sentimentos: {result}")

elif menu == "Classificação de Texto":
    st.write("""
        **Classificação de Texto**: Classifique o texto em categorias específicas.
    """)
    user_input = st.text_area("Digite o texto para classificação:")
    if st.button("Classificar Texto"):
        result = classification_model(user_input)
        st.write(f"Resultado da Classificação: {result}")

elif menu == "Resumos Automáticos":
    st.write("""
        **Resumos Automáticos**: Resuma textos longos de forma rápida e eficiente.
    """)
    user_input = st.text_area("Digite o texto para resumo:")
    if st.button("Gerar Resumo"):
        result = summarizer(user_input)
        st.write(f"Resumo Gerado: {result[0]['summary_text']}")

elif menu == "Desenvolvimento de Chatbots":
    st.write("""
        **Desenvolvimento de Chatbots**: Converse com um chatbot inteligente.
    """)
    user_message = st.text_input("Digite sua pergunta:")
    if st.button("Enviar"):
        response = chatbot(user_message, max_length=60, num_return_sequences=1)
        st.write(f"Resposta do Chatbot: {response[0]['generated_text']}")

elif menu == "Classificação de Imagens":
    st.write("""
        **Classificação de Imagens**: Classifique uma imagem com base em categorias predefinidas.
    """)
    uploaded_image = st.file_uploader("Carregue uma imagem para classificação", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Imagem carregada", use_column_width=True)
        result = image_classifier(uploaded_image)
        st.write(f"Classificação da Imagem: {result}")

elif menu == "Análise de Áudio":
    st.write("""
        **Análise de Áudio**: Classifique áudios em diferentes categorias.
    """)
    uploaded_audio = st.file_uploader("Carregue um arquivo de áudio para classificação", type=["mp3", "wav"])
    if uploaded_audio is not None:
        st.audio(uploaded_audio, format="audio/wav")
        result = audio_classifier(uploaded_audio)
        st.write(f"Classificação do Áudio: {result}")

elif menu == "Transcrição de Fala":
    st.write("""
        **Transcrição de Fala**: Converta fala em texto automaticamente.
    """)
    uploaded_audio = st.file_uploader("Carregue um arquivo de áudio para transcrição", type=["mp3", "wav"])
    if uploaded_audio is not None:
        st.audio(uploaded_audio, format="audio/wav")
        result = speech_to_text(uploaded_audio)
        st.write(f"Texto Transcrito: {result['text']}")

elif menu == "Detecção de Objetos":
    st.write("""
        **Detecção de Objetos**: Detecte objetos em imagens enviadas.
    """)
    uploaded_image = st.file_uploader("Carregue uma imagem para detectar objetos", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Imagem carregada", use_column_width=True)
        result = object_detection(uploaded_image)
        st.write(f"Objetos Detectados: {result}")

elif menu == "Resposta a Perguntas":
    st.write("""
        **Resposta a Perguntas**: Pergunte algo e receba uma resposta com base em um conjunto de dados.
    """)
    context = st.text_area("Digite o contexto para perguntas:")
    question = st.text_input("Digite a pergunta:")
    if st.button("Responder"):
        result = question_answering(question=question, context=context)
        st.write(f"Resposta: {result['answer']}")

elif menu == "Tradução Automática":
    st.write("""
        **Tradução Automática**: Traduza textos de inglês para francês automaticamente.
    """)
    user_input = st.text_area("Digite o texto em inglês para traduzir:")
    if st.button("Traduzir"):
        result = translation(user_input)
        st.write(f"Texto Traduzido: {result[0]['translation_text']}")

