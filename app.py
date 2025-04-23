import streamlit as st
from transformers import pipeline
import time

# Carregar os modelos de IA relevantes para cada serviço
@st.cache_resource
def load_models():
    sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
    classification_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
    summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")
    chatbot = pipeline("conversational", model="microsoft/DialoGPT-medium")
    image_classifier = pipeline("image-classification", model="resnet50")
    
    return sentiment_model, classification_model, summarizer, chatbot, image_classifier

# Carregar modelos quando a aplicação iniciar
sentiment_model, classification_model, summarizer, chatbot, image_classifier = load_models()

# Título da App
st.title("💡 Aplicação de Serviços de IA e Robótica")

# Menu lateral para selecionar o serviço
menu = st.sidebar.radio("Escolha um serviço:", [
    "Consultoria em IA e Robótica",
    "Desenvolvimento de Sistemas Inteligentes",
    "Automação de Processos com RPA",
    "Desenvolvimento de Robôs Industriais",
    "Visão Computacional",
    "Soluções para IoT",
    "Análise de Dados com IA",
    "Desenvolvimento de Chatbots",
    "Integração de Sistemas de IA"
])

# Funções específicas para cada serviço
def analyze_sentiment(text):
    return sentiment_model(text)

def analyze_classification(text):
    return classification_model(text)

def summarize_text(text):
    return summarizer(text)

def chatbot_response(user_message):
    return chatbot(user_message)

def analyze_image(uploaded_image):
    return image_classifier(uploaded_image)

if menu == "Consultoria em IA e Robótica":
    st.write("""
        **Consultoria personalizada em IA e Robótica** para sua empresa. 
        Oferecemos soluções que combinam automação e inteligência artificial para melhorar a eficiência.
    """)
    user_input = st.text_area("Conte-nos sobre seu projeto:")
    if len(user_input) == 0:
        st.warning("Por favor, insira um texto com detalhes sobre seu projeto para solicitar consultoria.")
    elif st.button("Solicitar Consultoria"):
        st.success(f"Consultoria solicitada para: {user_input}")

elif menu == "Desenvolvimento de Sistemas Inteligentes":
    st.write("""
        **Desenvolvimento de Sistemas Inteligentes** usando IA para otimizar processos e automação em sua empresa.
    """)
    user_input = st.text_area("Digite um texto para análise de sistema inteligente:")
    if len(user_input) == 0:
        st.warning("Por favor, insira um texto para análise.")
    elif st.button("Analisar Texto"):
        with st.spinner("Analisando..."):
            result = analyze_sentiment(user_input)
        st.write(f"Resultado da Análise: {result}")

elif menu == "Automação de Processos com RPA":
    st.write("""
        **Automação de processos com RPA**, otimização de tarefas repetitivas para reduzir custos e melhorar a eficiência operacional.
    """)
    process_choice = st.selectbox("Escolha o processo para automatizar:", ["Processamento de Faturas", "Gestão de Estoque", "Atendimento ao Cliente"])
    if st.button("Automatizar"):
        st.success(f"Processo {process_choice} automatizado com sucesso!")

elif menu == "Desenvolvimento de Robôs Industriais":
    st.write("""
        **Desenvolvimento de robôs industriais** para automação de linhas de produção e tarefas industriais com alta precisão e eficiência.
    """)
    robot_type = st.selectbox("Escolha o tipo de robô:", ["Robô Colaborativo", "Veículo Autônomo", "Robô de Inspeção"])
    if st.button("Configurar Robô"):
        st.success(f"Robô {robot_type} configurado com sucesso!")

elif menu == "Visão Computacional":
    st.write("""
        **Visão Computacional** para reconhecimento e análise de imagens em tempo real.
    """)
    uploaded_image = st.file_uploader("Carregue uma imagem para análise", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Imagem carregada", use_column_width=True)
        with st.spinner("Analisando imagem..."):
            result = analyze_image(uploaded_image)
        st.write(f"Classificação da Imagem: {result}")

elif menu == "Soluções para IoT":
    st.write("""
        **Soluções IoT** que conectam dispositivos inteligentes para monitoramento em tempo real e automação de processos.
    """)
    device_status = st.radio("Status do Dispositivo IoT", ["Ativo", "Inativo"])
    if st.button("Monitorar"):
        st.success(f"Dispositivo {device_status} monitorado com sucesso.")

elif menu == "Análise de Dados com IA":
    st.write("""
        **Análise de Dados com IA** para fornecer insights valiosos e otimizar a tomada de decisões em sua empresa.
    """)
    data_input = st.text_area("Digite os dados para análise de IA:")
    if len(data_input) == 0:
        st.warning("Por favor, insira os dados para análise.")
    elif st.button("Analisar Dados"):
        with st.spinner("Analisando..."):
            result = analyze_sentiment(data_input)
        st.write(f"Resultado da Análise: {result}")

elif menu == "Desenvolvimento de Chatbots":
    st.write("""
        **Desenvolvimento de Chatbots** para automatizar o atendimento ao cliente e aumentar a eficiência do suporte.
    """)
    user_message = st.text_input("Digite sua pergunta:")
    if len(user_message) == 0:
        st.warning("Por favor, digite uma pergunta para o chatbot.")
    elif st.button("Enviar"):
        with st.spinner("Chatbot processando..."):
            response = chatbot_response(user_message)
        st.write(f"Resposta do Chatbot: {response[0]['generated_text']}")

elif menu == "Integração de Sistemas de IA":
    st.write("""
        **Integração de IA com Sistemas Existentes** para garantir uma transição suave e a maximização de valor com soluções de IA.
    """)
    integration_type = st.selectbox("Escolha o tipo de integração:", ["Integração com CRM", "Integração com ERP", "Integração com Sistema de Atendimento"])
    if st.button("Iniciar Integração"):
        st.success(f"Integração com {integration_type} iniciada com sucesso!")

