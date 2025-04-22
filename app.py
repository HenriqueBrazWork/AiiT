import streamlit as st
from transformers import pipeline

# Carregar os modelos de IA relevantes para cada serviço
sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
classification_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")
chatbot = pipeline("text-generation", model="gpt2")
image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224-in21k")

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
if menu == "Consultoria em IA e Robótica":
    st.write("""
        **Consultoria personalizada em IA e Robótica** para sua empresa. 
        Oferecemos soluções que combinam automação e inteligência artificial para melhorar a eficiência.
    """)
    user_input = st.text_area("Conte-nos sobre seu projeto:")
    if st.button("Solicitar Consultoria"):
        st.write(f"Consultoria solicitada para: {user_input}")

elif menu == "Desenvolvimento de Sistemas Inteligentes":
    st.write("""
        **Desenvolvimento de Sistemas Inteligentes** usando IA para otimizar processos e automação em sua empresa.
    """)
    user_input = st.text_area("Digite um texto para análise de sistema inteligente:")
    if st.button("Analisar Texto"):
        result = sentiment_model(user_input)
        st.write(f"Resultado da Análise: {result}")

elif menu == "Automação de Processos com RPA":
    st.write("""
        **Automação de processos com RPA**, otimização de tarefas repetitivas para reduzir custos e melhorar a eficiência operacional.
    """)
    process_choice = st.selectbox("Escolha o processo para automatizar:", ["Processamento de Faturas", "Gestão de Estoque", "Atendimento ao Cliente"])
    if st.button("Automatizar"):
        st.write(f"Processo {process_choice} automatizado com sucesso!")

elif menu == "Desenvolvimento de Robôs Industriais":
    st.write("""
        **Desenvolvimento de robôs industriais** para automação de linhas de produção e tarefas industriais com alta precisão e eficiência.
    """)
    robot_type = st.selectbox("Escolha o tipo de robô:", ["Robô Colaborativo", "Veículo Autônomo", "Robô de Inspeção"])
    if st.button("Configurar Robô"):
        st.write(f"Robô {robot_type} configurado com sucesso!")

elif menu == "Visão Computacional":
    st.write("""
        **Visão Computacional** para reconhecimento e análise de imagens em tempo real.
    """)
    uploaded_image = st.file_uploader("Carregue uma imagem para análise", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        st.image(uploaded_image, caption="Imagem carregada", use_column_width=True)
        result = image_classifier(uploaded_image)
        st.write(f"Classificação da Imagem: {result}")

elif menu == "Soluções para IoT":
    st.write("""
        **Soluções IoT** que conectam dispositivos inteligentes para monitoramento em tempo real e automação de processos.
    """)
    device_status = st.radio("Status do Dispositivo IoT", ["Ativo", "Inativo"])
    if st.button("Monitorar"):
        st.write(f"Dispositivo {device_status} monitorado com sucesso.")

elif menu == "Análise de Dados com IA":
    st.write("""
        **Análise de Dados com IA** para fornecer insights valiosos e otimizar a tomada de decisões em sua empresa.
    """)
    data_input = st.text_area("Digite os dados para análise de IA:")
    if st.button("Analisar Dados"):
        result = sentiment_model(data_input)
        st.write(f"Resultado da Análise: {result}")

elif menu == "Desenvolvimento de Chatbots":
    st.write("""
        **Desenvolvimento de Chatbots** para automatizar o atendimento ao cliente e aumentar a eficiência do suporte.
    """)
    user_message = st.text_input("Digite sua pergunta:")
    if st.button("Enviar"):
        response = chatbot(user_message, max_length=60, num_return_sequences=1)
        st.write(f"Resposta do Chatbot: {response[0]['generated_text']}")

elif menu == "Integração de Sistemas de IA":
    st.write("""
        **Integração de IA com Sistemas Existentes** para garantir uma transição suave e a maximização de valor com soluções de IA.
    """)
    integration_type = st.selectbox("Escolha o tipo de integração:", ["Integração com CRM", "Integração com ERP", "Integração com Sistema de Atendimento"])
    if st.button("Iniciar Integração"):
        st.write(f"Integração com {integration_type} iniciada com sucesso!")

