import asyncio
import logging
from aiortc import VideoStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer
import streamlit as st
from transformers import (
    pipeline,
    AutoImageProcessor, AutoModelForImageClassification, 
    Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
)
import torch
from PIL import Image
import torchaudio

# Configuração de logging
logging.basicConfig(level=logging.INFO)

# Função para criar os pipelines de IA (exemplo de vários modelos)
def create_models():
    device = 0 if torch.cuda.is_available() else -1
    
    models = {
        'sentiment_analysis': pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device=device),
        'text_classification': pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", device=device),
        'summarization': pipeline("summarization", model="t5-small", device=device),
        'chatbot': pipeline("text-generation", model="gpt2", device=device),
        'image_classifier': {
            "processor": AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k"),
            "model": AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")
        },
        'speech_to_text': pipeline("automatic-speech-recognition", model="facebook/wav2vec2-large-xlsr-53", device=device),
        'object_detection': pipeline("object-detection", model="facebook/detr-resnet-50", device=device),
        'question_answering': pipeline("question-answering", model="deepset/roberta-base-squad2", device=device),
        'translation': pipeline("translation_en_to_fr", model="t5-small", device=device)
    }
    
    return models

# Função WebRTC para lidar com a captura de áudio/vídeo
class VideoTrack(VideoStreamTrack):
    def __init__(self):
        super().__init__()
        self._player = MediaPlayer("video.mp4")  # Pode ser um arquivo de vídeo ou capturador de câmera

    async def recv(self):
        frame = self._player.next_frame()  # Captura o próximo frame do vídeo
        return frame


# Função principal para gerenciar a conexão WebRTC e usar IA
async def run_webrtc():
    pc = RTCPeerConnection()
    
    # Adicionar um track de vídeo ao PeerConnection
    video_track = VideoTrack()
    pc.addTrack(video_track)

    # Configuração do offer (oferta para negociação da conexão)
    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)

    # Exemplo para integrar com o Streamlit (em uma aplicação real isso seria feito em um servidor WebRTC)
    # Aqui você pode obter a resposta do cliente WebRTC
    answer = RTCSessionDescription(sdp="YOUR_SDP_HERE", type="answer")
    await pc.setRemoteDescription(answer)

    # Iniciar o servidor WebRTC
    await asyncio.gather(pc.wait_closed())

# Função de uso dos modelos
def use_model(model_key, models, input_text=None, input_audio=None, input_image=None):
    model = models[model_key]

    if model_key == 'sentiment_analysis' and input_text:
        return model(input_text)

    elif model_key == 'text_classification' and input_text:
        return model(input_text)

    elif model_key == 'summarization' and input_text:
        return model(input_text)

    elif model_key == 'chatbot' and input_text:
        return model(input_text, max_length=100, num_return_sequences=1)[0]['generated_text']

    elif model_key == 'image_classifier' and input_image:
        image = Image.open(input_image).convert("RGB")
        processor = model["processor"]
        image_model = model["model"]
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = image_model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        label = image_model.config.id2label[predicted_class_idx]
        return f"Classe prevista: {label}"

    elif model_key == 'speech_to_text' and input_audio:
        return model(input_audio)

    elif model_key == 'object_detection' and input_image:
        image = Image.open(input_image).convert("RGB")
        return model(image)

    elif model_key == 'question_answering' and input_text:
        context = st.text_area("Digite o contexto:", height=200)
        if context:
            return model(question=input_text, context=context)
        else:
            return "Por favor, forneça o contexto para a pergunta."

    elif model_key == 'translation' and input_text:
        return model(input_text)

    else:
        return "Modelo ou entrada não reconhecida."

# Interface Streamlit
st.title("Aplicação de IA com Modelos da Hugging Face")

# Carregar os modelos
models = create_models()

# Seleção do modelo
model_key = st.selectbox(
    "Escolha o modelo",
    [
        'sentiment_analysis',
        'text_classification',
        'summarization',
        'chatbot',
        'image_classifier',
        'speech_to_text',
        'object_detection',
        'question_answering',
        'translation'
    ]
)

# Entrada do usuário conforme o tipo de modelo
if model_key in ['sentiment_analysis', 'text_classification', 'summarization', 'chatbot', 'question_answering', 'translation']:
    input_text = st.text_area(f"Digite o texto para {model_key.replace('_', ' ')}:")
    if st.button(f"Executar {model_key.replace('_', ' ')}"):
        if input_text:
            result = use_model(model_key, models, input_text=input_text)
            st.write(result)
        else:
            st.warning("Por favor, insira um texto.")

elif model_key in ['image_classifier', 'object_detection']:
    input_image = st.file_uploader("Carregue uma imagem", type=["jpg", "jpeg", "png"])
    if st.button(f"Executar {model_key.replace('_', ' ')}"):
        if input_image:
            result = use_model(model_key, models, input_image=input_image)
            st.image(input_image, caption="Imagem carregada.", use_column_width=True)
            st.write(result)
        else:
            st.warning("Por favor, carregue uma imagem.")

elif model_key in ['speech_to_text']:
    input_audio = st.file_uploader("Carregue um arquivo de áudio", type=["wav", "mp3", "flac"])
    if st.button(f"Executar {model_key.replace('_', ' ')}"):
        if input_audio:
            result = use_model(model_key, models, input_audio=input_audio)
            st.audio(input_audio, format="audio/wav")
            st.write(result)
        else:
            st.warning("Por favor, carregue um arquivo de áudio.")

elif model_key == 'chatbot':
    input_text = st.text_area("Digite uma mensagem para o chatbot:")
    if st.button(f"Enviar mensagem"):
        if input_text:
            result = use_model(model_key, models, input_text=input_text)
            st.write(result)
        else:
            st.warning("Por favor, insira uma mensagem.")

# Executar WebRTC para capturar vídeo
if st.button("Iniciar WebRTC"):
    st.write("Conectando ao WebRTC...")
    asyncio.run(run_webrtc())  # Inicia a captura WebRTC em paralelo
