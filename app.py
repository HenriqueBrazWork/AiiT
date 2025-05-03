import streamlit as st
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification, Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torch

# Definir o dispositivo (GPU ou CPU)
device = 0 if torch.cuda.is_available() else -1

# Carregar os modelos
models = {
    'sentiment_analysis': pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment", device=device),
    'text_classification': pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english", device=device),
    'summarization': pipeline("summarization", model="t5-small", device=device),
    'chatbot': pipeline("text-generation", model="gpt2", device=device),
    'image_classifier': {
        "processor": AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k"),
        "model": AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")
    },
    'audio_classifier': {
        "processor" = AutoProcessor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        "model" = AutoModelForPreTraining.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    },
    'speech_to_text': pipeline("automatic-speech-recognition", model="facebook/wav2vec2-large-xlsr-53", device=device),
    'object_detection': pipeline("object-detection", model="facebook/detectron2", device=device),
    'question_answering': pipeline("question-answering", model="deepset/roberta-base-squad2", device=device),
    'translation': pipeline("translation_en_to_fr", model="t5-small", device=device)
}

# Função de uso dos modelos
def use_model(model_key, input_text=None, input_audio=None, input_image=None):
    model = models[model_key]

    if model_key == 'sentiment_analysis' and input_text:
        return model(input_text)
    elif model_key == 'text_classification' and input_text:
        return model(input_text)
    elif model_key == 'summarization' and input_text:
        return model(input_text)
    elif model_key == 'chatbot' and input_text:
        return model(input_text)
    elif model_key == 'image_classifier' and input_image:
        inputs = image_processor(images=input_image, return_tensors="pt")
        outputs = image_model(**inputs)
        return outputs
    elif model_key == 'audio_classifier' and input_audio:
        audio_input = audio_processor(input_audio, return_tensors="pt", sampling_rate=16000)
        outputs = model["model"](**audio_input)
        return outputs
    elif model_key == 'speech_to_text' and input_audio:
        return model(input_audio)
    elif model_key == 'object_detection' and input_image:
        return model(input_image)
    elif model_key == 'question_answering' and input_text:
        return model(input_text)
    elif model_key == 'translation' and input_text:
        return model(input_text)
    else:
        return "Modelo ou entrada não reconhecida."

# Streamlit Interface
st.title("Aplicação de IA com Modelos da Hugging Face")

# Seleção do modelo
model_key = st.selectbox(
    "Escolha o modelo",
    [
        'sentiment_analysis',
        'text_classification',
        'summarization',
        'chatbot',
        'image_classifier',
        'audio_classifier',
        'speech_to_text',
        'object_detection',
        'question_answering',
        'translation'
    ]
)

# Entrada do usuário
if model_key in ['sentiment_analysis', 'text_classification', 'summarization', 'chatbot', 'question_answering', 'translation']:
    input_text = st.text_area(f"Digite o texto para {model_key.replace('_', ' ')}:")
    if st.button(f"Executar {model_key.replace('_', ' ')}"):
        if input_text:
            result = use_model(model_key, input_text=input_text)
            st.write(result)
        else:
            st.write("Por favor, insira um texto.")
elif model_key in ['image_classifier', 'object_detection']:
    input_image = st.file_uploader("Carregue uma imagem", type=["jpg", "png", "jpeg"])
    if st.button(f"Executar {model_key.replace('_', ' ')}"):
        if input_image:
            result = use_model(model_key, input_image=input_image)
            st.image(input_image, caption="Imagem carregada.", use_column_width=True)
            st.write(result)
        else:
            st.write("Por favor, carregue uma imagem.")
elif model_key in ['audio_classifier', 'speech_to_text']:
    input_audio = st.file_uploader("Carregue um arquivo de áudio", type=["wav", "mp3", "flac"])
    if st.button(f"Executar {model_key.replace('_', ' ')}"):
        if input_audio:
            result = use_model(model_key, input_audio=input_audio)
            st.audio(input_audio, format="audio/wav")
            st.write(result)
        else:
            st.write("Por favor, carregue um arquivo de áudio.")
