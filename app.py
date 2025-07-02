import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch
import logging
from typing import Dict, Any
import time

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuração da página
st.set_page_config(
    page_title="AiiT - IA Multifunções", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS melhorado com responsividade
st.markdown("""
    <style>
        .main {
            padding: 1rem;
        }
        .demo-card {
            padding: 1.5rem;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
            color: white;
        }
        .stButton > button {
            background: linear-gradient(90deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.5rem 2rem;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
    </style>
""", unsafe_allow_html=True)

# Constantes
MAX_TEXT_LENGTH = 1000
DEFAULT_MODELS = {
    "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "summarization": "facebook/bart-large-cnn",
    "chat": "Qwen/Qwen1.5-0.5B-Chat"
}

class ModelManager:
    def __init__(self):
        self.models = {}

    @st.cache_resource
    def load_sentiment_model(_self):
        try:
            return pipeline("sentiment-analysis", model=DEFAULT_MODELS["sentiment"], return_all_scores=True)
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de sentimentos: {e}")
            return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

    @st.cache_resource
    def load_summarization_model(_self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODELS["summarization"])
            model = AutoModelForSeq2SeqLM.from_pretrained(DEFAULT_MODELS["summarization"])
            return {"tokenizer": tokenizer, "model": model}
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de sumarização: {e}")
            tokenizer = AutoTokenizer.from_pretrained("t5-small")
            model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
            return {"tokenizer": tokenizer, "model": model}

    @st.cache_resource
    def load_chat_model(_self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODELS["chat"])
            model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODELS["chat"])
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            return {"tokenizer": tokenizer, "model": model}
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de chat: {e}")
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", trust_remote_code=True)
            return {"tokenizer": tokenizer, "model": model}

def init_session_state():
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = ModelManager()
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Como posso te ajudar hoje?"}]

def chat_tab():
    st.markdown('<div class="demo-card">', unsafe_allow_html=True)

    for msg in st.session_state.messages:
        avatar = "🤖" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Digite sua mensagem..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Pensando..."):
                try:
                    model_data = st.session_state.model_manager.load_chat_model()
                    tokenizer = model_data["tokenizer"]
                    model = model_data["model"]

                    messages_for_model = [
                        {"role": "system", "content": "Você é um assistente IA útil."},
                        *st.session_state.messages[-4:]
                    ]

                    prompt_text = tokenizer.apply_chat_template(
                        messages_for_model,
                        add_generation_prompt=True,
                        tokenize=False
                    )

                    inputs = tokenizer(
                        prompt_text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True
                    ).to(model.device)

                    outputs = model.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_new_tokens=500,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9
                    )

                    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
                    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

                    message_placeholder = st.empty()
                    full_response = ""
                    for chunk in response.split():
                        full_response += chunk + " "
                        time.sleep(0.05)
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)

                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                except Exception as e:
                    st.error(f"Erro ao gerar resposta: {str(e)}")
                    logger.error(f"Erro no chat: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

def main():
    init_session_state()
    st.title("🤖 AiiT - IA Multifunções")
    tab1, tab2, tab3 = st.tabs(["Chat IA", "(placeholder)", "(placeholder)"])

    with tab1:
        chat_tab()

if __name__ == "__main__":
    main()
