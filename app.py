import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch

st.set_page_config(page_title="AiiT - IA Multifunções", layout="wide")
st.markdown("""
    <style>
        body {
            background-color: #f9f9f9;
            color: #333333;
            font-family: 'Segoe UI', sans-serif;
        }
        .main {
            background-color: #ffffff;
            padding: 2rem;
            border-radius: 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #004d99;
        }
        .stTextInput>div>div>input {
            padding: 10px;
            font-size: 16px;
        }
        .stButton>button {
            background-color: #004d99;
            color: white;
            border-radius: 0.5rem;
            padding: 10px 20px;
        }
        .demo-card {
            padding: 2rem;
            background-color: #ffffff;
            border-radius: 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 AiiT - Aplicação de IA Multifunções")

models = {}

@st.cache_resource
def load_lite_models():
    sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    summarization_tokenizer = AutoTokenizer.from_pretrained("t5-small")
    summarization_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", trust_remote_code=True)
    qwen_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", trust_remote_code=True)
    return {
        "sentiment": sentiment_pipeline,
        "summarization": {
            "tokenizer": summarization_tokenizer,
            "model": summarization_model
        },
        "qwen": {
            "tokenizer": qwen_tokenizer,
            "model": qwen_model
        }
    }

with st.spinner("🔄 A carregar modelos..."):
    models = load_lite_models()

tab1, tab2, tab3 = st.tabs(["😊 Análise de Sentimentos", "📄 Resumo de Texto", "🤖 Chat IA (Qwen)"])

# --- TAB 1: Análise de Sentimentos ---
with tab1:
    st.markdown('<div class="demo-card">', unsafe_allow_html=True)
    st.subheader("😊 Análise de Sentimentos")
    st.write("Introduz um texto para obter a análise de sentimento.")

    user_input_sentiment = st.text_area("Texto:", height=150, key="sentiment_input")

    if st.button("🔍 Analisar Sentimento", key="sentiment_button"):
        if user_input_sentiment and 'sentiment' in models:
            result = models["sentiment"](user_input_sentiment)
            st.write("**Resultado:**")
            st.json(result)
        else:
            st.warning("⚠️ Introduz um texto para análise.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 2: Resumo de Texto ---
with tab2:
    st.markdown('<div class="demo-card">', unsafe_allow_html=True)
    st.subheader("📄 Resumo de Texto")
    st.write("Insere um texto longo para gerar um resumo.")

    user_input_summary = st.text_area("Texto para resumir:", height=200, key="summary_input")

    if st.button("📄 Resumir", key="summary_button"):
        if user_input_summary and 'summarization' in models:
            tokenizer = models["summarization"]["tokenizer"]
            model = models["summarization"]["model"]

            inputs = tokenizer("summarize: " + user_input_summary, return_tensors="pt", max_length=512, truncation=True)
            summary_ids = model.generate(inputs["input_ids"], max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            st.write("**Resumo:**")
            st.success(summary)
        else:
            st.warning("⚠️ Introduz um texto válido.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 3: Chat IA com Qwen ---
with tab3:
    st.markdown('<div class="demo-card">', unsafe_allow_html=True)
    st.subheader("🤖 Chat com IA - Modelo Qwen")
    st.write("Converse com uma IA baseada no modelo Qwen")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    user_input_qwen = st.text_area("Digite sua mensagem:", height=100, key="qwen_input")

    if st.button("💬 Enviar", key="qwen_send_btn"):
        if user_input_qwen and 'qwen' in models:
            tokenizer = models['qwen']['tokenizer']
            model = models['qwen']['model']

            try:
                prompt = ""
                for msg in st.session_state.chat_history:
                    prompt += f"Usuário: {msg['user']}\nQwen: {msg['bot']}\n"
                prompt += f"Usuário: {user_input_qwen}\nQwen:"

                inputs = tokenizer(prompt, return_tensors="pt")
                output = model.generate(**inputs, max_new_tokens=150, do_sample=True, temperature=0.8)
                decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

                # Extração da última resposta
                last_response = decoded_output.split("Qwen:")[-1].strip()

                st.session_state.chat_history.append({
                    'user': user_input_qwen,
                    'bot': last_response
                })

            except Exception as e:
                st.error(f"Erro ao gerar resposta: {e}")
        else:
            st.warning("⚠️ Escreve uma mensagem para iniciar a conversa.")

    # Mostrar o histórico
    if st.session_state.chat_history:
        st.markdown("### 🗨️ Histórico da Conversa")
        for msg in reversed(st.session_state.chat_history[-5:]):
            st.markdown(f"**Você:** {msg['user']}")
            st.markdown(f"**Qwen:** {msg['bot']}")
            st.markdown("---")

    st.markdown('</div>', unsafe_allow_html=True)
