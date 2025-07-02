import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch
import logging
from typing import Dict, Any, Tuple, List
import time
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
from bertopic import BERTopic
import numpy as np

# Solução para os warnings iniciais
try:
    from accelerate import Accelerator
    accelerator = Accelerator()
except ImportError:
    st.warning("Para melhor performance, instale o accelerate: `pip install accelerate`")

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuração da página
st.set_page_config(
    page_title="AiiT - IA Multifunções", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/seuuser/seurepo',
        'Report a bug': "https://github.com/seuuser/seurepo/issues",
        'About': "### AiiT - Plataforma de IA Multifunções\n\nVersão 2.0"
    }
)

# CSS melhorado
st.markdown("""
    <style>
        :root {
            --primary-color: #667eea;
            --secondary-color: #764ba2;
            --accent-color: #4facfe;
        }
        
        .main {
            padding: 1rem;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .feature-card {
            padding: 1.5rem;
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
            color: white;
            transition: all 0.3s ease;
        }
        
        .feature-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 40px rgba(0,0,0,0.15);
        }
        
        .stButton > button {
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.5rem 1.5rem;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .result-box {
            animation: fadeIn 0.5s ease;
            padding: 1.5rem;
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            margin: 1rem 0;
        }
        
        @media (max-width: 768px) {
            .feature-card {
                padding: 1rem;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Constantes
MAX_TEXT_LENGTH = 2000
DEFAULT_MODELS = {
    "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "summarization": "facebook/bart-large-cnn",
    "translation": "Helsinki-NLP/opus-mt-tc-big-pt-en",
    "ner": "dslim/bert-base-NER",
    "qa": "pierreguillou/bert-base-cased-squad-v1.1-portuguese",
    "generation": "gpt2",
    "embeddings": "paraphrase-multilingual-MiniLM-L12-v2"
}

class ModelManager:
    """Gerenciador otimizado de modelos com tratamento de erros"""
    
    def __init__(self):
        self.models = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    @st.cache_resource(show_spinner="Carregando modelo de sentimentos...")
    def load_sentiment_model(_self):
        try:
            logger.info("Carregando modelo de sentimentos")
            model = pipeline(
                "text-classification",
                model=DEFAULT_MODELS["sentiment"],
                device=_self.device,
                top_k=None  # Substitui return_all_scores
            )
            return model
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            return pipeline("sentiment-analysis", device=_self.device)

    @st.cache_resource(show_spinner="Carregando modelo de sumarização...")
    def load_summarization_model(_self):
        try:
            logger.info("Carregando modelo de sumarização")
            tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODELS["summarization"])
            model = AutoModelForSeq2SeqLM.from_pretrained(
                DEFAULT_MODELS["summarization"],
                device_map="auto" if torch.cuda.is_available() else None
            )
            return {"tokenizer": tokenizer, "model": model}
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            tokenizer = AutoTokenizer.from_pretrained("t5-small")
            model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
            return {"tokenizer": tokenizer, "model": model}

    @st.cache_resource(show_spinner="Carregando modelo de tradução...")
    def load_translation_model(_self):
        try:
            return pipeline(
                "translation",
                model=DEFAULT_MODELS["translation"],
                device=_self.device
            )
        except Exception as e:
            logger.error(f"Erro ao carregar tradutor: {e}")
            return pipeline("translation_en_to_pt", device=_self.device)

    @st.cache_resource(show_spinner="Carregando modelo de NER...")
    def load_ner_model(_self):
        try:
            return pipeline(
                "ner",
                model=DEFAULT_MODELS["ner"],
                device=_self.device,
                aggregation_strategy="simple"
            )
        except Exception as e:
            logger.error(f"Erro ao carregar NER: {e}")
            return pipeline("ner", device=_self.device)

    @st.cache_resource(show_spinner="Carregando modelo Q&A...")
    def load_qa_model(_self):
        try:
            return pipeline(
                "question-answering",
                model=DEFAULT_MODELS["qa"],
                device=_self.device
            )
        except Exception as e:
            logger.error(f"Erro ao carregar Q&A: {e}")
            return None

    @st.cache_resource(show_spinner="Carregando modelo de geração...")
    def load_generation_model(_self):
        try:
            return pipeline(
                "text-generation",
                model=DEFAULT_MODELS["generation"],
                device=_self.device
            )
        except Exception as e:
            logger.error(f"Erro ao carregar gerador: {e}")
            return None

    @st.cache_resource(show_spinner="Carregando modelo de embeddings...")
    def load_embeddings_model(_self):
        try:
            return SentenceTransformer(DEFAULT_MODELS["embeddings"], device=_self.device)
        except Exception as e:
            logger.error(f"Erro ao carregar embeddings: {e}")
            return None

def init_session_state():
    """Inicializa o estado da sessão"""
    defaults = {
        'chat_history': [],
        'usage_stats': {
            "Análises": 0,
            "Traduções": 0,
            "Resumos": 0,
            "Consultas": 0
        },
        'last_activity': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'active_features': []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def update_usage(feature: str):
    """Atualiza estatísticas de uso"""
    if feature in st.session_state.usage_stats:
        st.session_state.usage_stats[feature] += 1
    st.session_state.last_activity = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Funções das novas funcionalidades
def translation_feature():
    with st.expander("🌍 Tradutor Português-Inglês", expanded=True):
        text = st.text_area("Texto para traduzir", height=150)
        if st.button("Traduzir"):
            if not text.strip():
                st.warning("Por favor, insira um texto")
                return
            
            with st.spinner("Traduzindo..."):
                try:
                    translator = st.session_state.model_manager.load_translation_model()
                    result = translator(text)
                    update_usage("Traduções")
                    
                    st.markdown(f"""
                    <div class="result-box">
                        <h4>Texto Original</h4>
                        <p>{text}</p>
                        <h4>Tradução</h4>
                        <p>{result[0]['translation_text']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Erro na tradução: {str(e)}")

def ner_feature():
    with st.expander("🔍 Reconhecimento de Entidades", expanded=True):
        text = st.text_area("Texto para análise", height=150)
        if st.button("Identificar Entidades"):
            if not text.strip():
                st.warning("Por favor, insira um texto")
                return
            
            with st.spinner("Analisando..."):
                try:
                    ner = st.session_state.model_manager.load_ner_model()
                    results = ner(text)
                    update_usage("Análises")
                    
                    entities = []
                    for entity in results:
                        entities.append({
                            "Entidade": entity['word'],
                            "Tipo": entity['entity_group'],
                            "Confiança": f"{entity['score']:.2%}"
                        })
                    
                    st.markdown("""
                    <div class="result-box">
                        <h4>Entidades Identificadas</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.dataframe(entities)
                except Exception as e:
                    st.error(f"Erro na análise: {str(e)}")

def qa_feature():
    with st.expander("❓ Sistema de Perguntas e Respostas", expanded=True):
        context = st.text_area("Contexto", height=100)
        question = st.text_input("Pergunta")
        
        if st.button("Responder") and context and question:
            with st.spinner("Buscando resposta..."):
                try:
                    qa = st.session_state.model_manager.load_qa_model()
                    if qa:
                        result = qa(question=question, context=context)
                        update_usage("Consultas")
                        
                        st.markdown(f"""
                        <div class="result-box">
                            <h4>Resposta</h4>
                            <p>{result['answer']}</p>
                            <p>Confiança: {result['score']:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("Modelo Q&A não disponível")
                except Exception as e:
                    st.error(f"Erro ao responder: {str(e)}")

def text_generation_feature():
    with st.expander("✨ Geração de Texto Criativo", expanded=True):
        prompt = st.text_area("Prompt de geração", height=100)
        length = st.slider("Comprimento", 50, 500, 100)
        
        if st.button("Gerar Texto") and prompt:
            with st.spinner("Criando conteúdo..."):
                try:
                    generator = st.session_state.model_manager.load_generation_model()
                    if generator:
                        result = generator(
                            prompt,
                            max_length=length,
                            do_sample=True,
                            top_p=0.95,
                            temperature=0.7
                        )
                        update_usage("Consultas")
                        
                        st.markdown(f"""
                        <div class="result-box">
                            <h4>Texto Gerado</h4>
                            <p>{result[0]['generated_text']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("Modelo de geração não disponível")
                except Exception as e:
                    st.error(f"Erro na geração: {str(e)}")

def similarity_feature():
    with st.expander("📊 Comparação de Textos", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            text1 = st.text_area("Texto 1", height=100)
        with col2:
            text2 = st.text_area("Texto 2", height=100)
        
        if st.button("Calcular Similaridade") and text1 and text2:
            with st.spinner("Calculando..."):
                try:
                    model = st.session_state.model_manager.load_embeddings_model()
                    if model:
                        embeddings = model.encode([text1, text2])
                        similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
                        update_usage("Análises")
                        
                        st.markdown(f"""
                        <div class="result-box">
                            <h4>Similaridade</h4>
                            <p>{similarity:.2%} de similaridade</p>
                            <progress value="{similarity}" max="1"></progress>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.warning("Modelo de embeddings não disponível")
                except Exception as e:
                    st.error(f"Erro no cálculo: {str(e)}")

def main():
    # Inicialização
    init_session_state()
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = ModelManager()
    
    # Interface
    st.title("🤖 AiiT - Plataforma Avançada de NLP")
    st.markdown("### Todas as ferramentas de processamento de linguagem em um só lugar")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configurações")
        st.selectbox("Tema", ["Claro", "Escuro"], key="theme")
        st.header("📊 Estatísticas")
        for k, v in st.session_state.usage_stats.items():
            st.metric(k, v)
        st.caption(f"Última atividade: {st.session_state.last_activity}")
    
    # Abas principais
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Texto", "🌍 Idiomas", "🔍 Análise", "✨ Criativo"])
    
    with tab1:
        st.header("Ferramentas de Texto")
        col1, col2 = st.columns(2)
        with col1:
            summarization_feature()
        with col2:
            text_generation_feature()
    
    with tab2:
        st.header("Ferramentas de Idioma")
        translation_feature()
    
    with tab3:
        st.header("Ferramentas Analíticas")
        col1, col2 = st.columns(2)
        with col1:
            sentiment_feature()
        with col2:
            ner_feature()
        qa_feature()
        similarity_feature()
    
    with tab4:
        st.header("Ferramentas Criativas")
        col1, col2 = st.columns(2)
        with col1:
            poem_generation_feature()
        with col2:
            story_generation_feature()

if __name__ == "__main__":
    main()
