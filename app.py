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
        
        .metric-card {
            background: rgba(255,255,255,0.1);
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            margin: 0.5rem 0;
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
        
        .chat-message {
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 10px;
            background: rgba(255,255,255,0.1);
        }
        
        .user-message {
            background: rgba(102, 126, 234, 0.3);
            margin-left: 2rem;
        }
        
        .bot-message {
            background: rgba(118, 75, 162, 0.3);
            margin-right: 2rem;
        }
        
        @media (max-width: 768px) {
            .demo-card {
                padding: 1rem;
            }
        }
    </style>
""", unsafe_allow_html=True)

# Constantes
MAX_TEXT_LENGTH = 1000
MAX_CHAT_HISTORY = 10
DEFAULT_MODELS = {
    "sentiment": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "summarization": "facebook/bart-large-cnn",
    "chat": "Qwen/Qwen1.5-0.5B-Chat"
}

class ModelManager:
    """Classe para gerenciar o carregamento e cache dos modelos"""
    
    def __init__(self):
        self.models = {}
        self.loading_status = {}
    
    @st.cache_resource
    def load_sentiment_model(_self):
        """Carrega modelo de análise de sentimentos"""
        try:
            return pipeline(
                "sentiment-analysis", 
                model=DEFAULT_MODELS["sentiment"],
                return_all_scores=True
            )
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de sentimentos: {e}")
            # Fallback para modelo mais leve
            return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    
    @st.cache_resource
    def load_summarization_model(_self):
        """Carrega modelo de sumarização"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODELS["summarization"])
            model = AutoModelForSeq2SeqLM.from_pretrained(DEFAULT_MODELS["summarization"])
            return {"tokenizer": tokenizer, "model": model}
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de sumarização: {e}")
            # Fallback
            tokenizer = AutoTokenizer.from_pretrained("t5-small")
            model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
            return {"tokenizer": tokenizer, "model": model}
    
    @st.cache_resource
    def load_chat_model(_self):
        """Carrega modelo de chat"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODELS["chat"])
            model = AutoModelForCausalLM.from_pretrained(DEFAULT_MODELS["chat"])
            
            # Configurar pad_token se não existir
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            return {"tokenizer": tokenizer, "model": model}
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de chat: {e}")
            # Fallback para modelo mais leve
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B-Chat", trust_remote_code=True)
            return {"tokenizer": tokenizer, "model": model}

def validate_input(text: str, max_length: int = MAX_TEXT_LENGTH) -> tuple[bool, str]:
    """Valida entrada do usuário"""
    if not text or not text.strip():
        return False, "⚠️ Por favor, insira um texto válido."
    
    if len(text) > max_length:
        return False, f"⚠️ Texto muito longo. Máximo permitido: {max_length} caracteres."
    
    return True, ""

def init_session_state():
    """Inicializa variáveis de sessão"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = ModelManager()

def render_header():
    """Renderiza cabeçalho da aplicação"""
    st.title("🤖 AiiT - Aplicação de IA Multifunções")
    st.markdown("### Plataforma integrada de processamento de linguagem natural")
    
    # Sidebar com informações
    with st.sidebar:
        st.header("ℹ️ Informações")
        st.info("Esta aplicação oferece múltiplas funcionalidades de IA para processamento de texto.")
        
        st.header("🔧 Configurações")
        max_length = st.slider("Comprimento máximo do texto", 100, 2000, MAX_TEXT_LENGTH)
        
        st.header("📊 Estatísticas")
        if hasattr(st.session_state, 'usage_stats'):
            for key, value in st.session_state.usage_stats.items():
                st.metric(key, value)

def sentiment_analysis_tab():
    """Tab de análise de sentimentos"""
    st.markdown('<div class="demo-card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("😊 Análise de Sentimentos")
        st.write("Analise o sentimento de textos em múltiplos idiomas.")
        
        user_input = st.text_area(
            "Digite seu texto:",
            height=150,
            key="sentiment_input",
            help="Máximo de 1000 caracteres"
        )
        
        if st.button("🔍 Analisar Sentimento", key="sentiment_button"):
            is_valid, error_msg = validate_input(user_input)
            
            if not is_valid:
                st.warning(error_msg)
                return
            
            with st.spinner("Analisando sentimento..."):
                try:
                    model = st.session_state.model_manager.load_sentiment_model()
                    result = model(user_input)
                    
                    # Exibir resultados de forma mais visual
                    st.success("✅ Análise concluída!")
                    
                    if isinstance(result[0], list):
                        # Modelo que retorna todos os scores
                        for sentiment in result[0]:
                            label = sentiment['label']
                            score = sentiment['score']
                            
                            # Mapear labels para português
                            label_map = {
                                'NEGATIVE': 'Negativo 😞',
                                'NEUTRAL': 'Neutro 😐',
                                'POSITIVE': 'Positivo 😊',
                                'LABEL_0': 'Negativo 😞',
                                'LABEL_1': 'Neutro 😐',
                                'LABEL_2': 'Positivo 😊'
                            }
                            
                            display_label = label_map.get(label, label)
                            st.progress(score, text=f"{display_label}: {score:.2%}")
                    else:
                        # Modelo que retorna apenas o resultado principal
                        st.json(result)
                        
                except Exception as e:
                    st.error(f" Erro na análise: {str(e)}")
    
    with col2:
        st.markdown("###  Dicas")
        st.markdown("""
        - Textos mais longos geram análises mais precisas
        - Funciona com português, inglês e outros idiomas
        - Ideal para análise de reviews, comentários e feedback
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

def summarization_tab():
    """Tab de sumarização"""
    st.markdown('<div class="demo-card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(" Resumo Inteligente de Texto")
        st.write("Gere resumos concisos de textos longos.")
        
        user_input = st.text_area(
            "Texto para resumir:",
            height=200,
            key="summary_input",
            h
