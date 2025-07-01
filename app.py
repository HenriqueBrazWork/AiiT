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
            help="Insira textos longos para obter resumos eficazes"
        )
        
        # Opções avançadas
        with st.expander(" Configurações Avançadas"):
            max_length = st.slider("Comprimento máximo do resumo", 50, 300, 150)
            min_length = st.slider("Comprimento mínimo do resumo", 20, 100, 30)
        
        if st.button(" Gerar Resumo", key="summary_button"):
            is_valid, error_msg = validate_input(user_input, 2000)
            
            if not is_valid:
                st.warning(error_msg)
                return
            
            with st.spinner("Gerando resumo..."):
                try:
                    model_data = st.session_state.model_manager.load_summarization_model()
                    tokenizer = model_data["tokenizer"]
                    model = model_data["model"]
                    
                    # Preparar input com contexto adequado
                    if "t5" in tokenizer.name_or_path.lower():
                        input_text = "summarize: " + user_input
                    else:
                        input_text = user_input
                    
                    inputs = tokenizer(
                        input_text, 
                        return_tensors="pt", 
                        max_length=1024, 
                        truncation=True
                    )
                    
                    summary_ids = model.generate(
                        inputs["input_ids"],
                        max_length=max_length,
                        min_length=min_length,
                        length_penalty=2.0,
                        num_beams=4,
                        early_stopping=True,
                        no_repeat_ngram_size=2
                    )
                    
                    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                    
                    st.success(" Resumo gerado com sucesso!")
                    st.markdown("###  Resumo:")
                    st.info(summary)
                    
                    # Estatísticas
                    original_words = len(user_input.split())
                    summary_words = len(summary.split())
                    compression_ratio = (1 - summary_words/original_words) * 100
                    
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.metric("Palavras Originais", original_words)
                    with col_stat2:
                        st.metric("Palavras do Resumo", summary_words)
                    with col_stat3:
                        st.metric("Taxa de Compressão", f"{compression_ratio:.1f}%")
                    
                except Exception as e:
                    st.error(f" Erro na sumarização: {str(e)}")
    
    with col2:
        st.markdown("###  Dicas")
        st.markdown("""
        - Textos com 300+ palavras geram melhores resumos
        - Ideal para artigos, relatórios e documentos
        - Ajuste o comprimento conforme necessário
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

def chat_tab():
    """Tab de chat com IA - Versão Melhorada"""
    st.markdown('<div class="demo-card">', unsafe_allow_html=True)
    
    # Inicialização do estado da sessão
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Como posso te ajudar hoje?"}]

    # Exibe o histórico de mensagens
    for msg in st.session_state.messages:
        avatar = "🤖" if msg["role"] == "assistant" else "👤"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Input do usuário com container especial
    if prompt := st.chat_input("Digite sua mensagem..."):
        # Adiciona mensagem do usuário
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Exibe imediatamente a mensagem do usuário
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)
        
        # Resposta do assistente
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Pensando..."):
                try:
                    # Carrega o modelo (cacheado)
                    model_data = st.session_state.model_manager.load_chat_model()
                    tokenizer = model_data["tokenizer"]
                    model = model_data["model"]
                    
                    # Prepara as mensagens no formato correto
                    messages_for_model = [
                        {"role": "system", "content": "Você é um assistente IA útil."},
                        *[{"role": m["role"], "content": m["content"]} 
                          for m in st.session_state.messages[-4:]]  # Mantém contexto recente
                    
                    # Aplica o template de chat
                    inputs = tokenizer.apply_chat_template(
                        messages_for_model,
                        add_generation_prompt=True,
                        return_tensors="pt"
                    ).to(model.device)
                    
                    # Gera a resposta
                    outputs = model.generate(
                        inputs,
                        max_new_tokens=500,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9
                    )
                    
                    # Decodifica a resposta (removendo o prompt)
                    response = tokenizer.decode(
                        outputs[0][len(inputs[0]):], 
                        skip_special_tokens=True
                    )
                    
                    # Exibe a resposta gradualmente (efeito de digitação)
                    message_placeholder = st.empty()
                    full_response = ""
                    for chunk in response.split():
                        full_response += chunk + " "
                        time.sleep(0.05)  # Efeito de digitação
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                    
                    # Adiciona ao histórico
                    st.session_state.messages.append(
                        {"role": "assistant", "content": full_response}
                    )
                    
                except Exception as e:
                    st.error(f"Erro ao gerar resposta: {str(e)}")
                    logger.error(f"Erro no chat: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Função principal"""
    init_session_state()
    render_header()
    
    # Tabs principais
    tab1, tab2, tab3 = st.tabs([
        " Análise de Sentimentos", 
        " Resumo de Texto", 
        " Chat IA"
    ])
    
    with tab1:
        sentiment_analysis_tab()
    
    with tab2:
        summarization_tab()
    
    with tab3:
        chat_tab()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "AiiT - Powered by Transformers & Streamlit | "
        "Desenvolvida demonstração"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
