import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoModelForCausalLM
import torch
import logging
from typing import Dict, Any, Tuple, List
import time
from datetime import datetime

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
        'About': "### AiiT - Plataforma de IA Multifunções\n\nVersão 1.0"
    }
)

# CSS melhorado com responsividade
st.markdown("""
    <style>
        :root {
            --primary-color: #667eea;
            --secondary-color: #764ba2;
        }
        
        .main {
            padding: 1rem;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .demo-card {
            padding: 1.5rem;
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            margin-bottom: 1.5rem;
            color: white;
            transition: all 0.3s ease;
        }
        
        .demo-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 40px rgba(0,0,0,0.15);
        }
        
        .metric-card {
            background: rgba(255,255,255,0.1);
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            margin: 0.5rem 0;
        }
        
        .stButton > button {
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.5rem 2rem;
            font-weight: bold;
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            opacity: 0.9;
        }
        
        .chat-message {
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 10px;
            background: rgba(255,255,255,0.1);
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .user-message {
            background: rgba(102, 126, 234, 0.3);
            margin-left: 2rem;
        }
        
        .bot-message {
            background: rgba(118, 75, 162, 0.3);
            margin-right: 2rem;
        }
        
        .stProgress > div > div > div {
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        }
        
        .example-btn {
            border: 1px solid rgba(255,255,255,0.3);
            border-radius: 20px;
            padding: 0.5rem 1rem;
            margin: 0.2rem;
            font-size: 0.8rem;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        
        .example-btn:hover {
            background: rgba(255,255,255,0.2);
        }
        
        @media (max-width: 768px) {
            .demo-card {
                padding: 1rem;
            }
            
            .col1, .col2 {
                flex: 1 1 100% !important;
            }
            
            .user-message, .bot-message {
                margin-left: 0.5rem !important;
                margin-right: 0.5rem !important;
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

# Exemplos para cada funcionalidade
EXAMPLES = {
    "sentiment": [
        "Estou muito feliz com este produto! Funciona perfeitamente e superou minhas expectativas.",
        "O atendimento foi péssimo, demoraram muito para responder e não resolveram meu problema.",
        "O filme foi ok, nem bom nem ruim, apenas mediano."
    ],
    "summarization": [
        "A inteligência artificial está transformando diversos setores da economia. Desde healthcare até finanças, as aplicações são inúmeras. Empresas que não adotarem essas tecnologias correm o risco de ficar para trás.",
        "O aquecimento global é uma ameaça real para o planeta. Os últimos relatórios científicos mostram que precisamos agir agora para evitar consequências catastróficas nas próximas décadas."
    ],
    "chat": [
        "Me explique como funciona a fotossíntese de forma simples.",
        "Quais são as melhores práticas para escrever código limpo em Python?"
    ]
}

class ModelManager:
    """Classe para gerenciar o carregamento e cache dos modelos
    
    Attributes:
        models (dict): Dicionário para armazenar modelos carregados
        loading_status (dict): Status de carregamento dos modelos
    """
    
    def __init__(self):
        self.models = {}
        self.loading_status = {}
    
    @st.cache_resource(show_spinner="Carregando modelo de análise de sentimentos...")
    def load_sentiment_model(_self):
        """Carrega modelo de análise de sentimentos com fallback
        
        Returns:
            Pipeline: Modelo de análise de sentimentos configurado
        """
        try:
            logger.info("Carregando modelo principal de sentimentos")
            return pipeline(
                "sentiment-analysis", 
                model=DEFAULT_MODELS["sentiment"],
                return_all_scores=True,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de sentimentos: {e}")
            logger.info("Tentando carregar modelo fallback")
            # Fallback para modelo mais leve
            return pipeline(
                "sentiment-analysis", 
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
    
    @st.cache_resource(show_spinner="Carregando modelo de sumarização...")
    def load_summarization_model(_self):
        """Carrega modelo de sumarização com fallback
        
        Returns:
            dict: Dicionário com tokenizer e modelo
        """
        try:
            logger.info("Carregando modelo principal de sumarização")
            tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODELS["summarization"])
            model = AutoModelForSeq2SeqLM.from_pretrained(
                DEFAULT_MODELS["summarization"],
                device_map="auto"
            )
            return {"tokenizer": tokenizer, "model": model}
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de sumarização: {e}")
            logger.info("Tentando carregar modelo fallback")
            # Fallback
            tokenizer = AutoTokenizer.from_pretrained("t5-small")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                "t5-small",
                device_map="auto"
            )
            return {"tokenizer": tokenizer, "model": model}
    
    @st.cache_resource(show_spinner="Carregando modelo de conversação...")
    def load_chat_model(_self):
        """Carrega modelo de chat com fallback robusto
        
        Returns:
            dict: Dicionário com tokenizer e modelo
        """
        try:
            logger.info("Carregando modelo principal de chat")
            tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODELS["chat"])
            model = AutoModelForCausalLM.from_pretrained(
                DEFAULT_MODELS["chat"],
                device_map="auto"
            )
            
            # Configurar pad_token se não existir
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            return {"tokenizer": tokenizer, "model": model}
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de chat: {e}")
            logger.info("Tentando carregar modelo fallback")
            # Fallback para modelo mais leve e estável
            tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
            model = AutoModelForCausalLM.from_pretrained(
                "microsoft/DialoGPT-small",
                device_map="auto"
            )
            return {"tokenizer": tokenizer, "model": model}

def validate_input(text: str, max_length: int = MAX_TEXT_LENGTH) -> Tuple[bool, str]:
    """Valida entrada do usuário conforme critérios estabelecidos
    
    Args:
        text (str): Texto a ser validado
        max_length (int): Comprimento máximo permitido
        
    Returns:
        Tuple[bool, str]: (True, "") se válido, (False, mensagem_erro) se inválido
    """
    if not text or not text.strip():
        return False, "⚠️ Por favor, insira um texto válido."
    
    if len(text) > max_length:
        return False, f"⚠️ Texto muito longo. Máximo permitido: {max_length} caracteres."
    
    return True, ""

def init_session_state():
    """Inicializa variáveis de sessão necessárias"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = ModelManager()
    if 'usage_stats' not in st.session_state:
        st.session_state.usage_stats = {
            "Análises de Sentimento": 0,
            "Textos Resumidos": 0,
            "Mensagens Troca": 0,
            "Último Uso": "Nunca"
        }
    if 'last_activity' not in st.session_state:
        st.session_state.last_activity = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def update_usage_stats(metric: str):
    """Atualiza estatísticas de uso
    
    Args:
        metric (str): Métrica a ser incrementada
    """
    if metric in st.session_state.usage_stats:
        st.session_state.usage_stats[metric] += 1
    st.session_state.last_activity = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.usage_stats["Último Uso"] = st.session_state.last_activity

def render_header():
    """Renderiza cabeçalho da aplicação e sidebar"""
    st.title("🤖 AiiT - Aplicação de IA Multifunções")
    st.markdown("### Plataforma integrada de processamento de linguagem natural")
    
    # Sidebar com informações
    with st.sidebar:
        st.header("ℹ️ Informações")
        st.info("Esta aplicação oferece múltiplas funcionalidades de IA para processamento de texto.")
        
        st.header("🔧 Configurações")
        max_length = st.slider("Comprimento máximo do texto", 100, 2000, MAX_TEXT_LENGTH, help="Ajuste o tamanho máximo de texto permitido")
        
        st.header("📊 Estatísticas")
        for key, value in st.session_state.usage_stats.items():
            st.metric(key, value)
        
        # Mostrar informações do sistema
        st.header("⚙️ Sistema")
        st.caption(f"Streamlit v{st.__version__}")
        st.caption(f"PyTorch v{torch.__version__}")
        st.caption(f"Dispositivo: {'GPU ✅' if torch.cuda.is_available() else 'CPU ⚠️'}")

def render_example_buttons(feature: str):
    """Renderiza botões de exemplo para uma funcionalidade
    
    Args:
        feature (str): Funcionalidade ('sentiment', 'summarization' ou 'chat')
    """
    st.markdown("<div style='margin-bottom: 1rem;'>Exemplos rápidos:</div>", unsafe_allow_html=True)
    cols = st.columns(len(EXAMPLES[feature]))
    for i, example in enumerate(EXAMPLES[feature]):
        with cols[i]:
            if st.button(example[:30] + "..." if len(example) > 30 else example, 
                        key=f"example_{feature}_{i}",
                        help="Clique para carregar este exemplo"):
                if feature == "sentiment":
                    st.session_state.sentiment_input = example
                elif feature == "summarization":
                    st.session_state.summary_input = example
                elif feature == "chat":
                    st.session_state.chat_input = example
                st.rerun()

def sentiment_analysis_tab():
    """Tab de análise de sentimentos"""
    st.markdown('<div class="demo-card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("😊 Análise de Sentimentos")
        st.write("Analise o sentimento de textos em múltiplos idiomas.")
        
        render_example_buttons("sentiment")
        
        user_input = st.text_area(
            "Digite seu texto:",
            height=150,
            key="sentiment_input",
            help="Máximo de 1000 caracteres",
            placeholder="Digite aqui o texto que deseja analisar o sentimento..."
        )
        
        analyze_btn = st.button("🔍 Analisar Sentimento", key="sentiment_button")
        
        if analyze_btn:
            is_valid, error_msg = validate_input(user_input)
            
            if not is_valid:
                st.warning(error_msg)
                return
            
            with st.spinner("Analisando sentimento..."):
                try:
                    start_time = time.time()
                    model = st.session_state.model_manager.load_sentiment_model()
                    result = model(user_input)
                    processing_time = time.time() - start_time
                    
                    # Atualizar estatísticas
                    update_usage_stats("Análises de Sentimento")
                    
                    # Exibir resultados
                    st.success(f"✅ Análise concluída em {processing_time:.2f} segundos!")
                    
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
                        
                    # Explicação dos resultados
                    with st.expander("ℹ️ Como interpretar estes resultados?"):
                        st.markdown("""
                        - **Positivo (> 60%)**: Sentimento claramente positivo
                        - **Neutro (40-60%)**: Neutro ou misto
                        - **Negativo (> 60%)**: Sentimento claramente negativo
                        """)
                        
                except Exception as e:
                    logger.error(f"Erro na análise de sentimentos: {str(e)}")
                    st.error(f"❌ Ocorreu um erro durante a análise. Por favor, tente novamente com um texto diferente.")
    
    with col2:
        st.markdown("### 📝 Dicas")
        st.markdown("""
        - Textos mais longos (50+ palavras) geram análises mais precisas
        - Funciona com português, inglês, espanhol e outros idiomas
        - Ideal para análise de:
          - Reviews de produtos
          - Comentários em redes sociais
          - Feedback de clientes
        """)
        
        st.markdown("### 📊 Última Análise")
        if 'sentiment_input' in st.session_state and st.session_state.sentiment_input:
            st.caption(f"Texto analisado ({len(st.session_state.sentiment_input)} caracteres):")
            st.text(st.session_state.sentiment_input[:200] + ("..." if len(st.session_state.sentiment_input) > 200 else ""))
    
    st.markdown('</div>', unsafe_allow_html=True)

def summarization_tab():
    """Tab de sumarização de texto"""
    st.markdown('<div class="demo-card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Resumo Inteligente de Texto")
        st.write("Gere resumos concisos de textos longos preservando as informações mais importantes.")
        
        render_example_buttons("summarization")
        
        user_input = st.text_area(
            "Texto para resumir:",
            height=200,
            key="summary_input",
            help="Máximo de 1000 caracteres",
            placeholder="Cole aqui o texto longo que deseja resumir..."
        )
        
        # Controles de sumarização
        col1a, col1b = st.columns(2)
        with col1a:
            max_length = st.slider("Tamanho máximo do resumo", 50, 300, 150)
        with col1b:
            min_length = st.slider("Tamanho mínimo do resumo", 10, 100, 50)
        
        summarize_btn = st.button("✂️ Gerar Resumo", key="summarize_button")
        
        if summarize_btn:
            is_valid, error_msg = validate_input(user_input)
            
            if not is_valid:
                st.warning(error_msg)
                return
            
            with st.spinner("Processando resumo..."):
                try:
                    start_time = time.time()
                    models = st.session_state.model_manager.load_summarization_model()
                    tokenizer = models["tokenizer"]
                    model = models["model"]
                    
                    # Tokenizar entrada
                    inputs = tokenizer.encode(
                        "summarize: " + user_input,
                        return_tensors="pt",
                        max_length=512,
                        truncation=True
                    ).to(model.device)
                    
                    # Gerar resumo
                    summary_ids = model.generate(
                        inputs,
                        max_length=max_length,
                        min_length=min_length,
                        length_penalty=2.0,
                        num_beams=4,
                        early_stopping=True
                    )
                    
                    # Decodificar resultado
                    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                    processing_time = time.time() - start_time
                    
                    # Atualizar estatísticas
                    update_usage_stats("Textos Resumidos")
                    
                    # Exibir resultados
                    st.success(f"✅ Resumo gerado em {processing_time:.2f} segundos!")
                    st.markdown(f'<div class="chat-message bot-message">{summary}</div>', unsafe_allow_html=True)
                    
                    # Mostrar taxa de compressão
                    orig_len = len(user_input.split())
                    summ_len = len(summary.split())
                    ratio = summ_len / orig_len if orig_len > 0 else 0
                    st.metric("Taxa de compressão", f"{ratio:.0%}", 
                             help=f"De {orig_len} para {summ_len} palavras")
                    
                except Exception as e:
                    logger.error(f"Erro na sumarização: {str(e)}")
                    st.error("❌ Ocorreu um erro ao gerar o resumo. Por favor, tente com um texto diferente.")
    
    with col2:
        st.markdown("### 💡 Como funciona?")
        st.markdown("""
        O modelo:
        1. Analisa a estrutura do texto
        2. Identifica os conceitos principais
        3. Gera um novo texto coeso e conciso
        
        **Dicas:**
        - Textos bem estruturados geram melhores resumos
        - Funciona melhor com 2+ parágrafos
        - Ajuste os controles para resumos mais longos/curtos
        """)
        
        st.markdown("### 📈 Estatísticas")
        if 'summary_input' in st.session_state and st.session_state.summary_input:
            st.metric("Texto original", f"{len(st.session_state.summary_input.split())} palavras")
    
    st.markdown('</div>', unsafe_allow_html=True)

def chat_tab():
    """Tab de conversação com IA"""
    st.markdown('<div class="demo-card">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat com IA")
        st.write("Converse com um assistente de IA inteligente sobre diversos tópicos.")
        
        render_example_buttons("chat")
        
        # Exibir histórico de chat
        for message in st.session_state.chat_history[-MAX_CHAT_HISTORY:]:
            role = message["role"]
            content = message["content"]
            
            if role == "user":
                st.markdown(f'<div class="chat-message user-message"><b>Você:</b> {content}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message bot-message"><b>IA:</b> {content}</div>', 
                           unsafe_allow_html=True)
        
        # Entrada do usuário
        user_input = st.text_input(
            "Digite sua mensagem:",
            key="chat_input",
            placeholder="Escreva sua mensagem aqui...",
            label_visibility="collapsed"
        )
        
        chat_cols = st.columns([3, 1])
        with chat_cols[0]:
            send_btn = st.button("📤 Enviar", key="chat_button", use_container_width=True)
        with chat_cols[1]:
            clear_btn = st.button("🧹 Limpar", key="clear_chat", use_container_width=True)
        
        if clear_btn:
            st.session_state.chat_history = []
            st.rerun()
            
        if send_btn and user_input:
            is_valid, error_msg = validate_input(user_input)
            
            if not is_valid:
                st.warning(error_msg)
                return
            
            # Adicionar mensagem do usuário ao histórico
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            with st.spinner("IA está pensando..."):
                try:
                    start_time = time.time()
                    models = st.session_state.model_manager.load_chat_model()
                    tokenizer = models["tokenizer"]
                    model = models["model"]
                    
                    # Preparar input com histórico de conversa
                    chat_history = "\n".join(
                        f"{msg['role']}: {msg['content']}" 
                        for msg in st.session_state.chat_history
                    )
                    input_text = f"{chat_history}\nassistant:"
                    
                    # Codificar entrada
                    inputs = tokenizer.encode(
                        input_text,
                        return_tensors="pt",
                        max_length=1024,
                        truncation=True
                    ).to(model.device)
                    
                    # Gerar resposta
                    output = model.generate(
                        inputs,
                        max_length=1024,
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=True,
                        top_k=50,
                        top_p=0.95,
                        temperature=0.7
                    )
                    
                    # Decodificar resposta
                    response = tokenizer.decode(output[0], skip_special_tokens=True)
                    
                    # Extrair apenas a última resposta do assistente
                    assistant_response = response.split("assistant:")[-1].strip()
                    processing_time = time.time() - start_time
                    
                    # Adicionar resposta ao histórico
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": assistant_response
                    })
                    
                    # Atualizar estatísticas
                    update_usage_stats("Mensagens Troca")
                    
                    # Rerun para mostrar a nova mensagem
                    st.rerun()
                    
                except Exception as e:
                    logger.error(f"Erro no chat: {str(e)}")
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": "❌ Desculpe, ocorreu um erro. Por favor, tente novamente."
                    })
                    st.rerun()
    
    with col2:
        st.markdown("### 🎯 Dicas para Melhores Respostas")
        st.markdown("""
        - Seja claro e específico
        - Forneça contexto quando necessário
        - Reformule se a resposta não for satisfatória
        
        **Limitações:**
        - Pode gerar informações incorretas
        - Memória limitada a 10 mensagens
        - Não substitui conselho profissional
        """)
        
        st.markdown("### 📆 Histórico Recente")
        if st.session_state.chat_history:
            for msg in st.session_state.chat_history[-3:]:
                st.caption(f"**{msg['role'].title()}:** {msg['content'][:60]}...")
        else:
            st.caption("Nenhuma conversa ainda...")
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Função principal da aplicação"""
    # Inicializar estado da sessão
    init_session_state()
    
    # Renderizar cabeçalho
    render_header()
    
    # Criar abas
    tab1, tab2, tab3 = st.tabs([
        "😊 Análise de Sentimentos", 
        "📝 Sumarização de Texto", 
        "💬 Chat com IA"
    ])
    
    with tab1:
        sentiment_analysis_tab()
    
    with tab2:
        summarization_tab()
    
    with tab3:
        chat_tab()
    
    # Rodapé
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <small>🚀 AiiT - IA Multifunções | Versão 1.0 | Desenvolvido com Streamlit e Hugging Face</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
