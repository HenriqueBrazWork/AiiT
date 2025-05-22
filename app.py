import streamlit as st
import logging
from transformers import pipeline
import torch
from PIL import Image
import io
from typing import Dict, Any, Optional
from datetime import datetime
import base64

# Configuração da página
st.set_page_config(
    page_title="AiiT - Demo de IA",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para branding
def load_custom_css():
    st.markdown("""
    <style>
        .main-header {
            background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
            padding: 2rem;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .tech-card {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 10px;
            border-left: 4px solid #2a5298;
            margin: 1rem 0;
        }
        
        .demo-section {
            background: white;
            border-radius: 10px;
            padding: 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        
        .footer {
            text-align: center;
            padding: 2rem;
            background: #f1f3f4;
            border-radius: 10px;
            margin-top: 3rem;
        }
        
        .metric-card {
            background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache para modelos - otimizado para demonstração
@st.cache_resource
def create_ai_models() -> Dict[str, Any]:
    """Carrega modelos de IA otimizados para demonstração profissional"""
    try:
        device = 0 if torch.cuda.is_available() else -1
        
        models = {
            'sentiment_analysis': pipeline(
                "sentiment-analysis", 
                model="cardiffnlp/twitter-roberta-base-sentiment-latest", 
                device=device
            ),
            'text_summarization': pipeline(
                "summarization", 
                model="facebook/bart-large-cnn", 
                device=device
            ),
            'language_detection': pipeline(
                "text-classification",
                model="papluca/xlm-roberta-base-language-detection",
                device=device
            ),
            'content_moderation': pipeline(
                "text-classification",
                model="unitary/toxic-bert",
                device=device
            ),
            'object_detection': pipeline(
                "object-detection", 
                model="facebook/detr-resnet-50", 
                device=device
            ),
            'business_qa': pipeline(
                "question-answering", 
                model="deepset/roberta-base-squad2", 
                device=device
            )
        }
        
        logger.info("✅ Modelos AiiT carregados com sucesso")
        return models
        
    except Exception as e:
        logger.error(f"❌ Erro ao carregar modelos: {e}")
        st.error(f"Erro na inicialização dos modelos de IA: {e}")
        return {}

def process_ai_request(model_key: str, models: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Processa requisições de IA com telemetria profissional"""
    start_time = datetime.now()
    
    try:
        if model_key not in models:
            return {"error": "Modelo não disponível", "processing_time": 0}
        
        model = models[model_key]
        result = {}
        
        if model_key == 'sentiment_analysis':
            text = kwargs.get('text', '')
            if len(text.strip()) < 5:
                return {"error": "Texto muito curto para análise", "processing_time": 0}
            
            analysis = model(text)[0]
            confidence = analysis['score']
            
            # Mapeamento mais profissional
            sentiment_map = {
                'POSITIVE': {'label': 'Positivo', 'emoji': '😊', 'color': 'green'},
                'NEGATIVE': {'label': 'Negativo', 'emoji': '😞', 'color': 'red'},
                'NEUTRAL': {'label': 'Neutro', 'emoji': '😐', 'color': 'gray'}
            }
            
            sentiment_info = sentiment_map.get(analysis['label'].upper(), 
                                             {'label': analysis['label'], 'emoji': '🤔', 'color': 'blue'})
            
            result = {
                'sentiment': sentiment_info['label'],
                'confidence': confidence,
                'emoji': sentiment_info['emoji'],
                'color': sentiment_info['color'],
                'recommendation': get_business_recommendation(sentiment_info['label'], confidence)
            }
            
        elif model_key == 'text_summarization':
            text = kwargs.get('text', '')
            if len(text.split()) < 20:
                return {"error": "Texto precisa ter pelo menos 20 palavras", "processing_time": 0}
            
            summary = model(text, max_length=130, min_length=30, do_sample=False)[0]
            
            result = {
                'original_length': len(text.split()),
                'summary_length': len(summary['summary_text'].split()),
                'summary': summary['summary_text'],
                'compression_ratio': round(len(summary['summary_text']) / len(text) * 100, 1)
            }
            
        elif model_key == 'language_detection':
            text = kwargs.get('text', '')
            if len(text.strip()) < 3:
                return {"error": "Texto muito curto para detecção", "processing_time": 0}
            
            detection = model(text)[0]
            
            # Mapeamento de códigos de idioma
            language_map = {
                'pt': 'Português', 'en': 'Inglês', 'es': 'Espanhol', 
                'fr': 'Francês', 'de': 'Alemão', 'it': 'Italiano'
            }
            
            result = {
                'language': language_map.get(detection['label'], detection['label']),
                'confidence': detection['score'],
                'language_code': detection['label']
            }
            
        elif model_key == 'content_moderation':
            text = kwargs.get('text', '')
            moderation = model(text)[0]
            
            is_toxic = moderation['label'] == 'TOXIC'
            
            result = {
                'is_safe': not is_toxic,
                'toxicity_score': moderation['score'] if is_toxic else 1 - moderation['score'],
                'classification': 'Conteúdo Inadequado' if is_toxic else 'Conteúdo Seguro',
                'recommendation': 'Revisar conteúdo' if is_toxic else 'Aprovado para publicação'
            }
            
        elif model_key == 'object_detection':
            image_file = kwargs.get('image')
            if not image_file:
                return {"error": "Imagem não fornecida", "processing_time": 0}
                
            image = Image.open(image_file).convert("RGB")
            detections = model(image)
            
            # Processa detecções para apresentação profissional
            objects = []
            for detection in detections:
                objects.append({
                    'object': detection['label'],
                    'confidence': round(detection['score'] * 100, 1),
                    'coordinates': detection['box']
                })
            
            result = {
                'total_objects': len(objects),
                'objects': objects,
                'image_size': image.size
            }
            
        elif model_key == 'business_qa':
            question = kwargs.get('question', '')
            context = kwargs.get('context', '')
            
            if not question or not context:
                return {"error": "Pergunta e contexto são obrigatórios", "processing_time": 0}
            
            answer = model(question=question, context=context)
            
            result = {
                'answer': answer['answer'],
                'confidence': round(answer['score'] * 100, 1),
                'context_length': len(context.split()),
                'answer_position': f"Posição {answer.get('start', 0)}-{answer.get('end', 0)}"
            }
        
        # Calcula tempo de processamento
        processing_time = (datetime.now() - start_time).total_seconds()
        result['processing_time'] = round(processing_time, 3)
        result['status'] = 'success'
        
        return result
        
    except Exception as e:
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Erro no processamento {model_key}: {e}")
        return {
            "error": f"Erro no processamento: {str(e)}", 
            "processing_time": round(processing_time, 3),
            "status": "error"
        }

def get_business_recommendation(sentiment: str, confidence: float) -> str:
    """Gera recomendações de negócio baseadas na análise"""
    if confidence > 0.8:
        if sentiment == 'Positivo':
            return "💡 Ótimo feedback! Use em testimonials e marketing."
        elif sentiment == 'Negativo':
            return "⚠️ Atenção necessária. Considere resposta proativa."
        else:
            return "📊 Feedback neutro. Oportunidade de engajamento."
    else:
        return "🔍 Análise inconclusiva. Recomenda-se revisão manual."

def display_header():
    """Exibe cabeçalho profissional da AiiT"""
    st.markdown("""
    <div class="main-header">
        <h1>🚀 AiiT - Soluções Tecnológicas Inovadoras</h1>
        <h3>Demonstração Interativa de Inteligência Artificial</h3>
        <p>Explore o poder da IA aplicada aos seus negócios</p>
    </div>
    """, unsafe_allow_html=True)

def display_company_info():
    """Exibe informações da empresa na sidebar"""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🏢 Sobre a AiiT")
    st.sidebar.info("""
    **AiiT** oferece soluções de IA personalizadas para empresas que buscam inovação tecnológica.
    
    **Nossos Serviços:**
    • Análise de Sentimentos
    • Processamento de Linguagem Natural
    • Visão Computacional
    • Automação Inteligente
    • Consultoria em IA
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📞 Contato")
    st.sidebar.markdown("""
    📧 contato@aiit.com.br  
    📱 +55 (11) 99999-9999  
    🌐 www.aiit.com.br
    """)

def main():
    """Aplicação principal - Demo AiiT"""
    load_custom_css()
    display_header()
    
    # Sidebar com informações da empresa
    with st.sidebar:
        st.markdown("### ⚙️ Painel de Controle")
        
        # Status do sistema
        device_status = "🟢 GPU Ativa" if torch.cuda.is_available() else "🔵 CPU Ativa"
        st.info(f"**Status:** {device_status}")
        
        # Seleção de demonstração
        demo_options = {
            'sentiment_analysis': '😊 Análise de Sentimentos',
            'text_summarization': '📄 Resumo Inteligente',
            'language_detection': '🌐 Detecção de Idioma',
            'content_moderation': '🛡️ Moderação de Conteúdo',
            'object_detection': '👁️ Visão Computacional',
            'business_qa': '🤖 Assistente Virtual'
        }
        
        selected_demo = st.selectbox(
            "Escolha uma demonstração:",
            options=list(demo_options.keys()),
            format_func=lambda x: demo_options[x]
        )
        
        display_company_info()
    
    # Carregamento dos modelos
    with st.spinner("🚀 Inicializando sistemas de IA AiiT..."):
        models = create_ai_models()
    
    if not models:
        st.error("❌ Falha na inicialização. Entre em contato com o suporte AiiT.")
        return
    
    # Interface principal
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="demo-section">', unsafe_allow_html=True)
        st.subheader(f"🎯 {demo_options[selected_demo]}")
        
        # Interface específica para cada demo
        if selected_demo == 'sentiment_analysis':
            st.markdown("**Analise o sentimento de textos, reviews e feedbacks de clientes**")
            
            sample_texts = [
                "Adorei o produto! Superou minhas expectativas, recomendo!",
                "O atendimento foi péssimo, produto chegou danificado.",
                "Produto ok, nada excepcional mas cumpre o prometido."
            ]
            
            text_input = st.text_area(
                "Digite ou cole o texto para análise:",
                height=120,
                placeholder="Ex: Reviews de produtos, feedbacks de clientes, comentários em redes sociais..."
            )
            
            col_sample1, col_sample2, col_sample3 = st.columns(3)
            with col_sample1:
                if st.button("📝 Texto Positivo", help="Carregar exemplo"):
                    st.session_state.sample_text = sample_texts[0]
            with col_sample2:
                if st.button("📝 Texto Negativo", help="Carregar exemplo"):
                    st.session_state.sample_text = sample_texts[1]
            with col_sample3:
                if st.button("📝 Texto Neutro", help="Carregar exemplo"):
                    st.session_state.sample_text = sample_texts[2]
            
            if 'sample_text' in st.session_state:
                text_input = st.session_state.sample_text
                del st.session_state.sample_text
            
            if st.button("🚀 Analisar Sentimento", type="primary", use_container_width=True):
                if text_input:
                    with st.spinner("Processando análise de sentimentos..."):
                        result = process_ai_request('sentiment_analysis', models, text=text_input)
                        
                        if result.get('status') == 'success':
                            st.success("✅ Análise concluída!")
                            
                            metric_col1, metric_col2 = st.columns(2)
                            with metric_col1:
                                st.metric(
                                    "Sentimento Detectado", 
                                    f"{result['emoji']} {result['sentiment']}",
                                    delta=f"{result['confidence']:.1%} confiança"
                                )
                            with metric_col2:
                                st.metric(
                                    "Tempo de Processamento", 
                                    f"{result['processing_time']}s"
                                )
                            
                            st.info(f"💡 **Recomendação:** {result['recommendation']}")
                        else:
                            st.error(f"❌ {result.get('error', 'Erro desconhecido')}")
        
        elif selected_demo == 'text_summarization':
            st.markdown("**Gere resumos automáticos de documentos, artigos e relatórios**")
            
            text_input = st.text_area(
                "Cole o texto que deseja resumir:",
                height=200,
                placeholder="Cole aqui artigos, relatórios, documentos extensos..."
            )
            
            if st.button("📄 Gerar Resumo", type="primary", use_container_width=True):
                if text_input:
                    with st.spinner("Gerando resumo inteligente..."):
                        result = process_ai_request('text_summarization', models, text=text_input)
                        
                        if result.get('status') == 'success':
                            st.success("✅ Resumo gerado!")
                            
                            col_metric1, col_metric2, col_metric3 = st.columns(3)
                            with col_metric1:
                                st.metric("Palavras Originais", result['original_length'])
                            with col_metric2:
                                st.metric("Palavras no Resumo", result['summary_length'])
                            with col_metric3:
                                st.metric("Taxa de Compressão", f"{result['compression_ratio']}%")
                            
                            st.markdown("**📋 Resumo Gerado:**")
                            st.info(result['summary'])
                        else:
                            st.error(f"❌ {result.get('error', 'Erro desconhecido')}")
        
        elif selected_demo == 'object_detection':
            st.markdown("**Detecte e identifique objetos em imagens automaticamente**")
            
            uploaded_image = st.file_uploader(
                "📷 Faça upload de uma imagem:",
                type=["jpg", "jpeg", "png"],
                help="Formatos: JPG, JPEG, PNG (máx. 200MB)"
            )
            
            if uploaded_image:
                st.image(uploaded_image, caption="Imagem carregada", use_column_width=True)
                
                if st.button("👁️ Analisar Imagem", type="primary", use_container_width=True):
                    with st.spinner("Processando visão computacional..."):
                        result = process_ai_request('object_detection', models, image=uploaded_image)
                        
                        if result.get('status') == 'success':
                            st.success(f"✅ Detectados {result['total_objects']} objetos!")
                            
                            if result['objects']:
                                st.markdown("**🎯 Objetos Detectados:**")
                                for i, obj in enumerate(result['objects'], 1):
                                    st.write(f"{i}. **{obj['object']}** - Confiança: {obj['confidence']}%")
                            else:
                                st.info("Nenhum objeto detectado com alta confiança.")
                        else:
                            st.error(f"❌ {result.get('error', 'Erro desconhecido')}")
        
        elif selected_demo == 'business_qa':
            st.markdown("**Assistente virtual para responder perguntas sobre documentos**")
            
            context_input = st.text_area(
                "📖 Cole o contexto/documento base:",
                height=150,
                placeholder="Cole aqui manuais, políticas, FAQs, documentos técnicos..."
            )
            
            question_input = st.text_input(
                "❓ Faça sua pergunta:",
                placeholder="Ex: Qual é a política de devolução? Como funciona o suporte?"
            )
            
            if st.button("🤖 Consultar Assistente", type="primary", use_container_width=True):
                if context_input and question_input:
                    with st.spinner("Consultando base de conhecimento..."):
                        result = process_ai_request('business_qa', models, 
                                                  question=question_input, context=context_input)
                        
                        if result.get('status') == 'success':
                            st.success("✅ Resposta encontrada!")
                            
                            st.markdown("**🤖 Resposta do Assistente:**")
                            st.info(result['answer'])
                            
                            st.markdown(f"**📊 Confiança:** {result['confidence']}%")
                        else:
                            st.error(f"❌ {result.get('error', 'Erro desconhecido')}")
                else:
                    st.warning("⚠️ Forneça tanto o contexto quanto a pergunta.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="tech-card">', unsafe_allow_html=True)
        st.subheader("ℹ️ Informações Técnicas")
        
        demo_descriptions = {
            'sentiment_analysis': "Analisa emoções em textos usando modelos RoBERTa treinados em milhões de posts sociais. Útil para monitoramento de marca e análise de feedback.",
            'text_summarization': "Utiliza modelo BART para gerar resumos coerentes preservando informações essenciais. Ideal para processamento de documentos longos.",
            'language_detection': "Detecta idiomas usando XLM-RoBERTa multilingual. Suporta mais de 20 idiomas com alta precisão.",
            'content_moderation': "Sistema de moderação baseado em BERT para identificar conteúdo inadequado. Essencial para plataformas digitais.",
            'object_detection': "Usa modelo DETR (Detection Transformer) para identificar objetos em imagens. Aplicável em segurança e automação.",
            'business_qa': "Sistema de perguntas e respostas baseado em RoBERTa-SQuAD. Ideal para criar assistentes virtuais especializados."
        }
        
        st.info(demo_descriptions.get(selected_demo, "Tecnologia de IA avançada."))
        
        st.markdown("**🎯 Casos de Uso:**")
        use_cases = {
            'sentiment_analysis': ["Monitoramento de redes sociais", "Análise de reviews", "Pesquisa de satisfação"],
            'text_summarization': ["Resumo de relatórios", "Síntese de notícias", "Análise de documentos"],
            'object_detection': ["Segurança predial", "Controle de qualidade", "Automação industrial"],
            'business_qa': ["Suporte ao cliente", "FAQ automático", "Assistente interno"]
        }
        
        current_cases = use_cases.get(selected_demo, ["Aplicação empresarial"])
        for case in current_cases:
            st.write(f"• {case}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Métricas de demonstração
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.subheader("📊 Estatísticas da Demo")
        
        if 'demo_usage' not in st.session_state:
            st.session_state.demo_usage = {'total': 0, 'today': 0}
        
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("Demos Executadas", st.session_state.demo_usage['total'], "↗️")
        with col_stat2:
            st.metric("Hoje", st.session_state.demo_usage['today'], "🔥")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer corporativo
    st.markdown("""
    <div class="footer">
        <h4>🚀 AiiT - Transformando Negócios com Inteligência Artificial</h4>
        <p>Esta demonstração apresenta apenas uma pequena amostra das capacidades de IA que podemos implementar em seu negócio.</p>
        <p><strong>Interessado em uma solução personalizada? Entre em contato conosco!</strong></p>
        <p>© 2024 AiiT - Soluções Tecnológicas Inovadoras. Todos os direitos reservados.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
main()
