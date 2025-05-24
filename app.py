import streamlit as st
import logging
from transformers import pipeline
import torch
from PIL import Image
from datetime import datetime
import time

# Configuração da página
st.set_page_config(page_title="AiiT - Demo de IA",page_icon="🚀",layout="wide")

# CSS personalizado simples
st.markdown("""<style>.main-header {background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);padding: 2rem;border-radius: 10px;color: white;text-align: center;margin-bottom: 2rem;}.demo-card {background: #f8f9fa;padding: 1.5rem;border-radius: 10px;border-left: 4px solid #2a5298;margin: 1rem 0;</style>""", unsafe_allow_html=True)

# Modelos leves para demo rápida
@st.cache_resource
def load_lite_models():
    """Carrega modelos menores para demonstração rápida"""
    try:
        models = {}
    # Apenas modelos pequenos e rápidos
    with st.spinner("Carregando modelo de sentimentos..."):
        models['sentiment'] = pipeline("sentiment-analysis", 
        model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    with st.spinner("Carregando modelo de resumo..."):
        models['summarization'] = pipeline("summarization", 
        model="sshleifer/distilbart-cnn-6-6")
    return models
    except Exception as e:
        st.error(f"Erro ao carregar modelos: {e}")
    return {}

def analyze_sentiment(text, model):
"""Análise de sentimento otimizada"""
    start_time = time.time()
    try:
        result = model(text)[0]
        processing_time = round(time.time() - start_time, 2)

# Mapeamento mais claro
    sentiment_map = {
        'LABEL_0': {'name': 'Negativo', 'emoji': '😞', 'color': 'red'},
        'LABEL_1': {'name': 'Neutro', 'emoji': '😐', 'color': 'orange'},
        'LABEL_2': {'name': 'Positivo', 'emoji': '😊', 'color': 'green'},
        'NEGATIVE': {'name': 'Negativo', 'emoji': '😞', 'color': 'red'},
        'POSITIVE': {'name': 'Positivo', 'emoji': '😊', 'color': 'green'}
        }
    sentiment_info = sentiment_map.get(result['label'], 
        {'name': result['label'], 'emoji': '🤔', 'color': 'blue'})
    return {
        'sentiment': sentiment_info['name'],
        'emoji': sentiment_info['emoji'],
        'confidence': round(result['score'] * 100, 1),
        'processing_time': processing_time,
        'status': 'success'
        }
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def summarize_text(text, model):
"""Resumo de texto otimizado"""
    start_time = time.time()
    try:
        if len(text.split()) < 20:
    return {'status': 'error', 'error': 'Texto precisa ter pelo menos 20 palavras'}

    result = model(text, max_length=100, min_length=20, do_sample=False)[0]
    processing_time = round(time.time() - start_time, 2)
    return {
    'summary': result['summary_text'],
    'original_words': len(text.split()),
    'summary_words': len(result['summary_text'].split()),
    'processing_time': processing_time,
    'status': 'success'
}
    except Exception as e:
        return {'status': 'error', 'error': str(e)}

def main():
# Cabeçalho
st.markdown("""
    <div class="main-header">
    <h1>🚀 AiiT - Soluções Tecnológicas Inovadoras</h1>
    <h3>Demonstração de Inteligência Artificial</h3>
    <p>Versão Lite - Modelos Otimizados</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar com info da empresa
    with st.sidebar:
    st.markdown("### 🏢 AiiT")
    st.info("""
    **Transformando negócios com IA**
    
    • Análise de Sentimentos
    • Processamento de Texto
    • Automação Inteligente
    • Consultoria Especializada
    """)
    
    st.markdown("---")
    st.markdown("### 📞 Contato")
    st.markdown("📧 henriquebrazwork@hotmail.com")
    st.markdown("📱 +351 xxx xxx xxx")
    
    # Carregamento dos modelos
    st.write("### ⚡ Inicializando IA...")
    models = load_lite_models()
    
    if not models:
        st.error("❌ Falha ao carregar modelos")
    return st.success("✅ Modelos carregados! Pronto para usar.")
    # Tabs para diferentes funções
    tab1, tab2 = st.tabs(["😊 Análise de Sentimentos", "📄 Resumo de Texto"])
    with tab1:
        st.markdown('<div class="demo-card">', unsafe_allow_html=True)
        st.subheader("🎯 Análise de Sentimentos")
        st.write("Descubra se um texto expressa sentimento positivo, negativo ou neutro")
        
    # Exemplos rápidos
    examples = [
        "Adorei este produto! Superou todas as minhas expectativas!",
        "O atendimento foi péssimo, não recomendo para ninguém.",
        "O produto é ok, nada excepcional mas funciona bem."
        ]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("😊 Exemplo Positivo"):
            st.session_state.sentiment_text = examples[0]
    with col2:
        if st.button("😞 Exemplo Negativo"):
            st.session_state.sentiment_text = examples[1]
    with col3:
        if st.button("😐 Exemplo Neutro"):
            st.session_state.sentiment_text = examples[2]
    
    # Input de texto
    sentiment_text = st.text_area("Digite o texto para análise:",
    value=st.session_state.get('sentiment_text', ''),
    height=100, placeholder="Ex: Reviews, comentários, feedbacks..."
    )
    
    if st.button("🚀 Analisar Sentimento", type="primary"):
        if sentiment_text.strip():
            with st.spinner("Analisando..."):
                result = analyze_sentiment(sentiment_text, models['sentiment'])
    
    if result['status'] == 'success':
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sentimento", f"{result['emoji']} {result['sentiment']}")
        with col2:
            st.metric("Confiança", f"{result['confidence']}%")
        with col3:
            st.metric("Tempo", f"{result['processing_time']}s")
        
        # Recomendação
    if result['confidence'] > 80:
        if result['sentiment'] == 'Positivo':
             st.success("💡 Ótimo feedback! Use em marketing e testimonials.")
                elif result['sentiment'] == 'Negativo':
                    st.warning("⚠️ Feedback negativo. Considere ação corretiva.")
                    else:
                    st.info("📊 Feedback neutro. Oportunidade de engajamento.")
                    else:
                    st.info("🔍 Resultado inconclusivo. Análise manual recomendada.")
                    else:
                    st.error(f"❌ Erro: {result['error']}")
                    else:
                    st.warning("⚠️ Digite um texto para análise")
                    
        st.markdown('</div>', unsafe_allow_html=True)
        with tab2:
            st.markdown('<div class="demo-card">', unsafe_allow_html=True)
            st.subheader("📄 Resumo Automático")
            st.write("Gere resumos concisos de textos longos automaticamente")
            
        # Texto de exemplo
        sample_text = """
        A inteligência artificial (IA) está revolucionando diversos setores da economia mundial. 
        Empresas de todos os tamanhos estão implementando soluções de IA para automatizar processos, 
        melhorar a eficiência operacional e oferecer experiências mais personalizadas aos clientes. 
        No setor de saúde, a IA está sendo usada para diagnósticos mais precisos e desenvolvimento 
        de novos medicamentos. Na educação, sistemas inteligentes personalizam o aprendizado para 
        cada aluno. O varejo utiliza IA para recomendações de produtos e otimização de estoque. 
        Apesar dos benefícios, é importante considerar questões éticas e de privacidade no 
        desenvolvimento e implementação dessas tecnologias.
        """
        
        if st.button("📝 Carregar Texto de Exemplo"):
            st.session_state.summary_text = sample_text
            
            summary_text = st.text_area(
            "Cole o texto que deseja resumir:",
            value=st.session_state.get('summary_text', ''),
            height=200,
            placeholder="Cole aqui artigos, relatórios, documentos..."
            )
        if st.button("📄 Gerar Resumo", type="primary"):
            if summary_text.strip():
                with st.spinner("Gerando resumo..."):
                    result = summarize_text(summary_text, models['summarization'])
            if result['status'] == 'success':
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Palavras Originais", result['original_words'])
            with col2:
                st.metric("Palavras Resumo", result['summary_words'])
            with col3:
                st.metric("Tempo", f"{result['processing_time']}s")
                st.markdown("**📋 Resumo Gerado:**")
                st.info(result['summary'])    
                compression = round((1 - result['summary_words']/result['original_words']) * 100, 1)
                st.success(f"✨ Texto comprimido em {compression}%!")
                else:
                st.error(f"❌ Erro: {result['error']}")
                else:
                st.warning("⚠️ Cole um texto para resumir")
                
        st.markdown('</div>', unsafe_allow_html=True)
                
    # Footer
    st.markdown("---")
    st.markdown(""" 
    <div style="text-align: center; padding: 2rem; background: #f1f3f4; border-radius: 10px;">
    <h4>🚀 AiiT - Transformando Negócios com IA</h4>
    <p>Esta é uma demonstração das capacidades de IA que podemos implementar em seu negócio.</p>
    <p><strong>Interessado em uma solução personalizada? Entre em contato!</strong></p>
    </div>
    """, unsafe_allow_html=True)
                                                                                
     if __name__ == "__main__":
    main()
