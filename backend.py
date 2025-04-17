# backend/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel, pipeline

# Configurações para LLMs
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
generator = pipeline('text-generation', model='gpt-3', tokenizer='gpt-3')

# Funções para recuperar e gerar respostas
def encode(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()
    return embeddings

def create_faiss_index(documents):
    embeddings = [encode(doc) for doc in documents]
    faiss_index = faiss.IndexFlatL2(embeddings[0].shape[0])  # Índice de distância L2
    faiss_index.add(np.array(embeddings))  # Adicionar embeddings ao índice
    return faiss_index

def query_faiss_index(faiss_index, query, documents, top_k=5):
    query_embedding = encode(query).reshape(1, -1)
    distances, indices = faiss_index.search(query_embedding, top_k)
    results = [documents[i] for i in indices[0]]
    return results

def gerar_resposta(query, context):
    prompt = f"Baseado no seguinte contexto, responda à pergunta:\nContexto: {context}\nPergunta: {query}"
    resposta = generator(prompt, max_length=100, num_return_sequences=1)
    return resposta[0]['generated_text']

def chatbot(query, faiss_index, documents):
    relevant_documents = query_faiss_index(faiss_index, query, documents)
    context = " ".join(relevant_documents)
    resposta = gerar_resposta(query, context)
    return resposta

# Dados de exemplo (Documentos de FAQ e políticas de suporte)
documents = [
    "A nossa política de devolução permite que os produtos sejam devolvidos dentro de 30 dias após a compra.",
    "Os pagamentos podem ser feitos por cartão de crédito, PayPal e transferência bancária.",
    "Para rastrear o seu pedido, acesse o link de rastreamento enviado por e-mail.",
    "Se tiver problemas com o pagamento, por favor entre em contato com nosso suporte técnico pelo e-mail suporte@ecommerce.com.",
    "Caso tenha problemas técnicos com o produto, recomendamos verificar a seção de problemas comuns no nosso site."
]

# Criar índice FAISS
faiss_index = create_faiss_index(documents)

# Inicializar FastAPI
app = FastAPI()

# Modelo de dados para entrada
class Message(BaseModel):
    user_message: str

@app.post("/chat")
async def chat(message: Message):
    resposta = chatbot(message.user_message, faiss_index, documents)
    return {"response": resposta}
