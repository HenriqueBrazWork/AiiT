from fastapi import FastAPI
from pydantic import BaseModel
from .chat import chat

app = FastAPI(title="AiiT SmartChat API")

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    answer = chat(req.message)
    return {"response": answer}
