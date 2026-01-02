from .models import call_llm
from .rag import build_prompt

DEFAULT_CONTEXT = """
AiiT é uma empresa especializada em soluções de Inteligência Artificial,
chatbots empresariais e automação de processos.
"""

def chat(question: str) -> str:
    prompt = build_prompt(question, DEFAULT_CONTEXT)
    return call_llm(prompt)
