def build_prompt(user_question: str, context: str) -> str:
    return f"""
Contexto:
{context}

Pergunta do utilizador:
{user_question}

Responde de forma clara, profissional e objetiva.
"""
