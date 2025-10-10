# backend/main.py
from fastapi import FastAPI, Depends, HTTPException
from backend.auth import get_current_user
from backend.payments import create_checkout_session

app = FastAPI(title="AiiT Backend")

@app.get("/me")
def read_me(user: dict = Depends(get_current_user)):
    return user

@app.post("/pay/{plan}")
def pay(plan: str, user: dict = Depends(get_current_user)):
    session = create_checkout_session(user['email'], plan)
    return {"checkout_url": session.url}
