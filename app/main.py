from fastapi import FastAPI
from app.routes import pdf, chat, callback

app = FastAPI(title="RAG PDF Chatbot")

app.include_router(pdf.router, prefix="/pdf", tags=["PDF"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(callback.router, prefix="/callback", tags=["callback"])