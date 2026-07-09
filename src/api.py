from fastapi import FastAPI
from pydantic import BaseModel
from .rag import ask_question

app = FastAPI(title="RAG Chatbot API")


class Question(BaseModel):
    question: str


@app.get("/")
def root():
    return {"message": "RAG Chatbot API is running"}


@app.post("/chat")
def chat(payload: Question):
    result = ask_question(payload.question)
    return {
        "question": payload.question,
        "answer": result["answer"],
        "sources": result["sources"],
    }
