# 🧠 AI Chatbot with RAG (FastAPI + LangChain + FAISS)

This project is a simple Retrieval-Augmented Generation (RAG) chatbot:
it indexes local text documents and answers questions grounded in those documents.

## 🚀 Tech Stack
- FastAPI (API backend)
- LangChain (RAG orchestration)
- FAISS (vector store)
- OpenAI (LLM + embeddings)
- Python

## 📂 Project Structure
ai-chatbot-rag/
│── data/                # .txt documents
│── src/
│   ├── ingest.py        # build FAISS index
│   ├── rag.py           # RAG pipeline
│   ├── api.py           # FastAPI endpoints
│── requirements.txt
│── README.md

## 🔧 Setup

1. Create virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

pip install -r requirements.txt
