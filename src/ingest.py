import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "faiss_index")


def load_documents():
    # Load all .txt files from data/
    loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
    )
    docs = loader.load()
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(docs)
    return chunks


def build_index():
    docs = load_documents()
    if not docs:
        raise ValueError("No documents found in data/")

    chunks = split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Save index locally
    vectorstore.save_local(INDEX_PATH)
    print("✅ FAISS index created and saved at:", INDEX_PATH)


if __name__ == "__main__":
    build_index()
