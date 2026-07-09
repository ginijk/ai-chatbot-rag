import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()

INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "faiss_index")


def load_vectorstore():
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vectorstore


def get_rag_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )
    return qa_chain


def ask_question(question: str):
    qa_chain = get_rag_chain()
    result = qa_chain({"query": question})

    answer = result["result"]
    sources = result["source_documents"]

    source_titles = [os.path.basename(doc.metadata.get("source", "unknown")) for doc in sources]

    return {
        "answer": answer,
        "sources": source_titles,
    }


if __name__ == "__main__":
    print(ask_question("What is this knowledge base about?"))
