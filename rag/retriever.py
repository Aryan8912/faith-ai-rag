# rag/retriever.py
# Builds FAISS vector store using Google Gemini Embeddings
# No HuggingFace download needed — uses your existing GEMINI_API_KEY

import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rag.scripture_loader import load_documents

load_dotenv()

# Singletons — built once at startup
_vectorstore = None
_embeddings = None


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY")
        )
    return _embeddings


def build_vectorstore() -> FAISS:
    """Build FAISS index from all Bible verse documents (runs once)."""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    docs = load_documents()
    embeddings = get_embeddings()
    _vectorstore = FAISS.from_documents(docs, embeddings)
    print(f"[RAG] ✅ FAISS index built with {len(docs)} verses.")
    return _vectorstore


def retrieve_verses(query: str, k: int = 4) -> list[Document]:
    """Retrieve top-k most relevant Bible verses for a query."""
    store = build_vectorstore()
    return store.similarity_search(query, k=k)


def retrieve_with_scores(query: str, k: int = 4) -> list[tuple[Document, float]]:
    """Retrieve top-k verses with similarity scores."""
    store = build_vectorstore()
    return store.similarity_search_with_score(query, k=k)


def format_context(docs: list[Document]) -> str:
    """Format retrieved docs into a clean context block for the prompt."""
    if not docs:
        return "No specific verses retrieved."
    return "\n".join(f"• {doc.page_content}" for doc in docs)


if __name__ == "__main__":
    print("=== Retriever Test ===")
    print("Building FAISS index with Gemini embeddings...")
    build_vectorstore()

    print("\n--- Test 1: Query about love ---")
    docs = retrieve_verses("what does the Bible say about love", k=3)
    for doc in docs:
        print(f"  📖 {doc.page_content}")

    print("\n--- Test 2: Query about suffering ---")
    docs = retrieve_verses("why does God allow suffering and pain", k=3)
    for doc in docs:
        print(f"  📖 {doc.page_content}")

    print("\n--- Test 3: Query with scores ---")
    results = retrieve_with_scores("hope and future plans", k=2)
    for doc, score in results:
        print(f"  Score: {score:.4f} | {doc.metadata['ref']}")

    print("\n--- Test 4: Format context ---")
    docs = retrieve_verses("faith and trust in God", k=2)
    print(format_context(docs))