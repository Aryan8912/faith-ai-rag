# tests/test_retriever.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from langchain_core.documents import Document


def is_quota_error(e):
    return "429" in str(e) or "quota" in str(e).lower() or "Error embedding" in str(e)


def safe_retrieve(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        if is_quota_error(e):
            pytest.skip("⏳ Gemini embedding quota exceeded — run again after reset")
        raise


def test_vectorstore_builds():
    from rag.retriever import build_vectorstore
    store = safe_retrieve(build_vectorstore)
    assert store is not None


def test_retrieve_returns_results():
    from rag.retriever import retrieve_verses
    docs = safe_retrieve(retrieve_verses, "what does the Bible say about love", k=3)
    assert len(docs) == 3
    assert all(isinstance(d, Document) for d in docs)


def test_retrieve_relevant_verses():
    from rag.retriever import retrieve_verses
    docs = safe_retrieve(retrieve_verses, "love one another", k=3)
    refs = [d.metadata["ref"] for d in docs]
    assert any("John" in r or "Corinthians" in r for r in refs)


def test_retrieve_suffering_verses():
    from rag.retriever import retrieve_verses
    docs = safe_retrieve(retrieve_verses, "why does God allow suffering", k=3)
    assert len(docs) > 0


def test_retrieve_with_scores():
    from rag.retriever import retrieve_with_scores
    results = safe_retrieve(retrieve_with_scores, "hope and future", k=2)
    assert len(results) == 2
    # Fix: FAISS returns numpy.float32 not Python float — use numbers.Number
    import numbers
    assert all(isinstance(score, numbers.Number) for _, score in results)


def test_format_context():
    from rag.retriever import retrieve_verses, format_context
    docs = safe_retrieve(retrieve_verses, "faith", k=2)
    context = format_context(docs)
    assert context != ""
    assert "•" in context


def test_format_context_empty():
    from rag.retriever import format_context
    context = format_context([])
    assert context == "No specific verses retrieved."