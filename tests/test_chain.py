# tests/test_chain.py
# Tests for chain/qa_chain.py — needs Gemini API key + quota

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from chain.qa_chain import ask, get_history, clear_history


def is_quota_error(e):
    return "429" in str(e) or "quota" in str(e).lower() or "ResourceExhausted" in str(e)


def safe_ask(*args, **kwargs):
    """Wraps ask() — skips test if quota exceeded instead of failing."""
    try:
        return ask(*args, **kwargs)
    except Exception as e:
        if is_quota_error(e):
            pytest.skip("⏳ Gemini quota exceeded — run again after reset")
        raise


def test_ask_basic():
    result = safe_ask("What does the Bible say about love?", session_id="test-love")
    assert "answer" in result
    assert "citations" in result
    assert len(result["answer"]) > 0
    assert len(result["citations"]) > 0


def test_ask_denomination_catholic():
    result = safe_ask("What does the Catholic Church teach about purgatory?", session_id="test-catholic")
    assert result["denomination"] == "catholic"


def test_ask_denomination_orthodox():
    result = safe_ask("What is theosis in Orthodox theology?", session_id="test-orthodox")
    assert result["denomination"] == "orthodox"


def test_hallucination_warning():
    result = safe_ask("What does Revelation 25:3 say?", session_id="test-hall")
    assert result["hallucination_warning"] is not None


def test_memory_follow_up():
    session = "test-memory"
    safe_ask("What does the Bible say about salvation?", session_id=session)
    history = get_history(session)
    assert len(history) == 2
    safe_ask("Can you give me one more verse on that?", session_id=session)
    history2 = get_history(session)
    assert len(history2) == 4


def test_clear_history():
    session = "test-clear"
    safe_ask("Tell me about faith", session_id=session)
    clear_history(session)
    assert len(get_history(session)) == 0


def test_soft_flag_theodicy():
    result = safe_ask("Why does God allow suffering?", session_id="test-theodicy", flag="theodicy")
    assert len(result["answer"]) > 0


if __name__ == "__main__":
    print("\n=== test_chain.py ===")
    print("⚠️  These tests require Gemini API quota\n")
    try:
        test_ask_basic()
        test_ask_denomination_catholic()
        test_ask_denomination_orthodox()
        test_hallucination_warning()
        test_memory_follow_up()
        test_clear_history()
        test_soft_flag_theodicy()
        print("\n✅ All chain tests passed!")
    except Exception as e:
        if is_quota_error(e):
            print(f"\n⏳ Quota exceeded — run again after reset")
        else:
            raise