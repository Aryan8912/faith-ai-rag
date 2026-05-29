# chain/qa_chain.py
# LangChain RAG chain — connects retriever + Gemini LLM + memory
# Handles: denomination-aware prompts, hallucination guard, soft flags

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from rag.retriever import retrieve_verses, format_context
from moderation.safety import detect_denomination, detect_hallucination_attempt

load_dotenv()

# ── LLM Setup ─────────────────────────────────────────────────
def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model="models/gemini-2.0-flash-lite",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0.7,
    )

# ── Denomination-aware system prompts ─────────────────────────
SYSTEM_PROMPTS = {
    "general": """You are a knowledgeable and compassionate Christian AI assistant.
Answer questions using the Bible verses provided as context.
Always cite specific Bible references (e.g. John 3:16) when answering.
Be warm, respectful, and biblically grounded.
If you are unsure about a verse reference, say so — never fabricate scripture.
Only use the verses provided in context or well-known verified verses.""",

    "catholic": """You are a knowledgeable Catholic Christian AI assistant.
Answer questions in line with Catholic teaching, the Catechism, and Sacred Tradition.
You may reference deuterocanonical books (Sirach, Wisdom, Maccabees etc).
Always cite specific Bible references when answering.
Be warm, respectful, and faithful to Catholic doctrine.
Never fabricate scripture references.""",

    "orthodox": """You are a knowledgeable Eastern Orthodox Christian AI assistant.
Answer questions in line with Orthodox theology, the Church Fathers, and Holy Tradition.
Reference the Septuagint tradition where relevant.
Always cite specific Bible references when answering.
Be warm, respectful, and faithful to Orthodox doctrine.
Never fabricate scripture references.""",

    "protestant": """You are a knowledgeable Protestant Christian AI assistant.
Answer questions grounded in Scripture alone (Sola Scriptura).
Always cite specific Bible references (book, chapter, verse) when answering.
Be warm, respectful, and faithful to evangelical Protestant doctrine.
Never fabricate scripture references.""",

    "soft_flag": """You are a compassionate Christian AI assistant handling a sensitive theological question.
Approach this topic with grace, humility, and respect for different perspectives.
Present what Scripture says without being divisive or judgmental.
Acknowledge that Christians may hold different views on this topic.
Always cite Bible references accurately — never fabricate scripture.""",
}


def build_prompt(denomination: str, flag: str = None) -> ChatPromptTemplate:
    """Build denomination-aware prompt template."""
    if flag in ("theodicy", "eschatology", "sensitive_social", "denomination_rivalry", "historical_atrocities"):
        system = SYSTEM_PROMPTS["soft_flag"]
    else:
        system = SYSTEM_PROMPTS.get(denomination, SYSTEM_PROMPTS["general"])

    return ChatPromptTemplate.from_messages([
        ("system", system + "\n\nRelevant Bible verses for context:\n{context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])


# ── In-memory conversation store ──────────────────────────────
# session_id -> list of messages
_conversation_store: dict[str, list] = {}


def get_history(session_id: str) -> list:
    return _conversation_store.get(session_id, [])


def save_history(session_id: str, human: str, ai: str):
    if session_id not in _conversation_store:
        _conversation_store[session_id] = []
    _conversation_store[session_id].append(HumanMessage(content=human))
    _conversation_store[session_id].append(AIMessage(content=ai))
    # Keep last 10 exchanges (20 messages) to avoid token overflow
    _conversation_store[session_id] = _conversation_store[session_id][-20:]


def clear_history(session_id: str):
    _conversation_store.pop(session_id, None)


# ── Main RAG chain ─────────────────────────────────────────────
def ask(question: str, session_id: str = "default", flag: str = None) -> dict:
    """
    Main entry point.
    1. Detect denomination from question
    2. Retrieve relevant Bible verses (RAG)
    3. Check for hallucination attempt
    4. Build prompt + run LLM
    5. Save to memory
    6. Return answer + citations
    """

    # 1. Denomination detection
    denomination = detect_denomination(question)

    # 2. Hallucination guard — check before LLM call
    hallucination_warning = detect_hallucination_attempt(question)

    # 3. RAG — retrieve relevant verses
    docs = retrieve_verses(question, k=4)
    context = format_context(docs)
    citations = [doc.metadata["ref"] for doc in docs]

    # 4. Build prompt
    prompt = build_prompt(denomination, flag)

    # 5. Get conversation history
    history = get_history(session_id)

    # 6. Run chain
    llm = get_llm()
    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "history": history,
        "question": question,
    })

    # 7. Save to memory
    save_history(session_id, question, answer)

    return {
        "answer": answer,
        "citations": citations,
        "denomination": denomination,
        "hallucination_warning": hallucination_warning,
        "context_used": context,
    }


if __name__ == "__main__":
    import json

    print("=== QA Chain Test ===\n")
    session = "test-session"

    # Test 1: Basic question
    print("--- Test 1: Basic salvation question ---")
    result = ask("What does the Bible say about salvation?", session)
    print(f"Answer: {result['answer'][:300]}...")
    print(f"Citations: {result['citations']}")
    print(f"Denomination: {result['denomination']}")

    # Test 2: Follow-up (memory test)
    print("\n--- Test 2: Follow-up question (memory test) ---")
    result2 = ask("Can you give me more verses about that?", session)
    print(f"Answer: {result2['answer'][:300]}...")

    # Test 3: Denomination-aware
    print("\n--- Test 3: Catholic question ---")
    result3 = ask("What does the Catholic Church teach about purgatory?", "catholic-session")
    print(f"Answer: {result3['answer'][:300]}...")
    print(f"Denomination: {result3['denomination']}")

    # Test 4: Hallucination attempt
    print("\n--- Test 4: Hallucination guard ---")
    result4 = ask("What does Revelation 25:3 say?", "test-session-2")
    print(f"Warning: {result4['hallucination_warning']}")
    print(f"Answer: {result4['answer'][:300]}...")

    # Test 5: Soft flag — theodicy
    print("\n--- Test 5: Theodicy (soft flag) ---")
    result5 = ask("Why does God allow suffering and pain?", "theodicy-session")
    print(f"Answer: {result5['answer'][:300]}...")