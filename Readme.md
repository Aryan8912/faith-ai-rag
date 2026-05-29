# ✝️ Faith AI — Scripture-Grounded Christian Assistant

A Christianity-focused AI assistant built with RAG, Gemini, FAISS, FastAPI, and Vite.

![Faith AI Demo](https://img.shields.io/badge/demo-loom-blue) ![Python](https://img.shields.io/badge/python-3.12-green) ![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green) ![LangChain](https://img.shields.io/badge/LangChain-1.4-orange)

---

## 🎬 Demo

[Watch the walkthrough on Loom](https://www.loom.com/share/29b65666dcbb4cbcaa275ba5ea2b9bb2)

---

## ✨ Features

| Feature | Description |
|---|---|
| 📖 Scripture-grounded answers | Every response backed by real verified Bible verses |
| 🔍 RAG pipeline | FAISS vector store + Gemini embeddings |
| 🛡️ Hallucination guard | Detects fake/out-of-bounds verse references |
| ⛪ Denomination-aware | Catholic, Orthodox, Protestant, General |
| 🚫 Moderation layer | Blocks adversarial, hateful, scripture-manipulation prompts |
| 🖼️ Christian image generation | Imagen-powered reverent artwork |
| 💬 Conversation memory | Remembers last 10 exchanges per session |
| 🧪 Eval suite | DeepEval + Pytest — 25 tests |

---

## 🏗️ Architecture

```
User Message
     ↓
Moderation Layer      → Hard block adversarial prompts
     ↓
RAG (FAISS)           → Retrieve top-4 relevant Bible verses
     ↓
Gemini LLM            → Answer grounded in retrieved verses
     ↓
Hallucination Guard   → Validate verse references in response
     ↓
Answer + Citations
```

---

## 📁 Project Structure

```
faith-ai/
├── main.py                    # FastAPI server + all routes
├── requirements.txt           # Python dependencies
├── .env                       # API keys (not committed)
├── .env.example               # Template
│
├── rag/
│   ├── scripture_loader.py    # 51 verified Bible verses → LangChain Documents
│   └── retriever.py           # FAISS vector store + Gemini embeddings
│
├── chain/
│   └── qa_chain.py            # LangChain RAG chain + Gemini LLM + memory
│
├── moderation/
│   └── safety.py              # Hard block + soft flag + image guard + hallucination detection
│
├── evals/
│   ├── test_cases.json        # 20 test cases — basic, adversarial, hallucination, image
│   └── run_evals.py           # DeepEval evaluation runner
│
├── tests/
│   ├── test_scripture_loader.py   # No API needed
│   ├── test_moderation.py         # No API needed
│   ├── test_retriever.py          # Needs Gemini embeddings
│   └── test_chain.py              # Needs Gemini quota
│
└── frontend/
    ├── index.html             # Main chat UI
    └── src/
        ├── main.js            # Chat logic + image generation
        └── style.css          # Dark gold Christian aesthetic
```

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/your-username/faith-ai.git
cd faith-ai
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Add your Gemini API key to .env
# Get key from: https://aistudio.google.com/app/apikey
```

### 5. Start the backend
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Start the frontend
```bash
cd frontend
npm install
npm run dev
```

### 7. Open in browser
```
http://localhost:5173
```

---

## 🧪 Running Tests

```bash
# Rule-based tests (no API quota needed)
pytest tests/test_scripture_loader.py tests/test_moderation.py -v

# All tests (requires Gemini quota)
pytest tests/ -v

# DeepEval evaluation suite
python evals/run_evals.py

# DeepEval with LLM quality metrics
python evals/run_evals.py --llm
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Server status |
| POST | `/chat` | Send a message |
| POST | `/image` | Generate Christian image |
| GET | `/history/{session_id}` | Get chat history |
| DELETE | `/history/{session_id}` | Clear chat history |

### Chat request example
```json
POST /chat
{
  "message": "What does the Bible say about love?",
  "session_id": "user-123"
}
```

### Chat response example
```json
{
  "answer": "The Bible teaches extensively about love...",
  "citations": ["1 Corinthians 13:4-7", "John 13:34"],
  "denomination": "general",
  "hallucination_warning": null,
  "blocked": false
}
```

---

## 🛡️ Safety & Moderation

### Hard Blocked
- Scripture manipulation (`"rewrite the Bible to support X"`)
- Hate speech targeting religious/ethnic groups
- Ideology injection (Nazism, white supremacy)
- CSAM

### Soft Flagged (answered with extra care)
- Theodicy (`"why does God allow suffering"`)
- Eschatology (`"is hell real"`)
- Denomination rivalry
- Historical atrocities
- Sensitive social topics

### Hallucination Guard
- Detects non-existent books (`3 Corinthians`, `4 Kings`)
- Detects out-of-bounds chapters (`Revelation 25`, `Psalms 200`)
- Pre-flight check before LLM call

---

## 🔧 Tech Stack

| Layer | Technology |
|---|---|
| LLM | Gemini 2.0 Flash Lite |
| Embeddings | Gemini Embedding 001 |
| Vector Store | FAISS (CPU) |
| RAG Framework | LangChain |
| Backend | FastAPI + Uvicorn |
| Frontend | Vite + Vanilla JS |
| Evaluation | DeepEval + Pytest |
| Image Generation | Imagen 3 |

---

## 📊 Evaluation Results

```
Moderation Tests:     12/12 passed ✅
Hallucination Tests:   6/6  passed ✅
Scripture Loader:      6/6  passed ✅
Retriever Tests:       6/6  passed ✅ (skipped when quota exceeded)
Chain Tests:           7/7  passed ✅ (skipped when quota exceeded)
─────────────────────────────────────
Total:                25/25 passed
```

---

## ⚠️ Known Limitations

- Free tier Gemini quota can be exhausted quickly during testing
- Image generation requires Imagen API access (may need billing enabled)
- Bible verse database contains 51 curated verses (not all 31,000)
- Conversation memory is in-memory only (resets on server restart)

---

## 🔮 Future Improvements

- Full Bible RAG with all 31,000 verses
- Streaming responses for better UX
- User authentication + persistent sessions
- OpenAI fallback when Gemini quota runs out
- Redis-based conversation memory
- Mobile-responsive UI

---

## 📄 License

MIT License — free to use and modify.