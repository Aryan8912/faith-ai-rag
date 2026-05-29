# main.py
# FastAPI server — all routes for Faith AI
# Routes: /chat, /image, /history, /clear, /health

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

from moderation.safety import moderate_text, moderate_image_prompt, detect_hallucination_attempt
from chain.qa_chain import ask, get_history, clear_history

load_dotenv()

app = FastAPI(
    title="Faith AI",
    description="Christianity-focused AI assistant with RAG-based Bible grounding",
    version="1.0.0"
)

# ── CORS ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response Models ─────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    answer: str
    citations: list[str]
    denomination: str
    hallucination_warning: Optional[str]
    blocked: bool = False
    block_reason: Optional[str] = None

class ImageRequest(BaseModel):
    prompt: str
    session_id: Optional[str] = "default"

class ImageResponse(BaseModel):
    image_url: Optional[str]
    safe_prompt: Optional[str]
    blocked: bool = False
    block_reason: Optional[str] = None


# ── Routes ────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "service": "Faith AI"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # 1. Moderation check
    mod = moderate_text(req.message)
    if not mod.allowed:
        return ChatResponse(
            answer=mod.reason,
            citations=[],
            denomination="general",
            hallucination_warning=None,
            blocked=True,
            block_reason=mod.flag,
        )

    # 2. Hallucination pre-check
    hall_warning = detect_hallucination_attempt(req.message)

    # 3. Ask the RAG chain
    try:
        result = ask(
            question=req.message,
            session_id=req.session_id,
            flag=mod.flag,
        )
        return ChatResponse(
            answer=result["answer"],
            citations=result["citations"],
            denomination=result["denomination"],
            hallucination_warning=hall_warning or result.get("hallucination_warning"),
            blocked=False,
        )
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            raise HTTPException(
                status_code=429,
                detail="API quota exceeded. Please try again later."
            )
        raise HTTPException(status_code=500, detail=f"Error: {error_msg}")


@app.post("/image", response_model=ImageResponse)
async def generate_image(req: ImageRequest):
    # 1. Moderation check
    mod = moderate_image_prompt(req.prompt)
    if not mod.allowed:
        return ImageResponse(
            image_url=None,
            safe_prompt=None,
            blocked=True,
            block_reason=mod.reason,
        )

    # 2. Build safe Christian image prompt
    safe_prompt = (
        f"A beautiful, reverent, and uplifting Christian artwork depicting: {req.prompt}. "
        f"Style: classical religious painting, warm light, peaceful and inspiring."
    )

    # 3. Generate image using Gemini Imagen
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

        imagen = genai.ImageGenerationModel("imagen-3.0-generate-002")
        result = imagen.generate_images(
            prompt=safe_prompt,
            number_of_images=1,
            safety_filter_level="block_most",
            aspect_ratio="1:1",
        )

        if result.images:
            import base64
            img_data = base64.b64encode(result.images[0]._image_bytes).decode("utf-8")
            return ImageResponse(
                image_url=f"data:image/png;base64,{img_data}",
                safe_prompt=safe_prompt,
                blocked=False,
            )
        else:
            raise HTTPException(status_code=500, detail="Image generation returned no results.")

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            raise HTTPException(status_code=429, detail="Image generation quota exceeded.")
        raise HTTPException(status_code=500, detail=f"Image generation error: {error_msg}")


@app.get("/history/{session_id}")
def get_chat_history(session_id: str):
    history = get_history(session_id)
    messages = []
    for msg in history:
        messages.append({
            "role": "user" if msg.__class__.__name__ == "HumanMessage" else "assistant",
            "content": msg.content,
        })
    return {"session_id": session_id, "messages": messages}


@app.delete("/history/{session_id}")
def clear_chat_history(session_id: str):
    clear_history(session_id)
    return {"status": "cleared", "session_id": session_id}


# ── Serve frontend ────────────────────────────────────────────
if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend/src"), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse("frontend/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)