import os, re, json
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
DATABASE_FILE = BASE_DIR / "training_data.json"
INDEX_FILE = BASE_DIR / "index.html"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

GEMINI_KEYS = [os.getenv(f"GEMINI_API_KEY_{i}", "").strip() for i in range(1, 21)]
GEMINI_KEYS = [key for key in GEMINI_KEYS if key]
if not GEMINI_KEYS and os.getenv("GEMINI_API_KEY"):
    GEMINI_KEYS = [os.getenv("GEMINI_API_KEY").strip()]

http_client = None
db: dict[str, Any] = {}

def load_database():
    global db
    try:
        with DATABASE_FILE.open("r", encoding="utf-8") as f:
            db = json.load(f)
        db.setdefault("greetings", {})
        db.setdefault("knowledge", [])
        print(f"Loaded {len(db['knowledge'])} knowledge entries.")
    except Exception as exc:
        db = {"greetings": {}, "knowledge": []}
        print(f"Database loading error: {exc}")

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()

STOP_WORDS = {
    "what", "is", "are", "the", "a", "an", "of", "to", "in", "on", "for",
    "and", "or", "how", "why", "when", "where", "who", "tell", "me",
    "explain", "about", "please", "can", "you", "does", "do", "it", "its"
}

def tokens(text: str) -> set[str]:
    return {w for w in normalize(text).split() if len(w) > 1 and w not in STOP_WORDS}

def greeting_reply(message: str):
    cleaned = normalize(message)
    greetings_map = db.get("greetings", {})
    if cleaned in greetings_map:
        return greetings_map[cleaned]
    short_greetings = {"hii": "hi", "heyy": "hey", "goodmorning": "good morning"}
    if cleaned in short_greetings:
        return greetings_map.get(short_greetings[cleaned])
    return None

def search_database(message: str):
    query_tokens = tokens(message)
    if not query_tokens:
        return None
    best, best_score = None, 0.0
    for entry in db.get("knowledge", []):
        searchable = tokens(str(entry.get("topic", "")) + " " +
                            " ".join(map(str, entry.get("keywords", []))))
        if not searchable:
            continue
        overlap = query_tokens & searchable
        coverage = len(overlap) / max(1, len(query_tokens))
        precision = len(overlap) / max(1, len(searchable))
        score = coverage * 0.82 + precision * 0.18
        if normalize(str(entry.get("topic", ""))) in normalize(message):
            score += 0.35
        if score > best_score:
            best, best_score = entry, score
    if best and best_score >= 0.34:
        return {"topic": best.get("topic", ""), "answer": best.get("answer", ""),
                "score": round(best_score, 3)}
    return None

class ChatRequest(BaseModel):
    messages: list[dict[str, Any]] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    load_database()
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=15.0))
    yield
    await http_client.aclose()

app = FastAPI(title="Galaxy Nova AI", version="2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def home():
    if INDEX_FILE.exists():
        return FileResponse(INDEX_FILE)
    return {"message": "Galaxy Nova AI backend is running. Add index.html beside server.py."}

@app.get("/api/health")
async def health():
    return {"status": "online", "assistant": "Galaxy Nova AI",
            "database_entries": len(db.get("knowledge", [])),
            "gemini_configured": bool(GEMINI_KEYS)}

@app.post("/api/chat")
async def chat(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=400, detail="No message was provided.")
    messages = []
    for item in request.messages[-12:]:
        role = str(item.get("role", "user"))
        content = item.get("content", item.get("parts", ""))
        if isinstance(content, list):
            content = " ".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content[:8000]})
    latest = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "").strip()
    if not latest:
        raise HTTPException(status_code=400, detail="No user message was found.")

    # FIRST: dedicated greetings section.
    greeting = greeting_reply(latest)
    if greeting:
        return {"reply": greeting, "source": "greetings", "assistant": "Galaxy Nova AI"}

    # SECOND: local knowledge database. Gemini is not called for a database match.
    match = search_database(latest)
    if match:
        return {"reply": match["answer"], "source": "database",
                "matched_topic": match["topic"], "match_score": match["score"],
                "assistant": "Galaxy Nova AI"}

    # THIRD: Gemini fallback only when no suitable database entry is found.
    if not GEMINI_KEYS:
        return {"reply": "I couldn't find that in my database. Gemini fallback is not configured yet. Add GEMINI_API_KEY_1 in your hosting environment variables.",
                "source": "database_miss", "assistant": "Galaxy Nova AI"}

    contents = [{"role": "user" if m["role"] == "user" else "model",
                 "parts": [{"text": m["content"]}]} for m in messages]
    system_text = ("You are Galaxy Nova AI, a helpful and friendly assistant. "
                   "Answer accurately and clearly. If unsure, say so. "
                   "Use simple explanations when useful.")
    errors = []
    for key in GEMINI_KEYS:
        try:
            url = f"{GEMINI_BASE_URL}/{GEMINI_MODEL}:generateContent"
            payload = {
                "system_instruction": {"parts": [{"text": system_text}]},
                "contents": contents,
                "generationConfig": {"temperature": 0.7, "topP": 0.9, "maxOutputTokens": 2048}
            }
            response = await http_client.post(url, params={"key": key}, json=payload)
            if response.status_code == 200:
                data = response.json()
                candidates = data.get("candidates", [])
                answer = ""
                if candidates:
                    answer = "".join(p.get("text", "") for p in candidates[0].get("content", {}).get("parts", [])).strip()
                if answer:
                    return {"reply": answer, "source": "gemini", "assistant": "Galaxy Nova AI"}
                errors.append("Gemini returned an empty response.")
            else:
                try:
                    detail = response.json().get("error", {}).get("message", response.text)
                except Exception:
                    detail = response.text
                errors.append(f"HTTP {response.status_code}: {detail[:180]}")
        except Exception as exc:
            errors.append(str(exc)[:180])
    return {"reply": "Gemini is temporarily unavailable. Please try again. " + " | ".join(errors[:2]),
            "source": "gemini_error", "assistant": "Galaxy Nova AI"}
