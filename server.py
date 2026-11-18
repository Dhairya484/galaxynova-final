import os
import re
import json
import asyncio
import random
import logging
import base64
import binascii
from pathlib import Path
from datetime import datetime, timezone

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

# ==================================================
# 1. CONFIGURATION
# ==================================================
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DATABASE_FILE = BASE_DIR / "training_data.json"
MAX_RETRIES_PER_KEY = 2
REQUEST_TIMEOUT = 90
MAX_IMAGE_BYTES = 5 * 1024 * 1024
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GalaxyNovaAI")

# ==================================================
# 2. LOAD MULTIPLE GROQ API KEYS
# ==================================================
API_KEYS = []
for number in range(1, 21):
    key = os.getenv(f"GROQ_API_KEY_{number}")
    if key and key.strip():
        API_KEYS.append(key.strip())

if not API_KEYS:
    key = os.getenv("GROQ_API_KEY")
    if key and key.strip():
        API_KEYS.append(key.strip())

logger.info("Galaxy Nova AI starting")
logger.info("Groq API keys loaded: %s", len(API_KEYS))
logger.info("Text model: %s | Vision model: %s", MODEL, VISION_MODEL)

# ==================================================
# 3. FASTAPI APPLICATION
# ==================================================
app = FastAPI(
    title="Galaxy Nova AI",
    description="Galaxy Nova AI Assistant with image understanding",
    version="5.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ==================================================
# 4. GREETINGS
# ==================================================
GREETINGS = {
    "hi": "Hello! 👋 Welcome to Galaxy Nova AI. How can I help you?",
    "hello": "Hey there! 🚀 What would you like to know?",
    "hey": "Hey! How can I assist you today?",
    "hii": "Hii! 😊 Welcome to Galaxy Nova AI!",
    "good morning": "Good morning! ☀️ Have a wonderful day!",
    "good afternoon": "Good afternoon! How can I help you?",
    "good evening": "Good evening! 🌙 What can I do for you?",
    "good night": "Good night! Have a wonderful rest. 🌙",
    "how are you": "I'm doing great! Ready to answer your questions.",
    "who are you": "I'm Galaxy Nova AI, your intelligent AI assistant.",
    "what is your name": "My name is Galaxy Nova AI! 🚀",
    "thanks": "You're welcome! 😊",
    "thank you": "Happy to help!",
    "bye": "Goodbye! See you again on Galaxy Nova AI. 👋"
}

def greeting_reply(message: str):
    cleaned = re.sub(r"[^a-zA-Z\s]", "", message.lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return GREETINGS.get(cleaned)

# ==================================================
# 5. LOAD LOCAL KNOWLEDGE DATABASE
# ==================================================
def load_database():
    if not DATABASE_FILE.exists():
        logger.warning("training_data.json not found")
        return []
    try:
        with open(DATABASE_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("entries", "data", "knowledge", "topics"):
                if isinstance(data.get(key), list):
                    return data[key]
        logger.warning("Unknown database structure")
        return []
    except Exception:
        logger.exception("Database loading failed")
        return []

DATABASE = load_database()
logger.info("Database entries loaded: %s", len(DATABASE))

STOP_WORDS = {
    "the", "is", "am", "are", "was", "were", "a", "an", "and", "or",
    "but", "to", "of", "in", "on", "at", "for", "with", "what", "who",
    "when", "where", "why", "how", "can", "could", "would", "should",
    "do", "does", "did", "tell", "me", "about", "please", "explain",
    "i", "you", "it", "this", "that", "my", "your"
}

def tokenize(text):
    words = re.findall(r"[a-zA-Z0-9]+", str(text).lower())
    return {word for word in words if len(word) > 1 and word not in STOP_WORDS}

def search_database(question):
    question_words = tokenize(question)
    if not question_words:
        return None
    best_answer = None
    best_score = 0
    for entry in DATABASE:
        if not isinstance(entry, dict):
            continue
        answer = entry.get("answer") or entry.get("response") or entry.get("text")
        if not answer:
            continue
        searchable = " ".join(str(entry.get(k, "")) for k in
                              ("question", "topic", "category", "keywords", "title"))
        entry_words = tokenize(searchable)
        if not entry_words:
            continue
        score = len(question_words.intersection(entry_words)) / max(len(question_words), 1)
        saved_question = str(entry.get("question", "")).lower().strip()
        if saved_question and saved_question == question.lower().strip():
            score += 1
        if score > best_score:
            best_score, best_answer = score, str(answer)
    if best_score >= 0.45:
        logger.info("Answer found in local database")
        return best_answer
    return None

# ==================================================
# 6. REQUEST MODELS
# ==================================================
class ImageInput(BaseModel):
    data_uri: str
    mime_type: str = ""

class ChatRequest(BaseModel):
    messages: list = Field(default_factory=list)
    message: str = ""
    image: ImageInput | None = None

# ==================================================
# 7. SYSTEM INSTRUCTION
# ==================================================
SYSTEM_INSTRUCTION = """
You are Galaxy Nova AI, an intelligent and helpful AI assistant.
Answer accurately and clearly. Be friendly and respectful. Explain difficult
topics simply. Help with HTML, CSS, JavaScript, Python and other programming.
When asked for code, provide complete working code and explain where files go.
Use Markdown formatting when useful: headings, bold, lists, fenced code blocks,
and valid Markdown tables. Do not wrap the entire answer in a code block.
Admit uncertainty. Never invent facts or sources. Do not claim to browse unless
browsing is actually available. When an image is attached, inspect it carefully
and answer the user's question about what is visible. Do not claim to identify
a person in an image.
You are Galaxy Nova AI.
"""

# ==================================================
# 8. PREPARE CHAT HISTORY
# ==================================================
def prepare_contents(messages, fallback_message=""):
    contents = []
    if not messages and fallback_message:
        messages = [{"role": "user", "content": fallback_message}]
    for item in messages[-20:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "user")).lower()
        if role in ("assistant", "model"):
            groq_role = "assistant"
        elif role == "user":
            groq_role = "user"
        else:
            continue
        content = item.get("content") or item.get("text") or item.get("parts") or ""
        # Conversation history is text-only. The current image is added separately.
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    text_parts.append(str(part.get("text", "")))
                else:
                    text_parts.append(str(part))
            content = "\n".join(text_parts)
        if not isinstance(content, str):
            content = str(content)
        content = content.strip()
        if content:
            contents.append({"role": groq_role, "content": content})
    while contents and contents[0]["role"] != "user":
        contents.pop(0)
    return contents

def validate_image(image: ImageInput):
    mime = image.mime_type.lower().strip()
    if mime not in ALLOWED_IMAGE_TYPES:
        raise ValueError("Unsupported image type. Use PNG, JPG or WEBP.")
    match = re.fullmatch(r"data:(image/(?:jpeg|png|webp));base64,([A-Za-z0-9+/=\s]+)", image.data_uri)
    if not match:
        raise ValueError("The uploaded image data is invalid.")
    if match.group(1) != mime:
        raise ValueError("Image MIME type does not match its data.")
    encoded = re.sub(r"\s+", "", match.group(2))
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("The uploaded image could not be decoded.")
    if not raw:
        raise ValueError("The uploaded image is empty.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Image is too large. Maximum upload size is 5 MB.")
    return image.data_uri

# ==================================================
# 9. GROQ API WITH KEY ROTATION AND RETRIES
# ==================================================
async def ask_groq(messages, image_data_uri=None):
    if not API_KEYS:
        return "No Groq API keys configured. Add GROQ_API_KEY_1 to your environment variables."

    model = VISION_MODEL if image_data_uri else MODEL
    outgoing = [dict(item) for item in messages]
    if image_data_uri:
        # Attach the image to the latest user message using Groq's OpenAI-compatible format.
        for item in reversed(outgoing):
            if item["role"] == "user":
                item["content"] = [
                    {"type": "text", "text": item["content"]},
                    {"type": "image_url", "image_url": {"url": image_data_uri}}
                ]
                break

    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_INSTRUCTION}] + outgoing,
        "temperature": 0.7,
        "max_completion_tokens": 4096,
        "stream": False
    }
    errors = []
    retryable_statuses = {408, 409, 429, 500, 502, 503, 504}

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for index, api_key in enumerate(API_KEYS, start=1):
            for attempt in range(MAX_RETRIES_PER_KEY):
                try:
                    logger.info("Trying Groq key %s, attempt %s, model %s",
                                index, attempt + 1, model)
                    response = await client.post(
                        GROQ_URL,
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {api_key}"
                        },
                        json=payload
                    )
                    if response.status_code == 200:
                        data = response.json()
                        choices = data.get("choices", [])
                        if choices:
                            reply = choices[0].get("message", {}).get("content", "")
                            if isinstance(reply, str) and reply.strip():
                                logger.info("Groq succeeded using key %s", index)
                                return reply.strip()
                        errors.append(f"Key {index}: Empty response")
                        break

                    status = response.status_code
                    error_text = response.text[:500]
                    errors.append(f"Key {index}: HTTP {status}: {error_text}")
                    logger.warning("Groq key %s returned HTTP %s", index, status)
                    if status not in retryable_statuses:
                        break
                    retry_after = response.headers.get("retry-after")
                    try:
                        delay = min(max(float(retry_after), 1), 30) if retry_after else 2 ** (attempt + 1)
                    except ValueError:
                        delay = 2 ** (attempt + 1)
                except httpx.TimeoutException:
                    errors.append(f"Key {index}: Request timed out")
                    delay = 2 ** (attempt + 1)
                except httpx.TransportError:
                    errors.append(f"Key {index}: Network error")
                    delay = 2 ** (attempt + 1)
                except Exception as error:
                    logger.exception("Unexpected Groq error")
                    errors.append(f"Key {index}: {str(error)}")
                    break

                if attempt < MAX_RETRIES_PER_KEY - 1:
                    await asyncio.sleep(delay + random.uniform(0, 1))
            logger.info("Moving to next Groq API key")

    logger.error("All Groq keys failed: %s", errors)
    if any("401" in error for error in errors):
        return "A Groq API key was rejected. Check your API keys."
    if any("403" in error for error in errors):
        return "Groq denied access. Check project permissions and model access."
    if any("404" in error for error in errors):
        return f"Groq model '{model}' was not found or is unavailable. Check your environment variables."
    if any("429" in error for error in errors):
        return "Groq rate limit or usage quota reached. Please try again later."
    if any(any(str(code) in error for code in (500, 502, 503, 504)) for error in errors):
        return "Groq is temporarily experiencing a service problem. Please try again shortly."
    if any("timed out" in error.lower() for error in errors):
        return "Groq requests timed out. Please try again shortly."
    return "Groq AI is temporarily unavailable. Check your API keys and Render logs."

# ==================================================
# 10. MAIN CHAT ENDPOINT
# ==================================================
@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        messages = prepare_contents(request.messages, request.message)
        if not messages:
            return JSONResponse(status_code=400, content={"reply": "Please enter a message first."})

        latest_question = next(
            (item["content"] for item in reversed(messages) if item["role"] == "user"), ""
        )
        if not latest_question:
            return JSONResponse(status_code=400, content={"reply": "Please send a valid message."})

        image_data_uri = None
        if request.image:
            try:
                image_data_uri = validate_image(request.image)
            except ValueError as error:
                return JSONResponse(status_code=400, content={"reply": str(error), "detail": str(error)})

        # Greetings and local database are used only for text-only messages.
        if not image_data_uri:
            greeting = greeting_reply(latest_question)
            if greeting:
                return {"reply": greeting, "source": "greeting"}
            database_answer = search_database(latest_question)
            if database_answer:
                return {"reply": database_answer, "source": "database"}

        reply = await ask_groq(messages, image_data_uri)
        return {"reply": reply, "source": "groq_vision" if image_data_uri else "groq"}
    except Exception:
        logger.exception("Chat endpoint error")
        return JSONResponse(status_code=500, content={"reply": "An unexpected server error occurred."})

# ==================================================
# 11. HEALTH CHECK
# ==================================================
@app.get("/api/health")
async def health():
    return {
        "status": "online",
        "app": "Galaxy Nova AI",
        "provider": "Groq",
        "model": MODEL,
        "vision_model": VISION_MODEL,
        "api_keys_loaded": len(API_KEYS),
        "database_entries": len(DATABASE),
        "image_upload_enabled": True,
        "time": datetime.now(timezone.utc).isoformat()
    }

# ==================================================
# 12. WEBSITE HOME PAGE
# ==================================================
@app.get("/")
async def home():
    index_file = BASE_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"app": "Galaxy Nova AI", "status": "online", "message": "Backend is running."}

# ==================================================
# 13. LOCAL SERVER START
# ==================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
