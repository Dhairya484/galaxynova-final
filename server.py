
import os
import re
import json
import asyncio
import random
import logging

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

# Load environment variables
load_dotenv(BASE_DIR / ".env")

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

DATABASE_FILE = BASE_DIR / "training_data.json"

MAX_RETRIES_PER_KEY = 2
REQUEST_TIMEOUT = 90

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s"
)

logger = logging.getLogger("GalaxyNovaAI")


# ==================================================
# 2. LOAD GEMINI API KEYS
# ==================================================

API_KEYS = []

for number in range(1, 21):

    key = os.getenv(f"GEMINI_API_KEY_{number}")

    if key and key.strip():
        API_KEYS.append(key.strip())

# Fallback to a single key if numbered keys are absent
if not API_KEYS:

    single_key = os.getenv("GEMINI_API_KEY")

    if single_key and single_key.strip():
        API_KEYS.append(single_key.strip())


logger.info("Galaxy Nova AI starting")
logger.info("Gemini keys loaded: %s", len(API_KEYS))
logger.info("Gemini model: %s", MODEL)


# ==================================================
# 3. FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="Galaxy Nova AI",
    description="Galaxy Nova AI Assistant",
    version="3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ==================================================
# 4. GREETING DATABASE
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

    "thank you": "Happy to help! Ask me anything else.",

    "bye": "Goodbye! See you again on Galaxy Nova AI. 👋"
}


def greeting_reply(message):

    cleaned = re.sub(
        r"[^a-zA-Z\s]",
        "",
        message.lower()
    ).strip()

    cleaned = re.sub(r"\s+", " ", cleaned)

    return GREETINGS.get(cleaned)


# ==================================================
# 5. LOAD KNOWLEDGE DATABASE
# ==================================================

def load_database():

    if not DATABASE_FILE.exists():

        logger.warning("training_data.json not found")

        return []

    try:

        with open(
            DATABASE_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):

            for key in [
                "entries",
                "data",
                "knowledge",
                "topics"
            ]:

                if isinstance(data.get(key), list):
                    return data[key]

        logger.warning("Unknown database structure")

        return []

    except Exception:

        logger.exception("Database loading failed")

        return []


DATABASE = load_database()

logger.info(
    "Database entries loaded: %s",
    len(DATABASE)
)


# ==================================================
# 6. DATABASE SEARCH ENGINE
# ==================================================

STOP_WORDS = {
    "the", "is", "am", "are", "was", "were",
    "a", "an", "and", "or", "but", "to",
    "of", "in", "on", "at", "for", "with",
    "what", "who", "when", "where", "why",
    "how", "can", "could", "would", "should",
    "do", "does", "did", "tell", "me",
    "about", "please", "explain", "i",
    "you", "it", "this", "that", "my",
    "your"
}


def tokenize(text):

    words = re.findall(
        r"[a-zA-Z0-9]+",
        str(text).lower()
    )

    return {
        word
        for word in words
        if len(word) > 1 and word not in STOP_WORDS
    }


def search_database(question):

    question_words = tokenize(question)

    if not question_words:
        return None

    best_answer = None
    best_score = 0

    for entry in DATABASE:

        if not isinstance(entry, dict):
            continue

        answer = (
            entry.get("answer")
            or entry.get("response")
            or entry.get("text")
        )

        if not answer:
            continue

        searchable_text = " ".join([

            str(entry.get("question", "")),

            str(entry.get("topic", "")),

            str(entry.get("category", "")),

            str(entry.get("keywords", "")),

            str(entry.get("title", ""))

        ])

        entry_words = tokenize(searchable_text)

        if not entry_words:
            continue

        common_words = question_words.intersection(
            entry_words
        )

        score = (
            len(common_words)
            / max(len(question_words), 1)
        )

        saved_question = str(
            entry.get("question", "")
        ).lower().strip()

        if (
            saved_question
            and saved_question == question.lower().strip()
        ):

            score += 1

        if score > best_score:

            best_score = score
            best_answer = str(answer)

    if best_score >= 0.45:

        logger.info("Database answer found")

        return best_answer

    return None


# ==================================================
# 7. CHAT REQUEST FORMAT
# ==================================================

class ChatRequest(BaseModel):

    messages: list = Field(default_factory=list)

    message: str = ""


# ==================================================
# 8. AI SYSTEM INSTRUCTIONS
# ==================================================

SYSTEM_INSTRUCTION = """
You are Galaxy Nova AI, a helpful and intelligent AI assistant.

Your responsibilities:

1. Answer questions clearly and accurately.
2. Be friendly and respectful.
3. Explain difficult topics in simple language.
4. Help with HTML, CSS, JavaScript, Python and other coding.
5. When asked to write code, provide complete working code.
6. Include all necessary files when a project requires multiple files.
7. Explain how to run the code when useful.
8. Help debug and fix programming errors.
9. Use headings and lists when appropriate.
10. Admit when you do not know something.
11. Never invent facts or sources.
12. Do not claim to browse the internet unless browsing is available.

You are Galaxy Nova AI.
"""


# ==================================================
# 9. CONVERSATION PREPARATION
# ==================================================

def prepare_contents(messages, fallback_message=""):

    contents = []

    if not messages and fallback_message:

        messages = [
            {
                "role": "user",
                "content": fallback_message
            }
        ]

    for item in messages[-20:]:

        if not isinstance(item, dict):
            continue

        role = str(
            item.get("role", "user")
        ).lower()

        if role in ("assistant", "model"):

            gemini_role = "model"

        elif role == "user":

            gemini_role = "user"

        else:

            continue

        content = (
            item.get("content")
            or item.get("text")
            or item.get("parts")
            or ""
        )

        if isinstance(content, list):

            text_parts = []

            for part in content:

                if isinstance(part, dict):

                    text_parts.append(
                        str(part.get("text", ""))
                    )

                else:

                    text_parts.append(str(part))

            content = "\n".join(text_parts)

        if not isinstance(content, str):

            content = str(content)

        content = content.strip()

        if content:

            contents.append({

                "role": gemini_role,

                "parts": [
                    {
                        "text": content
                    }
                ]

            })

    # Gemini conversations should start with a user message
    while contents and contents[0]["role"] != "user":

        contents.pop(0)

    return contents


# ==================================================
# 10. GEMINI API WITH MULTI-KEY RETRIES
# ==================================================

async def ask_gemini(messages):

    if not API_KEYS:

        return (
            "No Gemini API keys are configured. "
            "Please add your API keys to Render Environment Variables."
        )

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{MODEL}:generateContent"
    )

    payload = {

        "systemInstruction": {

            "parts": [
                {
                    "text": SYSTEM_INSTRUCTION
                }
            ]

        },

        "contents": messages,

        "generationConfig": {

            "temperature": 0.7,

            "maxOutputTokens": 2048

        }

    }

    errors = []

    retryable_statuses = {
        408, 429, 500, 502, 503, 504
    }

    async with httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT
    ) as client:

        # Try every available key
        for index, api_key in enumerate(
            API_KEYS,
            start=1
        ):

            for attempt in range(
                MAX_RETRIES_PER_KEY
            ):

                try:

                    logger.info(
                        "Trying Gemini key %s, attempt %s",
                        index,
                        attempt + 1
                    )

                    response = await client.post(

                        url,

                        headers={

                            "Content-Type": "application/json",

                            "x-goog-api-key": api_key

                        },

                        json=payload

                    )

                    # SUCCESS
                    if response.status_code == 200:

                        data = response.json()

                        candidates = data.get(
                            "candidates",
                            []
                        )

                        if candidates:

                            parts = (
                                candidates[0]
                                .get("content", {})
                                .get("parts", [])
                            )

                            reply = "".join(

                                part.get("text", "")

                                for part in parts

                                if isinstance(part, dict)

                            ).strip()

                            if reply:

                                logger.info(
                                    "Gemini succeeded with key %s",
                                    index
                                )

                                return reply

                        errors.append(
                            f"Key {index}: Empty response"
                        )

                        break

                    # FAILURE
                    status = response.status_code

                    error_text = response.text[:500]

                    logger.warning(
                        "Key %s returned HTTP %s",
                        index,
                        status
                    )

                    errors.append(
                        f"Key {index}: HTTP {status}: {error_text}"
                    )

                    # Do not retry permanent errors
                    if status not in retryable_statuses:

                        break

                except httpx.TimeoutException:

                    logger.warning(
                        "Key %s request timed out",
                        index
                    )

                    errors.append(
                        f"Key {index}: Request timed out"
                    )

                except httpx.TransportError as error:

                    logger.warning(
                        "Key %s network error: %s",
                        index,
                        str(error)
                    )

                    errors.append(
                        f"Key {index}: Network error"
                    )

                except Exception as error:

                    logger.exception(
                        "Unexpected Gemini error"
                    )

                    errors.append(
                        f"Key {index}: {str(error)}"
                    )

                    break

                # Retry with exponential backoff
                if attempt < MAX_RETRIES_PER_KEY - 1:

                    delay = (
                        2 ** (attempt + 1)
                        + random.uniform(0, 2)
                    )

                    logger.info(
                        "Waiting %.1f seconds before retry",
                        delay
                    )

                    await asyncio.sleep(delay)

            logger.info(
                "Moving to Gemini key %s",
                index + 1
            )

    # ==================================================
    # 11. ERROR RESPONSES
    # ==================================================

    logger.error(
        "All Gemini keys failed: %s",
        errors
    )

    if any("404" in error for error in errors):

        return (
            f"Gemini model '{MODEL}' is unavailable "
            "or not accessible to your API project. "
            "Please check the model name and API access."
        )

    if any("429" in error for error in errors):

        return (
            "Gemini API rate limit or quota exceeded. "
            "Please try again later."
        )

    if any("503" in error for error in errors):

        return (
            "Gemini is experiencing high demand. "
            "I tried multiple keys and retries, "
            "but the service is still unavailable. "
            "Please try again shortly."
        )

    if any("timed out" in error.lower() for error in errors):

        return (
            "Gemini requests timed out. "
            "Please try again shortly."
        )

    return (
        "Gemini is temporarily unavailable. "
        "Please check your API keys and Render logs."
    )


# ==================================================
# 12. MAIN CHAT ENDPOINT
# ==================================================

@app.post("/api/chat")
async def chat(request: ChatRequest):

    try:

        messages = prepare_contents(
            request.messages,
            request.message
        )

        if not messages:

            return JSONResponse(

                status_code=400,

                content={
                    "reply": "Please enter a message first."
                }

            )

        # Get latest user message
        latest_question = ""

        for item in reversed(messages):

            if item["role"] == "user":

                latest_question = (
                    item["parts"][0]["text"]
                )

                break

        if not latest_question:

            return JSONResponse(

                status_code=400,

                content={
                    "reply": "Please send a valid message."
                }

            )

        # FIRST: Greeting database
        greeting = greeting_reply(
            latest_question
        )

        if greeting:

            return {

                "reply": greeting,

                "source": "greeting"

            }

        # SECOND: Knowledge database
        database_answer = search_database(
            latest_question
        )

        if database_answer:

            return {

                "reply": database_answer,

                "source": "database"

            }

        # THIRD: Gemini API
        gemini_reply = await ask_gemini(
            messages
        )

        return {

            "reply": gemini_reply,

            "source": "gemini"

        }

    except Exception:

        logger.exception("Chat endpoint error")

        return JSONResponse(

            status_code=500,

            content={

                "reply": (
                    "An unexpected server error occurred. "
                    "Please try again."
                )

            }

        )


# ==================================================
# 13. HEALTH CHECK
# ==================================================

@app.get("/api/health")
async def health():

    return {

        "status": "online",

        "app": "Galaxy Nova AI",

        "model": MODEL,

        "api_keys_loaded": len(API_KEYS),

        "database_entries": len(DATABASE),

        "time": datetime.now(
            timezone.utc
        ).isoformat()

    }


# ==================================================
# 14. WEBSITE HOME PAGE
# ==================================================

@app.get("/")
async def home():

    index_file = BASE_DIR / "index.html"

    if index_file.exists():

        return FileResponse(index_file)

    return {

        "app": "Galaxy Nova AI",

        "status": "online",

        "message": "Backend is running."

    }


# ==================================================
# 15. LOCAL SERVER START
# ==================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "server:app",

        host="0.0.0.0",

        port=int(
            os.getenv("PORT", "8000")
        )

    )
    
