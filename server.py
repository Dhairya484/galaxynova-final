
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

load_dotenv(BASE_DIR / ".env")

MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b"
)

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

DATABASE_FILE = BASE_DIR / "training_data.json"

MAX_RETRIES_PER_KEY = 2
REQUEST_TIMEOUT = 90

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

# Support a single key as well
if not API_KEYS:

    key = os.getenv("GROQ_API_KEY")

    if key and key.strip():
        API_KEYS.append(key.strip())


logger.info("Galaxy Nova AI starting")
logger.info("Groq API keys loaded: %s", len(API_KEYS))
logger.info("Groq model: %s", MODEL)


# ==================================================
# 3. FASTAPI APPLICATION
# ==================================================

app = FastAPI(
    title="Galaxy Nova AI",
    description="Galaxy Nova AI Assistant",
    version="4.0"
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

    "thank you": "Happy to help!",

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

            for key in (
                "entries",
                "data",
                "knowledge",
                "topics"
            ):

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

        logger.info("Answer found in local database")

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
You are Galaxy Nova AI, an intelligent and helpful AI assistant.

Your responsibilities:

1. Answer questions accurately and clearly.
2. Be friendly and respectful.
3. Explain difficult topics in simple language.
4. Help with HTML, CSS, JavaScript, Python and other programming.
5. When asked to write code, provide complete working code.
6. Include all required files for multi-file projects.
7. Explain how to run the code when useful.
8. Debug programming errors.
9. Use headings and lists when appropriate.
10. Admit when you do not know something.
11. Never invent facts or sources.
12. Do not claim to browse the internet unless browsing is available.

For coding requests:
- Understand the user's requirements.
- Write complete code.
- Include necessary HTML, CSS and JavaScript.
- Explain where each file belongs.
- Avoid unnecessary placeholders.

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

            groq_role = "assistant"

        elif role == "user":

            groq_role = "user"

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

                "role": groq_role,

                "content": content

            })

    # Conversation must start with a user message
    while contents and contents[0]["role"] != "user":

        contents.pop(0)

    return contents


# ==================================================
# 10. GROQ API WITH MULTIPLE KEYS AND RETRIES
# ==================================================

async def ask_groq(messages):

    if not API_KEYS:

        return (
            "No Groq API keys configured. "
            "Add GROQ_API_KEY_1 to your environment variables."
        )

    payload = {

        "model": MODEL,

        "messages": [
            {
                "role": "system",
                "content": SYSTEM_INSTRUCTION
            }
        ] + messages,

        "temperature": 0.7,

        "max_completion_tokens": 4096,

        "stream": False

    }

    errors = []

    retryable_statuses = {
        408, 409, 429, 500, 502, 503, 504
    }

    async with httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT
    ) as client:

        for index, api_key in enumerate(
            API_KEYS,
            start=1
        ):

            for attempt in range(MAX_RETRIES_PER_KEY):

                try:

                    logger.info(
                        "Trying Groq key %s, attempt %s",
                        index,
                        attempt + 1
                    )

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

                            reply = (
                                choices[0]
                                .get("message", {})
                                .get("content", "")
                            )

                            if isinstance(reply, str) and reply.strip():

                                logger.info(
                                    "Groq succeeded using key %s",
                                    index
                                )

                                return reply.strip()

                        errors.append(
                            f"Key {index}: Empty response"
                        )

                        break

                    status = response.status_code

                    error_text = response.text[:700]

                    logger.warning(
                        "Groq key %s returned HTTP %s",
                        index,
                        status
                    )

                    errors.append(
                        f"Key {index}: HTTP {status}: {error_text}"
                    )

                    if status not in retryable_statuses:

                        break

                    # Respect Groq's retry-after header if present
                    retry_after = response.headers.get(
                        "retry-after"
                    )

                    if retry_after:

                        try:
                            delay = min(
                                max(float(retry_after), 1),
                                30
                            )
                        except ValueError:
                            delay = 2 ** (attempt + 1)

                    else:

                        delay = 2 ** (attempt + 1)

                except httpx.TimeoutException:

                    logger.warning(
                        "Groq key %s timed out",
                        index
                    )

                    errors.append(
                        f"Key {index}: Request timed out"
                    )

                    delay = 2 ** (attempt + 1)

                except httpx.TransportError as error:

                    logger.warning(
                        "Groq network error: %s",
                        str(error)
                    )

                    errors.append(
                        f"Key {index}: Network error"
                    )

                    delay = 2 ** (attempt + 1)

                except Exception as error:

                    logger.exception(
                        "Unexpected Groq error"
                    )

                    errors.append(
                        f"Key {index}: {str(error)}"
                    )

                    break

                if attempt < MAX_RETRIES_PER_KEY - 1:

                    delay += random.uniform(0, 1)

                    logger.info(
                        "Retrying in %.1f seconds",
                        delay
                    )

                    await asyncio.sleep(delay)

            logger.info(
                "Moving to next Groq API key"
            )

    logger.error(
        "All Groq keys failed: %s",
        errors
    )

    if any("401" in error for error in errors):

        return (
            "A Groq API key was rejected. "
            "Please check your API keys."
        )

    if any("403" in error for error in errors):

        return (
            "Groq denied access. Check your account, "
            "project permissions and model access."
        )

    if any("404" in error for error in errors):

        return (
            f"Groq model '{MODEL}' was not found or is unavailable. "
            "Check the model name in your environment variables."
        )

    if any("429" in error for error in errors):

        return (
            "Groq rate limit or usage quota reached. "
            "Please try again later."
        )

    if any(
        str(code) in error
        for error in errors
        for code in (500, 502, 503, 504)
    ):

        return (
            "Groq is temporarily experiencing a service problem. "
            "Multiple retries were attempted. Please try again shortly."
        )

    if any("timed out" in error.lower() for error in errors):

        return (
            "Groq requests timed out. "
            "Please try again shortly."
        )

    return (
        "Groq AI is temporarily unavailable. "
        "Check your API keys and Render logs."
    )


# ==================================================
# 11. MAIN CHAT ENDPOINT
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

        latest_question = ""

        for item in reversed(messages):

            if item["role"] == "user":

                latest_question = item["content"]

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

        # SECOND: Local knowledge database
        database_answer = search_database(
            latest_question
        )

        if database_answer:

            return {
                "reply": database_answer,
                "source": "database"
            }

        # THIRD: Groq AI
        groq_reply = await ask_groq(
            messages
        )

        return {
            "reply": groq_reply,
            "source": "groq"
        }

    except Exception:

        logger.exception("Chat endpoint error")

        return JSONResponse(

            status_code=500,

            content={
                "reply": "An unexpected server error occurred."
            }

        )


# ==================================================
# 12. HEALTH CHECK
# ==================================================

@app.get("/api/health")
async def health():

    return {

        "status": "online",

        "app": "Galaxy Nova AI",

        "provider": "Groq",

        "model": MODEL,

        "api_keys_loaded": len(API_KEYS),

        "database_entries": len(DATABASE),

        "time": datetime.now(
            timezone.utc
        ).isoformat()

    }


# ==================================================
# 13. WEBSITE HOME PAGE
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
# 14. LOCAL SERVER START
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
