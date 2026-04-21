from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
KIMI_API_KEY = os.getenv("KIMI_API_KEY")

# Validate API keys
if not OPENAI_API_KEY:
    raise RuntimeError("❌ Missing OPENAI_API_KEY in environment variables.")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ Missing GEMINI_API_KEY in environment variables.")
if not DEEPSEEK_API_KEY:
    raise RuntimeError("❌ Missing DEEPSEEK_API_KEY in environment variables.")
if not KIMI_API_KEY:
    raise RuntimeError("❌ Missing KIMI_API_KEY in environment variables.")
# OpenAI Client
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Gemini Endpoint
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1/models/"
    f"gemini-2.5-pro:generateContent?key={GEMINI_API_KEY}"
)
# DeepSeek endpoint
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

# Kimi endpoint (Moonshot API)
KIMI_URL = "https://api.moonshot.cn/v1/chat/completions"

# FastAPI App
app = FastAPI(
    title="Multi-AI Chat API",
    description="Unified API for ChatGPT and Google Gemini",
    version="1.0.0"
)

# CORS (allow Flutter frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: change to your Flutter app domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request body model
class ChatRequest(BaseModel):
    message: str
    history: list = []  # For ChatGPT conversation context

# ----------------------
# ChatGPT Endpoint
# ----------------------
@app.post("/chatgpt")
async def chatgpt_chat(request: ChatRequest):
    try:
        messages = request.history + [{"role": "user", "content": request.message}]
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",  # or gpt-4o, gpt-3.5-turbo
            messages=messages
        )
        reply = response.choices[0].message.content
        return {"reply": reply}
    except Exception as e:
        return {"reply": f"ChatGPT Error: {str(e)}"}

# ----------------------
# Gemini Endpoint
# ----------------------
@app.post("/gemini")
async def gemini_chat(request: ChatRequest):
    try:
        payload = {
            "contents": [
                {
                    "parts": [{"text": request.message}]
                }
            ]
        }
        response = requests.post(GEMINI_URL, json=payload, timeout=30)
        gemini_reply = response.json()

        if "error" in gemini_reply:
            return {"reply": f"Gemini Error: {gemini_reply['error'].get('message', 'Unknown error')}"}

        reply_text = gemini_reply["candidates"][0]["content"]["parts"][0]["text"]
        return {"reply": reply_text}

    except requests.exceptions.RequestException as e:
        return {"reply": f"Network Error: {str(e)}"}
    except Exception:
        return {"reply": "Sorry, could not parse Gemini's response."}
        
# ----------------------
# Deepseek Endpoint
# ----------------------
@app.post("/deepseek")
async def deepseek_chat(request: ChatRequest):
    try:
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": request.message}
            ],
            "stream": False
        }

        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }

        response = requests.post(DEEPSEEK_URL, json=payload, headers=headers, timeout=30)
        deepseek_reply = response.json()

        # 🛠 Debug log
        print("DeepSeek Raw:", deepseek_reply)

        if "choices" in deepseek_reply and len(deepseek_reply["choices"]) > 0:
            reply_text = deepseek_reply["choices"][0]["message"]["content"]
            return {"reply": reply_text}
        elif "error" in deepseek_reply:
            return {"reply": f"DeepSeek Error: {deepseek_reply['error'].get('message', 'Unknown error')}"}
        else:
            return {"reply": f"Unexpected response format: {deepseek_reply}"}

    except requests.exceptions.RequestException as e:
        return {"reply": f"Network Error: {str(e)}"}
    except Exception as e:
        return {"reply": f"Parsing Error: {str(e)}"}


# ----------------------
# Kimi Endpoint
# ----------------------
@app.post("/kimi")
async def kimi_chat(request: ChatRequest):
    try:
        messages = request.history + [{"role": "user", "content": request.message}]
        payload = {
            "model": "moonshot-v1-8k",
            "messages": messages
        }
        headers = {
            "Authorization": f"Bearer {KIMI_API_KEY}",
            "Content-Type": "application/json"
        }
        response = requests.post(KIMI_URL, headers=headers, json=payload, timeout=30)
        
        # 🛠 Debug log
        print("Kimi Raw:", response.text)
        
        kimi_reply = response.json()
        
        if "choices" in kimi_reply and len(kimi_reply["choices"]) > 0:
            reply_text = kimi_reply["choices"][0]["message"]["content"]
            return {"reply": reply_text}
        elif "error" in kimi_reply:
            return {"reply": f"Kimi Error: {kimi_reply['error'].get('message', 'Unknown error')}"}
        else:
            return {"reply": f"Unexpected response format: {kimi_reply}"}
            
    except requests.exceptions.RequestException as e:
        return {"reply": f"Network Error: {str(e)}"}
    except Exception as e:
        return {"reply": f"Parsing Error: {str(e)}"}

# ----------------------
# Health Check
# ----------------------
@app.get("/")
async def root():
    return {"status": "ok", "message": "Multi-AI Chat API is running 🚀"}
