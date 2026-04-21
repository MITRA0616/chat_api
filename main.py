from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
import ollama

# Load environment variables
load_dotenv()

# API Keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
KIMI_API_KEY = os.getenv("KIMI_API_KEY")

# Validate API keys
if not OPENROUTER_API_KEY:
    raise RuntimeError("❌ Missing OPENROUTER_API_KEY in environment variables.")
if not DEEPSEEK_API_KEY:
    raise RuntimeError("❌ Missing DEEPSEEK_API_KEY in environment variables.")
if not KIMI_API_KEY:
    raise RuntimeError("❌ Missing KIMI_API_KEY in environment variables.")

# OpenRouter Client
openai_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
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
            model="openai/gpt-oss-120b:free",
            messages=messages
        )
        reply = response.choices[0].message.content
        return {"reply": reply}
    except Exception as e:
        return {"reply": f"OpenRouter Error: {str(e)}"}

# ----------------------
# Gemini Endpoint (via OpenRouter)
# ----------------------
@app.post("/gemini")
async def gemini_chat(request: ChatRequest):
    try:
        messages = request.history + [{"role": "user", "content": request.message}]
        response = openai_client.chat.completions.create(
            model="google/gemma-4-31b-it:free",
            messages=messages
        )
        reply = response.choices[0].message.content
        return {"reply": reply}
    except Exception as e:
        return {"reply": f"Gemini (OpenRouter) Error: {str(e)}"}
        
# ----------------------
# Deepseek Endpoint (via Ollama)
# ----------------------
@app.post("/deepseek")
async def deepseek_chat(request: ChatRequest):
    try:
        messages = request.history + [{"role": "user", "content": request.message}]
        
        response = ollama.chat(
            model='deepseek-v3.2:cloud',
            messages=messages,
        )
        
        reply_text = response['message']['content']
        return {"reply": reply_text}

    except Exception as e:
        return {"reply": f"Ollama/DeepSeek Error: {str(e)}"}


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
