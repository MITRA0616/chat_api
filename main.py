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

# Validate API keys
if not OPENROUTER_API_KEY:
    raise RuntimeError("❌ Missing OPENROUTER_API_KEY in environment variables.")
if not DEEPSEEK_API_KEY:
    raise RuntimeError("❌ Missing DEEPSEEK_API_KEY in environment variables.")

# OpenRouter Client
openai_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

# Ollama Client
ollama_client = ollama.Client(host='https://ollama.com')

# DeepSeek endpoint
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

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
        
        response = ollama_client.chat(
            model='deepseek-v3.2:cloud',
            messages=messages,
        )
        
        reply_text = response['message']['content']
        return {"reply": reply_text}

    except Exception as e:
        return {"reply": f"Ollama/DeepSeek Error: {str(e)}"}


# ----------------------
# Llama Endpoint (via OpenRouter)
# ----------------------
@app.post("/llama")
async def llama_chat(request: ChatRequest):
    try:
        messages = request.history + [{"role": "user", "content": request.message}]
        response = openai_client.chat.completions.create(
            model="meta-llama/llama-3-8b-instruct",
            messages=messages
        )
        reply = response.choices[0].message.content
        return {"reply": reply}
    except Exception as e:
        return {"reply": f"Llama Error: {str(e)}"}

# ----------------------
# Health Check
# ----------------------
@app.get("/")
async def root():
    return {"status": "ok", "message": "Multi-AI Chat API is running 🚀"}
