# Mitra - Multi-AI Chat API

A **FastAPI-based backend** for the Mitra chatbot that unifies **OpenAI (ChatGPT)** and **Google Gemini** into a single API.  
This service is designed to work seamlessly with a **Flutter frontend** (or any other client) to provide intelligent conversational responses from multiple AI providers.

---

## 🚀 Features

- **Dual AI Support**: Access **ChatGPT** and **Gemini** from one backend.
- **CORS Enabled**: Ready to connect with your Flutter app or any frontend.
- **Conversation History**: Supports chat context for ChatGPT.
- **Environment-based Configuration**: Secure API keys using `.env`.
- **Health Check Endpoint**: Quick status check for deployment monitoring.

---

## 📦 Tech Stack

- **[FastAPI](https://fastapi.tiangolo.com/)** – High-performance Python web framework
- **[OpenAI Python SDK](https://github.com/openai/openai-python)** – ChatGPT integration
- **[Google Gemini API](https://ai.google.dev/)** – Gemini integration
- **[Pydantic](https://docs.pydantic.dev/)** – Data validation
- **[Requests](https://docs.python-requests.org/)** – HTTP requests
- **[Python-dotenv](https://pypi.org/project/python-dotenv/)** – Environment variables

---

## 📂 Project Structure

```
chat_api/
│── main.py            # Main FastAPI app
│── .env               # Environment variables (not committed to git)
│── requirements.txt   # Python dependencies
└── README.md          # Project documentation
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/MitraAI/chat_api.git
cd chat_api
```

### 2️⃣ Create & Activate Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Create `.env` File
Create a `.env` file in the root directory and add your API keys:
```
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

### 5️⃣ Run the Server
```bash
uvicorn main:app --reload --port 8000
```
The server will be available at:  
```
http://127.0.0.1:8000
```

---

## 🔌 API Endpoints

### **1. Chat with ChatGPT**
`POST /chatgpt`

**Request Body:**
```json
{
  "message": "Hello, ChatGPT!",
  "history": [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello! How can I help you today?"}
  ]
}
```

**Response:**
```json
{
  "reply": "Hello! How can I assist you today?"
}
```

---

### **2. Chat with Gemini**
`POST /gemini`

**Request Body:**
```json
{
  "message": "Tell me a joke."
}
```

**Response:**
```json
{
  "reply": "Why don't scientists trust atoms? Because they make up everything!"
}
```

---

### **3. Health Check**
`GET /`

**Response:**
```json
{
  "status": "ok",
  "message": "Multi-AI Chat API is running 🚀"
}
```

---

## 🛡 Security Notes
- Never commit your `.env` file.
- In production, restrict `allow_origins` in CORS to your frontend domain.
- Rotate API keys periodically.

---

## 🏗 Roadmap
- [ ] Add streaming responses for ChatGPT
- [ ] Integrate more AI providers
- [ ] Add authentication for API access
- [ ] Dockerize for deployment

---

## 📜 License
This project is licensed under the **MIT License**.

---

## 🤝 Contributing
Pull requests are welcome! If you'd like to improve the API or add features,  
fork the repo and submit a PR.

---

## 🌟 Acknowledgements
- [FastAPI](https://fastapi.tiangolo.com/)
- [OpenAI](https://platform.openai.com/)
- [Google Gemini](https://ai.google.dev/)
