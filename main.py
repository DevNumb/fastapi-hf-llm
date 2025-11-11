from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import requests
import os
import time
import uuid
from typing import Optional
import json

# === Environment setup ===
HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_NAME = os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1")
API_URL = "https://router.huggingface.co/v1/chat/completions"



HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}

# === Simple cache ===
class TTLCache:
    def __init__(self, ttl_seconds=600):
        self.ttl = ttl_seconds
        self.store = {}

    def get(self, key):
        v = self.store.get(key)
        if not v:
            return None
        ts, val = v
        if time.time() - ts > self.ttl:
            del self.store[key]
            return None
        return val

    def set(self, key, val):
        self.store[key] = (time.time(), val)

cache = TTLCache()

# === Rate limiter ===
class RateLimiter:
    def __init__(self, max_requests=20, window_seconds=60):
        self.max_requests = max_requests
        self.window = window_seconds
        self.clients = {}

    def allow(self, ip):
        now = time.time()
        timestamps = [t for t in self.clients.get(ip, []) if now - t < self.window]
        if len(timestamps) >= self.max_requests:
            self.clients[ip] = timestamps
            return False
        timestamps.append(now)
        self.clients[ip] = timestamps
        return True

rate_limiter = RateLimiter()

# === Hugging Face query ===
def query_hf(prompt: str, model: Optional[str] = None):
    model = model or MODEL_NAME
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=30)
    if r.status_code != 200:
        raise Exception(f"HF error {r.status_code}: {r.text}")
    data = r.json()
    return data["choices"][0]["message"]["content"]

# === FastAPI App ===
app = FastAPI(title="StudyBuddy AI", description="Free AI Assistant powered by Hugging Face")

# Create templates directory if it doesn't exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Store chat sessions (in production, use Redis or database)
chat_sessions = {}

# === Routes ===
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/health")
def health():
    return {"status": "healthy", "model": MODEL_NAME}

@app.post("/api/generate")
async def generate(request: Request):
    body = await request.json()
    prompt = body.get("prompt")
    session_id = body.get("session_id", str(uuid.uuid4()))
    
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing 'prompt'.")

    ip = request.client.host
    if not rate_limiter.allow(ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")

    cache_key = f"{MODEL_NAME}::{prompt.strip()}"
    cached = cache.get(cache_key)
    
    if cached:
        return {
            "cached": True, 
            "response": cached,
            "session_id": session_id
        }

    try:
        response = query_hf(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    cache.set(cache_key, response)
    
    # Store in session
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    
    chat_sessions[session_id].append({
        "prompt": prompt,
        "response": response,
        "timestamp": time.time()
    })
    
    return {
        "cached": False, 
        "response": response,
        "session_id": session_id
    }

@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    return chat_sessions.get(session_id, [])

# === Run FastAPI ===
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

