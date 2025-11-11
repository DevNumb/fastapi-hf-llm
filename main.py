from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
import requests
import gradio as gr
import os
import time
from typing import Optional

# === Environment setup ===
HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_NAME = os.environ.get("MODEL_NAME", "mistralai/Mistral-7B-Instruct-v0.1")
API_URL = "https://router.huggingface.co/v1/chat/completions"

if not HF_TOKEN:
    print("⚠️ Warning: HF_TOKEN not set — API calls will fail.")

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

# === FastAPI ===
app = FastAPI()

@app.get("/")
def home():
    return {"status": "ok", "note": "Go to /ui for the AI chat interface."}

@app.post("/generate")
async def generate(request: Request):
    body = await request.json()
    prompt = body.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing 'prompt'.")

    ip = request.client.host
    if not rate_limiter.allow(ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")

    cache_key = f"{MODEL_NAME}::{prompt.strip()}"
    cached = cache.get(cache_key)
    if cached:
        return {"cached": True, "response": cached}

    try:
        response = query_hf(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    cache.set(cache_key, response)
    return {"cached": False, "response": response}

# === Gradio UI ===
def gradio_interface(prompt):
    try:
        r = requests.post("http://127.0.0.1:8000/generate", json={"prompt": prompt})
        if r.status_code == 200:
            return r.json()["response"]
        else:
            return f"Error {r.status_code}: {r.text}"
    except Exception as e:
        return f"Request failed: {e}"

demo = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Textbox(label="Ask anything", lines=4),
    outputs="text",
    title="🧠 StudyBuddy — Free Hugging Face AI",
    description="Chat with a free open-source model hosted by Hugging Face.",
)

@app.get("/ui", response_class=HTMLResponse)
def ui():
    return demo.launch(share=False, inline=True, inbrowser=False, prevent_thread_lock=True)

# === Run FastAPI (only) ===
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)

