from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import gradio as gr
import requests
import os
import threading
import os
HF_TOKEN = os.environ.get("HF_TOKEN")
API_URL = "https://router.huggingface.co/v1/chat/completions"
PORT = int(os.getenv("PORT", "7860"))  # gradio port (unused in Render; example only)

if not HF_TOKEN:
    print("Warning: HF_TOKEN is not set. App will fail trying to use HF API unless set in env.")

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}

# --- Simple in-memory TTL cache to reduce duplicate calls ---
class TTLCache:
    def __init__(self, ttl_seconds=60 * 60):
        self.ttl = ttl_seconds
        self.store = {}  # prompt -> (timestamp, response)

    def get(self, key):
        v = self.store.get(key)
        if not v:
            return None
        ts, val = v
        if time.time() - ts > self.ttl:
            del self.store[key]
            return None
        return val

    def set(self, key, value):
        self.store[key] = (time.time(), value)

cache = TTLCache(ttl_seconds=60 * 10)  # cache identical prompts for 10 minutes

# --- Simple per-IP rate limiter ---
class RateLimiter:
    def __init__(self, max_requests=30, window_seconds=60):
        self.max_requests = max_requests
        self.window = window_seconds
        self.clients = {}  # ip -> [timestamps]

    def allow(self, ip):
        now = time.time()
        arr = self.clients.get(ip, [])
        # drop old
        arr = [t for t in arr if now - t < self.window]
        if len(arr) >= self.max_requests:
            self.clients[ip] = arr
            return False
        arr.append(now)
        self.clients[ip] = arr
        return True

rate_limiter = RateLimiter(max_requests=int(os.getenv("RATE_MAX", "20")), window_seconds=int(os.getenv("RATE_WINDOW", "60")))

# --- Query HF chat completions router ---
def query_hf_chat(prompt: str, model: Optional[str] = None, timeout: int = 30):
    model = model or MODEL_NAME
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        # optionally add more params like temperature, max_new_tokens, etc.
        # "temperature": 0.2,
        # "max_new_tokens": 512,
    }
    r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=timeout)
    if r.status_code != 200:
        # bubble up helpful error for logs; return structured error for UI
        raise Exception(f"Hugging Face API error {r.status_code}: {r.text}")
    data = r.json()
    # expected structure: data["choices"][0]["message"]["content"]
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return f"Unexpected HF response: {data}"

# --- FastAPI app ---
app = FastAPI()

@app.get("/")
def root():
    return {"ok": True, "note": "StudyBuddy running. Use /generate POST or /ui for Gradio."}

@app.post("/generate")
async def generate(request: Request):
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    prompt = body.get("prompt") or body.get("text") or ""
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing 'prompt' in JSON body.")
    # rate limit by client IP
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.allow(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")

    # caching
    cache_key = f"{MODEL_NAME}::" + prompt.strip()
    cached = cache.get(cache_key)
    if cached:
        return JSONResponse({"cached": True, "response": cached})

    # call HF
    try:
        resp = query_hf_chat(prompt, MODEL_NAME)
    except Exception as e:
        # return helpful error but avoid leaking token
        return JSONResponse({"error": str(e)}, status_code=500)

    cache.set(cache_key, resp)
    return {"cached": False, "response": resp}

# --- Gradio frontend ---
def gr_submit(prompt):
    # simple local call to FastAPI generate endpoint (works both locally and deployed)
    try:
        r = requests.post(f"http://127.0.0.1:{PORT}/generate", json={"prompt": prompt}, timeout=20)
        if r.status_code == 200:
            j = r.json()
            return j.get("response") or j
        else:
            return f"Error {r.status_code}: {r.text}"
    except Exception as e:
        # fallback: call API directly (useful if Gradio runs outside FastAPI in same process)
        try:
            return query_hf_chat(prompt, MODEL_NAME)
        except Exception as e2:
            return f"Both local and HF calls failed: {e} / {e2}"

def start_gradio():
    demo = gr.Interface(
        fn=gr_submit,
        inputs=gr.Textbox(lines=4, placeholder="Ask StudyBuddy..."),
        outputs="text",
        title="StudyBuddy — lightweight AI assistant",
        description="Summaries, answers, quizzes. Uses a configurable HF model. Cache + rate-limit enabled to protect free quota.",
        allow_flagging="never"
    )
    demo.launch(server_name="0.0.0.0", server_port=PORT, share=False)

# start gradio in background (for local dev). For Render/production you may run only FastAPI and not start Gradio.
if __name__ == "__main__":
    # start Gradio thread for local dev convenience
    t = threading.Thread(target=start_gradio, daemon=True)
    t.start()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
