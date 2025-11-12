import os
import requests
import uvicorn
import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# === CONFIG ===
HF_TOKEN = os.getenv("HF_TOKEN")
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
API_URL = "https://router.huggingface.co/v1/chat/completions"

if not HF_TOKEN:
    raise ValueError("❌ HF_TOKEN not set. Please add it in Render → Environment Variables.")

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}

# === FASTAPI APP ===
app = FastAPI(title="Mistral ChatGPT Clone", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Conversation Store ===
conversations = {}

# === Hugging Face Query ===
def query_hf(messages):
    payload = {"model": MODEL_NAME, "messages": messages}
    response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=40)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise HTTPException(status_code=500, detail=f"Unexpected HF response: {data}")

# === Gradio Chat Function ===
def chat_with_ai(message, history):
    conversation_id = "default"
    if conversation_id not in conversations:
        conversations[conversation_id] = [{"role": "system", "content": "You are a helpful AI assistant."}]
    
    # Add user message
    conversations[conversation_id].append({"role": "user", "content": message})

    # Query model
    try:
        ai_reply = query_hf(conversations[conversation_id])
    except Exception as e:
        return [(message, f"⚠️ Error: {e}")]

    # Add assistant message
    conversations[conversation_id].append({"role": "assistant", "content": ai_reply})
    conversations[conversation_id] = conversations[conversation_id][-12:]

    # Format for Gradio chat history
    history.append((message, ai_reply))
    return history

# === Gradio UI ===
chat_ui = gr.ChatInterface(
    fn=chat_with_ai,
    title="💬 Mistral ChatGPT Clone",
    description="A fast, open-source ChatGPT-style AI assistant powered by Hugging Face’s Mistral 7B.",
    theme="soft",  # Elegant built-in Gradio theme
    examples=["Explain quantum computing in simple terms.", "Write a short poem about AI."],
    retry_btn="🔁 Regenerate",
    undo_btn="↩ Undo last message",
)

# === FastAPI Routes ===
@app.get("/")
def home():
    return {
        "message": "✅ Mistral ChatGPT Clone running!",
        "ui": "/ui",
        "api": "/chat",
        "model": MODEL_NAME,
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/ui")
def launch_ui():
    # Launch Gradio in inline mode for Render
    return chat_ui.launch(share=False, inline=True, inbrowser=False, prevent_thread_lock=True)

# Required by Render: listen on PORT
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)




