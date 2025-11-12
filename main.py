import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import requests
import uvicorn

# === CONFIG ===
HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
API_URL = "https://router.huggingface.co/v1/chat/completions"

if not HF_TOKEN:
    raise ValueError("❌ HF_TOKEN not found. Please set it in Render → Environment → Secret Keys")

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json",
}

# === FastAPI Setup ===
app = FastAPI(title="Mistral Chat API", version="1.0.0")

# Enable CORS for any frontend (Gradio, React, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory conversation storage
conversations = {}

# === Helper to query Hugging Face ===
def query_huggingface(messages):
    payload = {
        "model": MODEL_NAME,
        "messages": messages
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=40)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise HTTPException(status_code=500, detail=f"Unexpected response: {data}")

# === ROUTES ===
@app.get("/")
async def root():
    return {"message": "✅ Mistral Chat API running!", "usage": "POST /chat/{conversation_id}?message=Hello"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/chat/{conversation_id}")
async def chat(conversation_id: str, request: Request):
    """
    Send a user message to the model and return the AI's response.
    """
    try:
        body = await request.json()
        message = body.get("message")
        if not message:
            raise HTTPException(status_code=400, detail="Missing 'message' field.")

        # Initialize conversation if new
        if conversation_id not in conversations:
            conversations[conversation_id] = [{"role": "system", "content": "You are a helpful assistant."}]

        # Add user message
        conversations[conversation_id].append({"role": "user", "content": message})

        # Get AI reply
        ai_response = query_huggingface(conversations[conversation_id])

        # Add AI message to conversation
        conversations[conversation_id].append({"role": "assistant", "content": ai_response})

        # Limit conversation history (avoid memory bloat)
        conversations[conversation_id] = conversations[conversation_id][-10:]

        return {
            "conversation_id": conversation_id,
            "user_message": message,
            "bot_response": ai_response,
            "conversation_history": conversations[conversation_id],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Retrieve chat history"""
    return {
        "conversation_id": conversation_id,
        "history": conversations.get(conversation_id, [])
    }

@app.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Clear chat history"""
    if conversation_id in conversations:
        del conversations[conversation_id]
        return {"message": f"Conversation {conversation_id} deleted."}
    raise HTTPException(status_code=404, detail="Conversation not found.")

# === Render Entrypoint ===
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

