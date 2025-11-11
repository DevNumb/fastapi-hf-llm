import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import json

# Get Hugging Face token from environment variables
hf_token = os.environ.get("HF_TOKEN")

if not hf_token:
    raise ValueError("HF_TOKEN environment variable not set. Please set it in Render dashboard.")

API_URL = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-large"
headers = {
    "Authorization": f"Bearer {hf_token}",
    "Content-Type": "application/json"
}

app = FastAPI(title="Chat API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store conversations (in production, use a proper database)
conversations = {}

def query_huggingface(payload):
    """Query Hugging Face API"""
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Hugging Face API error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Chat API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/chat/{conversation_id}")
async def chat(conversation_id: str, message: str):
    """Send a message and get a response"""
    if conversation_id not in conversations:
        conversations[conversation_id] = []
    
    # Add user message to conversation history
    conversations[conversation_id].append(f"User: {message}")
    
    # Prepare payload for DialoGPT
    payload = {
        "inputs": {
            "text": message,
            "past_user_inputs": [],
            "generated_responses": []
        }
    }
    
    try:
        # Get response from Hugging Face
        response = query_huggingface(payload)
        
        # Extract the generated text
        if isinstance(response, list) and len(response) > 0:
            bot_response = response[0].get('generated_text', 'Sorry, I did not understand that.')
        else:
            bot_response = response.get('generated_text', 'Sorry, I did not understand that.')
        
        # Add bot response to conversation history
        conversations[conversation_id].append(f"Bot: {bot_response}")
        
        return {
            "conversation_id": conversation_id,
            "user_message": message,
            "bot_response": bot_response,
            "conversation_history": conversations[conversation_id][-6:]  # Last 3 exchanges
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history"""
    if conversation_id not in conversations:
        return {"conversation_id": conversation_id, "history": []}
    
    return {
        "conversation_id": conversation_id,
        "history": conversations[conversation_id]
    }

@app.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    if conversation_id in conversations:
        del conversations[conversation_id]
        return {"message": f"Conversation {conversation_id} deleted"}
    else:
        raise HTTPException(status_code=404, detail="Conversation not found")

# Required for Render deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)



