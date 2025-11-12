import os
import gradio as gr
import requests

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set on Render.")

API_URL = "https://router.huggingface.co/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}


def chat_with_ai(message, history):
    """Chat function for Hugging Face model"""
    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.1",  # Free-tier model
        "messages": [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": message},
        ],
    }

    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "⚠️ No response.")
    except Exception as e:
        return f"❌ Error: {e}"


app = gr.ChatInterface(
    fn=chat_with_ai,
    title="💬 Professional AI Chatbot",
    description="Chat with a free Hugging Face model hosted on Render.",
    theme="soft",
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # <- required for Render
    app.launch(server_name="0.0.0.0", server_port=port)





