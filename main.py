import os
import gradio as gr
import requests

HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable not set. Please set it in Render Dashboard.")

API_URL = "https://router.huggingface.co/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}


def chat_with_ai(message, history):
    """Chat with Hugging Face free-tier model"""
    payload = {
        "model": "mistralai/Mistral-7B-Instruct-v0.1",
        "messages": [
            {"role": "system", "content": "You are a friendly and professional assistant."},
            {"role": "user", "content": message},
        ],
    }
    try:
        r = requests.post(API_URL, headers=HEADERS, json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "⚠️ No response from model.")
    except Exception as e:
        return f"❌ Error: {e}"


app = gr.ChatInterface(
    fn=chat_with_ai,
    title="🤖 Professional AI Chatbot",
    description="Built with FastAPI + Gradio + Hugging Face free-tier model.",
    theme="soft",
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))  # Required for Render
    app.launch(server_name="0.0.0.0", server_port=port)


