import os
import gradio as gr
import requests

# Get Hugging Face API key from Render environment variables
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
API_URL = "https://router.huggingface.co/v1/chat/completions"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}


def chat_with_ai(message, history):
    """Simple AI chat using Hugging Face Inference API"""
    payload = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a professional AI assistant."},
            {"role": "user", "content": message},
        ],
    }

    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    data = response.json()
    return data.get("choices", [{}])[0].get("message", {}).get("content", "⚠️ Error: No reply")


# Gradio chat interface
app = gr.ChatInterface(
    fn=chat_with_ai,
    title="💬 Professional AI Chat",
    description="A clean ChatGPT-like interface using Hugging Face open models.",
    theme="soft",
)

# Render automatically sets PORT, so we use it
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.launch(server_name="0.0.0.0", server_port=port)






