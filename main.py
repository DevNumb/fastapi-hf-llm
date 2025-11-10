from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import gradio as gr
import requests
import os
import threading
import requests

app = FastAPI()
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


API_URL = "https://router.huggingface.co/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
}

# --- FastAPI endpoint (for API use) ---
@app.get("/")
def home():
    return {"message": "Welcome to my FastAPI + Gradio LLM app!"}

@app.post("/generate")
def generate_text(prompt: str):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt}

    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)
    result = response.json()
    return {"prompt": prompt, "generated_text": result[0]["generated_text"]}

# --- Gradio frontend function ---
def gradio_generate(prompt):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        return f"Error: {response.text}"
    result = response.json()
    return result[0]["generated_text"]

# --- Launch Gradio Interface in background thread ---
def start_gradio():
    demo = gr.Interface(
        fn=gradio_generate,
        inputs="text",
        outputs="text",
        title="FastAPI + Hugging Face LLM",
        description="Type a prompt below and let the AI complete it!"
    )
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

threading.Thread(target=start_gradio).start()

# --- Serve Gradio via HTML page ---
@app.get("/ui", response_class=HTMLResponse)
def gradio_ui():
    return """
    <html>
        <head><title>LLM Web UI</title></head>
        <body style="text-align:center; font-family:sans-serif">
            <h2>🤖 FastAPI + Gradio LLM Interface</h2>
            <iframe src="http://localhost:7860" width="100%" height="700" frameborder="0"></iframe>
            <p>If you're online (Render), replace 'localhost' with your Render hostname in this iframe URL.</p>
        </body>
    </html>

    """


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
