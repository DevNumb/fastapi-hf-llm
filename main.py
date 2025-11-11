from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import gradio as gr
import requests
import os
import threading
import os
HF_TOKEN = os.environ.get("HF_TOKEN")

# --- Hugging Face API config ---
API_URL = "https://router.huggingface.co/v1/chat/completions"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# --- FastAPI app ---
app = FastAPI()

@app.get("/")
def home():
    return {"message": "Welcome to FastAPI + Gradio + Hugging Face (DeepSeek-R1)!"}

# --- Function to query the Hugging Face chat API ---
def query_huggingface(message: str):
    payload = {
        "model": "deepseek-ai/DeepSeek-R1:novita",
        "messages": [
            {"role": "user", "content": message}
        ]
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    data = response.json()
    try:
        return data["choices"][0]["message"]["content"]
    except KeyError:
        return f"Unexpected response format: {data}"

# --- FastAPI endpoint ---
@app.post("/generate")
def generate_text(prompt: str):
    result = query_huggingface(prompt)
    return {"prompt": prompt, "response": result}

# --- Gradio frontend function ---
def gradio_generate(prompt):
    try:
        return query_huggingface(prompt)
    except Exception as e:
        return f"Error: {e}"

# --- Launch Gradio in background thread ---
def start_gradio():
    demo = gr.Interface(
        fn=gradio_generate,
        inputs="text",
        outputs="text",
        title="🧠 FastAPI + DeepSeek-R1 (via Hugging Face)",
        description="Ask anything and get an AI response powered by DeepSeek-R1:novita!"
    )
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

threading.Thread(target=start_gradio, daemon=True).start()

# --- Serve Gradio via HTML iframe ---
@app.get("/ui", response_class=HTMLResponse)
def gradio_ui():
    return """
    <html>
        <head><title>DeepSeek-R1 LLM Interface</title></head>
        <body style="text-align:center; font-family:sans-serif;">
            <h2>🤖 FastAPI + DeepSeek-R1 (Hugging Face Router)</h2>
            <iframe src="http://localhost:7860" width="100%" height="700" frameborder="0"></iframe>
            <p>⚙️ If you're deploying to Render or similar, replace 'localhost' with your public Render hostname.</p>
        </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)


