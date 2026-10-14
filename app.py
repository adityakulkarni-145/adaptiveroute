"""Hugging Face Spaces (Gradio SDK) entry point: serves the FastAPI app and its web UI on port 7860."""
import uvicorn

from app.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
