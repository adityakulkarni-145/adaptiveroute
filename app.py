"""Hugging Face Spaces (Gradio SDK) entry point: serves the FastAPI app and its web UI on port 7860."""
try:  # ZeroGPU hardware requires the `spaces` package, imported before torch, and one @spaces.GPU function.
    import spaces

    @spaces.GPU
    def _zerogpu_placeholder():
        """Never called: both models run on CPU. Exists only so a ZeroGPU Space starts."""
except ImportError:
    pass

import uvicorn

from app.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
