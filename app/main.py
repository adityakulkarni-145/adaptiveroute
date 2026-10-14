"""FastAPI service: small model first, escalate uncertain requests to the large model."""
import time
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import db
from app.models import ARTIFACTS, ROOT, LargeModel, SmallModel
from app.router import DEFAULT_THRESHOLD, FixedRouter, LearnedRouter

STATIC = ROOT / "app" / "static"
RESULTS = ARTIFACTS / "results.json"
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    small, large = SmallModel(), LargeModel()

    @lru_cache(maxsize=2048)
    def small_predict(text: str):
        return small.predict(text)

    @lru_cache(maxsize=2048)
    def large_predict(text: str):
        return large.predict(text)

    state.update(
        small=small_predict,
        large=large_predict,
        fixed=FixedRouter(),
        learned=LearnedRouter(),
    )
    db.init_db()
    yield
    state.clear()


app = FastAPI(
    title="AdaptiveRoute",
    description="Confidence-routed sentiment inference: a cheap model first, a larger model when uncertain.",
    version="1.0.0",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)
    router: Literal["fixed", "learned"] = "fixed"
    threshold: float = Field(DEFAULT_THRESHOLD, ge=0.5, le=1.0, description="Fixed-router confidence threshold")
    cutoff: float | None = Field(None, ge=0.0, le=1.0, description="Learned-router cutoff (default: tuned value)")


class PredictResponse(BaseModel):
    label: Literal["positive", "negative"]
    confidence: float
    model: Literal["small", "large"]
    escalated: bool
    small_label: Literal["positive", "negative"]
    small_confidence: float
    router: Literal["fixed", "learned"]
    threshold: float
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": bool(state)}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be blank")
    start = time.perf_counter()
    small, p_raw = state["small"](req.text)
    if req.router == "fixed":
        threshold = req.threshold
        escalate = state["fixed"].should_escalate(small.confidence, threshold)
    else:
        router = state["learned"]
        threshold = req.cutoff if req.cutoff is not None else router.default
        escalate = router.should_escalate(router.score(req.text, small.p_pos, p_raw), threshold)
    final = state["large"](req.text) if escalate else small
    latency_ms = (time.perf_counter() - start) * 1000
    model = "large" if escalate else "small"
    db.log_request(req.router, model, escalate, len(req.text), latency_ms)
    return PredictResponse(
        label=final.label,
        confidence=round(final.confidence, 4),
        model=model,
        escalated=escalate,
        small_label=small.label,
        small_confidence=round(small.confidence, 4),
        router=req.router,
        threshold=threshold,
        latency_ms=round(latency_ms, 2),
    )


@app.get("/metrics")
def metrics():
    return db.metrics()


@app.get("/results")
def results():
    if not RESULTS.exists():
        raise HTTPException(status_code=404, detail="results.json not found; run scripts/evaluate.py")
    return FileResponse(RESULTS, media_type="application/json")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")
