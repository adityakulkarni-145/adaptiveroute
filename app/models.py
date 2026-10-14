"""Small (TF-IDF + calibrated LR) and large (DistilBERT) sentiment models."""
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
LARGE_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
LABELS = ("negative", "positive")


@dataclass
class Prediction:
    label: str
    confidence: float
    p_pos: float


class SmallModel:
    def __init__(self, path: Path = ARTIFACTS / "small_model.joblib"):
        bundle = joblib.load(path)
        self.calibrated = bundle["calibrated"]
        self.base = bundle["base"]

    def probs(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Return (calibrated P(positive), uncalibrated P(positive))."""
        return self.calibrated.predict_proba(texts)[:, 1], self.base.predict_proba(texts)[:, 1]

    def predict(self, text: str) -> tuple[Prediction, float]:
        cal, raw = self.probs([text])
        p = float(cal[0])
        return Prediction(LABELS[int(p >= 0.5)], max(p, 1 - p), p), float(raw[0])


class LargeModel:
    def __init__(self, name: str = LARGE_MODEL_NAME):
        from transformers import pipeline

        self.pipe = pipeline("sentiment-analysis", model=name, device=-1)

    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        out = []
        for scores in self.pipe(texts, top_k=None, truncation=True):
            p = next(s["score"] for s in scores if s["label"].upper() == "POSITIVE")
            out.append(Prediction(LABELS[int(p >= 0.5)], max(p, 1 - p), float(p)))
        return out

    def predict(self, text: str) -> Prediction:
        return self.predict_batch([text])[0]
