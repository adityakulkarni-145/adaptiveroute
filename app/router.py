"""Fixed-threshold and learned escalation routers."""
from pathlib import Path

import joblib
import numpy as np

from app.features import router_features
from app.models import ARTIFACTS

DEFAULT_THRESHOLD = 0.85


class FixedRouter:
    name = "fixed"

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.default = threshold

    def should_escalate(self, small_confidence: float, threshold: float) -> bool:
        return small_confidence < threshold


class LearnedRouter:
    """Predicts P(large right and small wrong) from small-model-only features."""

    name = "learned"

    def __init__(self, path: Path = ARTIFACTS / "router.joblib"):
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.default = float(bundle["cutoff"])

    def score(self, text: str, p_pos_cal: float, p_pos_raw: float) -> float:
        x = np.array([router_features(text, p_pos_cal, p_pos_raw)])
        return float(self.model.predict_proba(x)[0, 1])

    def should_escalate(self, score: float, cutoff: float) -> bool:
        return score > cutoff
