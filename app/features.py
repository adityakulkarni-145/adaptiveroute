"""Small-model-only features for the learned router."""
import re

import numpy as np

NEGATIONS = {
    "not", "no", "never", "nothing", "none", "nobody", "neither", "nor", "nowhere",
    "cannot", "without", "hardly", "barely", "n't",
}
CONTRAST = {"but", "however", "although"}
FEATURE_NAMES = [
    "calibrated_confidence",
    "raw_margin",
    "char_length",
    "token_count",
    "negation_count",
    "has_contrast",
]

_TOKEN_RE = re.compile(r"n't|[a-z]+(?:'[a-z]+)?")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def router_features(text: str, p_pos_cal: float, p_pos_raw: float) -> list[float]:
    """Calibrated confidence, raw (uncalibrated) probability margin, and surface cues."""
    tokens = tokenize(text)
    neg = sum(1 for t in tokens if t in NEGATIONS or t.endswith("n't"))
    return [
        max(p_pos_cal, 1.0 - p_pos_cal),
        abs(2.0 * p_pos_raw - 1.0),
        float(len(text)),
        float(len(tokens)),
        float(neg),
        float(any(t in CONTRAST for t in tokens)),
    ]


def feature_matrix(texts, p_pos_cal, p_pos_raw) -> np.ndarray:
    return np.array(
        [router_features(t, c, r) for t, c, r in zip(texts, p_pos_cal, p_pos_raw)],
        dtype=float,
    )
