"""Shared data loading and calibration metrics for the offline scripts."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"
CACHE = ROOT / "cache"
SEED = 42


def load_sst2():
    """GLUE SST-2: (train texts, train labels), validation first half, validation second half."""
    from datasets import load_dataset

    ds = load_dataset("nyu-mll/glue", "sst2")
    val = ds["validation"]
    half = len(val) // 2
    split = lambda d: (list(d["sentence"]), np.array(d["label"]))  # noqa: E731
    return split(ds["train"]), split(val.select(range(half))), split(val.select(range(half, len(val))))


def load_imdb_ood(n: int = 2000, max_chars: int = 1000):
    """Fixed random sample (seed 42) of IMDB test reviews, truncated to the first 1000 characters."""
    from datasets import load_dataset

    test = load_dataset("stanfordnlp/imdb", split="test")
    idx = np.sort(np.random.default_rng(SEED).choice(len(test), size=n, replace=False))
    sub = test.select(idx)
    return [t[:max_chars] for t in sub["text"]], np.array(sub["label"])


def ece(p_pos: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error on the predicted-class confidence."""
    conf = np.maximum(p_pos, 1 - p_pos)
    correct = (p_pos >= 0.5).astype(int) == y
    edges = np.linspace(0.5, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def reliability_points(p_pos: np.ndarray, y: np.ndarray, bins: int = 10):
    conf = np.maximum(p_pos, 1 - p_pos)
    correct = (p_pos >= 0.5).astype(int) == y
    edges = np.linspace(0.5, 1.0, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    xs, ys, ns = [], [], []
    for b in range(bins):
        m = idx == b
        if m.any():
            xs.append(conf[m].mean())
            ys.append(correct[m].mean())
            ns.append(int(m.sum()))
    return np.array(xs), np.array(ys), np.array(ns)
