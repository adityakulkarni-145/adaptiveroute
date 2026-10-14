"""SQLite request log. Stores text length only, never the text."""
import os
import sqlite3
import threading
import time

import numpy as np

DB_PATH = os.environ.get("AR_DB_PATH", "/tmp/adaptiveroute.db")
_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db() -> None:
    with _lock, _connect() as con:
        con.execute(
            """CREATE TABLE IF NOT EXISTS requests (
                ts REAL, router TEXT, model TEXT, escalated INTEGER,
                text_length INTEGER, latency_ms REAL)"""
        )


def log_request(router: str, model: str, escalated: bool, text_length: int, latency_ms: float) -> None:
    with _lock, _connect() as con:
        con.execute(
            "INSERT INTO requests VALUES (?, ?, ?, ?, ?, ?)",
            (time.time(), router, model, int(escalated), text_length, latency_ms),
        )


def _summary(rows: list[tuple[int, float]]) -> dict:
    if not rows:
        return {"requests": 0, "escalation_rate": None, "p50_ms": None, "p95_ms": None}
    esc = np.array([r[0] for r in rows], dtype=float)
    lat = np.array([r[1] for r in rows], dtype=float)
    return {
        "requests": len(rows),
        "escalation_rate": round(float(esc.mean()), 4),
        "p50_ms": round(float(np.percentile(lat, 50)), 2),
        "p95_ms": round(float(np.percentile(lat, 95)), 2),
    }


def metrics() -> dict:
    with _lock, _connect() as con:
        rows = con.execute("SELECT router, escalated, latency_ms FROM requests").fetchall()
    out = _summary([(e, l) for _, e, l in rows])
    out["by_router"] = {
        name: _summary([(e, l) for r, e, l in rows if r == name]) for name in ("fixed", "learned")
    }
    return out
