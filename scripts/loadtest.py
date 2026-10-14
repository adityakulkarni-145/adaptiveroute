"""Headless locust load test: 20 users for 60 s against a running server.

Usage:  python scripts/loadtest.py --host http://127.0.0.1:8000
Writes results/load.json (p50/p95 latency and requests/sec, overall and per router).
"""
import argparse
import csv
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

from locust import HttpUser, between, task

SENTENCES = [
    "a gorgeous, witty and moving film", "the plot is a tedious mess", "not bad, but not memorable either",
    "one of the best performances of the year", "i wanted to like it but it never comes together",
    "an utterly forgettable sequel", "surprisingly charming and smart", "the dialogue is flat and lifeless",
    "it has its moments, although the ending drags", "a triumph of style over substance",
]
ADJ = ["really", "honestly", "frankly", "overall", "in the end", "to be fair", "simply", "at times"]


class RouteUser(HttpUser):
    wait_time = between(0.1, 0.5)

    def _text(self) -> str:
        # Vary the text so the server's LRU cache does not answer most requests.
        return f"{random.choice(ADJ)}, {random.choice(SENTENCES)} ({random.randint(0, 10**6)})"

    @task
    def fixed(self):
        self.client.post("/predict", json={"text": self._text(), "router": "fixed"}, name="/predict fixed")

    @task
    def learned(self):
        self.client.post("/predict", json={"text": self._text(), "router": "learned"}, name="/predict learned")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://127.0.0.1:8000")
    ap.add_argument("--users", type=int, default=20)
    ap.add_argument("--duration", default="60s")
    args = ap.parse_args()
    out_dir = Path(__file__).resolve().parent.parent / "results"
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "load"
        subprocess.run([sys.executable, "-m", "locust", "-f", __file__, "--headless", "-u", str(args.users),
                        "-r", str(args.users), "-t", args.duration, "--host", args.host, "--csv", str(prefix),
                        "--only-summary"], check=True)
        with open(f"{prefix}_stats.csv") as f:
            rows = list(csv.DictReader(f))
    summary = {
        r["Name"]: {
            "requests": int(r["Request Count"]), "failures": int(r["Failure Count"]),
            "p50_ms": float(r["50%"]), "p95_ms": float(r["95%"]),
            "requests_per_sec": round(float(r["Requests/s"]), 2),
        }
        for r in rows
    }
    report = {"users": args.users, "duration": args.duration, "host": args.host, "stats": summary}
    (out_dir / "load.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
