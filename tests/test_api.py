import pytest
from fastapi.testclient import TestClient

from app.main import app

KEYS = {
    "label", "confidence", "model", "escalated", "small_label",
    "small_confidence", "router", "threshold", "latency_ms",
}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from app import db

    db.DB_PATH = str(tmp_path_factory.mktemp("db") / "test.db")
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.parametrize("router", ["fixed", "learned"])
def test_predict_valid(client, router):
    r = client.post("/predict", json={"text": "A warm, funny and beautifully acted film.", "router": router})
    assert r.status_code == 200
    body = r.json()
    assert KEYS <= body.keys()
    assert body["router"] == router
    assert body["label"] in {"positive", "negative"}
    assert body["model"] in {"small", "large"}
    assert body["escalated"] == (body["model"] == "large")
    assert 0.5 <= body["confidence"] <= 1.0


def test_threshold_one_always_escalates(client):
    r = client.post("/predict", json={"text": "fine.", "router": "fixed", "threshold": 1.0})
    assert r.json()["escalated"] is True


@pytest.mark.parametrize(
    "payload",
    [
        {"text": ""},
        {"text": "   "},
        {"text": "ok", "threshold": 0.3},
        {"text": "ok", "threshold": 1.5},
        {"text": "x" * 1001},
    ],
)
def test_validation_422(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_metrics_shape(client):
    client.post("/predict", json={"text": "terrible plot", "router": "learned"})
    m = client.get("/metrics").json()
    for block in [m, m["by_router"]["fixed"], m["by_router"]["learned"]]:
        assert {"requests", "escalation_rate", "p50_ms", "p95_ms"} <= block.keys()
    assert m["requests"] >= 1
