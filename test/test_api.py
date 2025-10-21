from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_search_endpoint(tmp_db, monkeypatch):
    monkeypatch.setenv("VECTOR_DB", "sqlite")
    client.app.dependency_overrides = {}
    r = client.post("/search", json={"query": "refund policy", "top_k": 1})
    assert r.status_code == 200
    assert "hits" in r.json()
