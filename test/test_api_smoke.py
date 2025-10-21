# tests/test_api_smoke.py


def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert "ok" in body and body["ok"] is True
    # "env" may exist if you added it; we don't require it.


def test_chat_returns_stub(client):
    r = client.post("/chat", json={"prompt": "hello there"})
    assert r.status_code == 200
    data = r.json()
    assert "output" in data
    assert data["output"].startswith("[stubbed LLM]")


def test_search_returns_hits(client):
    r = client.post("/search", json={"query": "test", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert "hits" in body
    assert isinstance(body["hits"], list)
    assert len(body["hits"]) >= 1
    hit = body["hits"][0]
    for key in ["chunk_id", "text", "section", "source", "score", "rank"]:
        assert key in hit


def test_ask_returns_answer_string(client):
    r = client.post("/ask", json={"session_id": "s1", "question": "What is this?", "top_k_ctx": 4})
    assert r.status_code == 200
    # Your /ask returns a plain string (the answer), not JSON object
    assert isinstance(r.json(), str)
    assert r.json() == "stubbed answer"


def test_validation_error_422(client):
    # Missing required field "prompt" in /chat should trigger 422
    r = client.post("/chat", json={})
    assert r.status_code == 422
    body = r.json()
    # Depending on your error handler, "error" may be present; at minimum FastAPI returns 'detail'
    assert "detail" in body
