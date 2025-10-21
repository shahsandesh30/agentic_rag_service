def test_422_shape(client):
    # Missing prompt on /chat triggers validation error
    r = client.post("/chat", json={})
    assert r.status_code == 422
    payload = r.json()
    # FastAPI default has 'detail'; our custom handler (if you added it) has 'error' etc.
    # Accept either shape to avoid brittleness across setups:
    assert "detail" in payload or "error" in payload
