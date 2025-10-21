from app.retrieval.store import connect, fetch_chunk_texts, load_embeddings


def test_embeddings_roundtrip(tmp_db):
    conn = connect(tmp_db)
    ids, mat = load_embeddings(conn, "BAAI/bge-small-en-v1.5")
    assert ids == ["chunk1"]
    assert mat.shape == (1, 384)
    chunks = fetch_chunk_texts(conn, ["chunk1"])
    assert "refund" in chunks["chunk1"]["text"].lower()
