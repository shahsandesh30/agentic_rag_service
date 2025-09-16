import os
from app.retrieval.vector import VectorSearcher

def test_sqlite_search(tmp_db, monkeypatch):
    monkeypatch.setenv("VECTOR_DB", "sqlite")  # or sqlite, qdrant
    vs = VectorSearcher(db_path=tmp_db, model_name="BAAI/bge-small-en-v1.5")
    hits = vs.search("refund policy", top_k=1)

    # ✅ Check we got results
    assert hits, "Expected at least one hit"

    # ✅ Check semantic correctness instead of chunk_id
    text = hits[0]["text"].lower()
    assert "refund" in text
