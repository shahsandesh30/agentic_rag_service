import os
from app.retrieval.vector import VectorSearcher

def test_sqlite_search(tmp_db, monkeypatch):
    monkeypatch.setenv("VECTOR_DB", "faiss")  # or sqlite, qdrant
    print("Temporary DB path:", tmp_db)
    vs = VectorSearcher(db_path=tmp_db, model_name="BAAI/bge-small-en-v1.5")
    print(f"Using VECTOR_DB={os.getenv('VECTOR_DB')}")
    print("vsssssss--------", vs)
    hits = vs.search("refund policy", top_k=1)

    # ✅ Check we got results
    assert hits, "Expected at least one hit"

    # ✅ Check semantic correctness instead of chunk_id
    text = hits[0]["text"].lower()
    assert "refund" in text
