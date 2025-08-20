# app/retrieval/vector.py
from __future__ import annotations
from typing import List, Optional
import numpy as np
from app.embed.model import Embedder
from app.retrieval.store import connect, load_embeddings, fetch_chunk_texts

class VectorSearcher:
    """
    In-memory cosine similarity search over SQLite-stored embeddings.
    Good up to ~100k chunks on a typical dev machine.
    """
    def __init__(self, db_path: str = "rag_local.db", model_name: str = "BAAI/bge-small-en-v1.5"):
        self.db_path = db_path
        self.model_name = model_name
        self.embedder = Embedder(model_name=model_name)
        self._ids: list[str] = []
        self._mat: np.ndarray = np.zeros((0,1), dtype=np.float32)
        self.reload()  # initial load

    def reload(self):
        conn = connect(self.db_path)
        self._ids, self._mat = load_embeddings(conn, self.model_name)
        conn.close()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self._mat.shape[0] == 0:
            return []

        # encode query (already L2-normalized by our Embedder)
        q = self.embedder.encode([query])[0]          # (dim,)
        # cosine since both q and mat rows are normalized → dot product
        scores = self._mat @ q                        # (N,)
        # top-k indices (descending)
        k = min(top_k, scores.shape[0])
        idx = np.argpartition(scores, -k)[-k:]
        idx = idx[np.argsort(scores[idx])[::-1]]

        # hydrate chunk texts
        conn = connect(self.db_path)
        hit_ids = [self._ids[i] for i in idx.tolist()]
        meta_map = fetch_chunk_texts(conn, hit_ids)
        conn.close()

        results = []
        for i in idx.tolist():
            cid = self._ids[i]
            m = meta_map.get(cid, {})
            results.append({
                "chunk_id": cid,
                "score": float(scores[i]),
                "section": m.get("section"),
                "source": m.get("source"),
                "path": m.get("path"),
                "text": (m.get("text") or "")[:400]  # return a snippet; keep payload small
            })
        return results
