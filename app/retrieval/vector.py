# app/retrieval/vector.py
from __future__ import annotations
import os
from typing import List, Optional
import numpy as np
from app.embed.model import Embedder
from app.retrieval.store import connect, load_embeddings, fetch_chunk_texts
from app.retrieval.store import connect as db_connect

# Optional vector backends
_BACKEND = os.getenv("VECTOR_DB", "faiss").lower()

class VectorSearcher:
    """
    Vector candidate provider. If VECTOR_DB=sqlite (default), use local matrix search.
    Else use FAISS or Qdrant ANN and hydrate metadata from DB.
    """
    def __init__(self, db_path: str = "rag_local.db", model_name: str = "BAAI/bge-small-en-v1.5"):
        self.db_path = db_path
        self.model_name = model_name
        self.embedder = Embedder(model_name=model_name)
        self._ids: list[str] = []
        self._mat: np.ndarray = np.zeros((0,1), dtype=np.float32)
        self._id2idx: dict[str,int] = {}
        self._store = None

        if _BACKEND == "sqlite":
            self.reload()
        elif _BACKEND == "faiss":
            from app.vector_store.faiss_store import FaissStore
            # dim needed: load 1 row to discover; or use embedder dimension
            conn = db_connect(self.db_path)
            ids, mat = load_embeddings(conn, self.model_name)
            conn.close()
            dim = mat.shape[1] if mat.size else self.embedder.dim
            print(f"Using FAISS vector store with dim={dim}----------------------")
            self._store = FaissStore(dim=dim, path="faiss_index", metric="cosine")
        elif _BACKEND == "qdrant":
            from app.vector_store.qdrant_store import QdrantStore
            conn = db_connect(self.db_path)
            ids, mat = load_embeddings(conn, self.model_name)
            conn.close()
            dim = mat.shape[1] if mat.size else self.embedder.dim
            self._store = QdrantStore(dim=dim, collection="chunks")
        else:
            raise RuntimeError(f"Unknown VECTOR_DB={_BACKEND}")

    def reload(self):
        conn = connect(self.db_path)
        self._ids, self._mat = load_embeddings(conn, self.model_name)
        conn.close()
        self._id2idx = {cid: i for i, cid in enumerate(self._ids)}

    def get_vectors(self, chunk_ids: list[str]) -> np.ndarray:
        # Always read from SQLite (source of truth)
        if not chunk_ids:
            return np.zeros((0, self._mat.shape[1] if self._mat.size else self.embedder.dim), dtype=np.float32)
        # Pull from embeddings table directly
        conn = connect(self.db_path)
        # SQLite IN clause
        out = []
        for i in range(0, len(chunk_ids), 999):
            batch = chunk_ids[i:i+999]
            q = f"SELECT chunk_id, dim, vec FROM embeddings WHERE model=? AND chunk_id IN ({','.join('?'*len(batch))})"
            rows = conn.execute(q, (self.model_name, *batch)).fetchall()
            m = {r["chunk_id"]: np.frombuffer(r["vec"], dtype=np.float32, count=r["dim"]) for r in rows}
            out.extend(m.get(cid, np.zeros((self.embedder.dim,), dtype=np.float32)) for cid in batch)
        conn.close()
        mat = np.vstack(out).astype(np.float32, copy=False)
        # Normalize to use cosine properly
        norm = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
        return mat / norm

    def _search_sqlite(self, q: np.ndarray, top_k: int) -> list[dict]:
        if self._mat.shape[0] == 0:
            return []
        scores = self._mat @ q
        k = min(top_k, scores.shape[0])
        idx = np.argpartition(scores, -k)[-k:]
        idx = idx[np.argsort(scores[idx])[::-1]]
        conn = connect(self.db_path)
        hit_ids = [self._ids[i] for i in idx.tolist()]
        meta_map = fetch_chunk_texts(conn, hit_ids)
        conn.close()
        return [{
            "chunk_id": self._ids[i],
            "score": float(scores[i]),
            "section": meta_map.get(self._ids[i], {}).get("section"),
            "source":  meta_map.get(self._ids[i], {}).get("source"),
            "path":    meta_map.get(self._ids[i], {}).get("path"),
            "text":   (meta_map.get(self._ids[i], {}).get("text") or "")[:400],
        } for i in idx.tolist()]

    def _search_store(self, q: np.ndarray, top_k: int) -> list[dict]:
        pairs = self._store.search(q, top_k=top_k)
        ids = [cid for cid, _ in pairs]
        conn = connect(self.db_path)
        meta_map = fetch_chunk_texts(conn, ids)
        print("meta_map:--------------", meta_map)
        conn.close()
        return [{
            "chunk_id": cid,
            "score": float(score),
            "section": meta_map.get(cid, {}).get("section"),
            "source":  meta_map.get(cid, {}).get("source"),
            "path":    meta_map.get(cid, {}).get("path"),
            "text":   (meta_map.get(cid, {}).get("text") or "")[:400],
        } for cid, score in pairs]

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        q = self.embedder.encode([query])[0]  # already L2-normalized by our Embedder
        if _BACKEND == "sqlite":
            return self._search_sqlite(q, top_k)
        else:
            return self._search_store(q, top_k)
