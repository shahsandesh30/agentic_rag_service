# app/vector_store/faiss_store.py
from __future__ import annotations

import json
import os

import faiss
import numpy as np

from .base import BaseVectorStore


class FaissStore(BaseVectorStore):
    name = "faiss"

    def __init__(
        self,
        dim: int,
        path: str = "faiss_index",
        metric: str = "cosine",
        hnsw_m: int = 32,
        ef_search: int = 64,
    ):
        self.dim = dim
        self.metric = metric
        self.path = path
        os.makedirs(self.path, exist_ok=True)
        self.index_path = os.path.join(self.path, "index.faiss")
        self.ids_path = os.path.join(self.path, "ids.json")

        if metric == "cosine":
            # cosine = inner product over normalized vectors
            self.index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        else:
            self.index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_L2)
        self.index.hnsw.efSearch = ef_search

        self._ids: list[str] = []
        self._id2row = {}

        if os.path.exists(self.index_path) and os.path.exists(self.ids_path):
            self._load()

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.ids_path, "w", encoding="utf-8") as f:
            json.dump(self._ids, f)

    def _load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.ids_path, encoding="utf-8") as f:
            self._ids = json.load(f)
        self._id2row = {cid: i for i, cid in enumerate(self._ids)}

    def _ensure_norm(self, v: np.ndarray) -> np.ndarray:
        if self.metric != "cosine":
            return v.astype(np.float32, copy=False)
        # L2 normalize rows
        norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
        return (v / norm).astype(np.float32, copy=False)

    def build(
        self, ids: list[str], vectors: np.ndarray, payloads: list[dict] | None = None
    ) -> None:
        self._ids = list(ids)
        self._id2row = {cid: i for i, cid in enumerate(self._ids)}
        self.index.reset()
        vecs = self._ensure_norm(vectors)
        self.index.add(vecs)
        self._save()

    def upsert(
        self, ids: list[str], vectors: np.ndarray, payloads: list[dict] | None = None
    ) -> None:
        # simple: rebuild (FAISS HNSW doesn't support delete by id easily)
        self.build(ids, vectors, payloads)

    def search(self, query_vec: np.ndarray, top_k: int = 40) -> list[tuple[str, float]]:
        q = query_vec.astype(np.float32, copy=False).reshape(1, -1)
        if self.metric == "cosine":
            q = self._ensure_norm(q)
        D, I = self.index.search(q, top_k)
        idxs = I[0].tolist()
        scores = D[0].tolist()
        out = []
        for i, s in zip(idxs, scores, strict=False):
            if 0 <= i < len(self._ids):
                out.append((self._ids[i], float(s)))
        return out

    def get(self, ids: list[str]) -> np.ndarray:
        # We don't store original vectors internally → caller should still read from SQLite.
        # Return empty to signal: fetch from DB
        return np.zeros((0, self.dim), dtype=np.float32)
