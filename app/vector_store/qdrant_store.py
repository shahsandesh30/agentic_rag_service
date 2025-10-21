# app/vector_store/qdrant_store.py
from __future__ import annotations

import os

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams

from .base import BaseVectorStore


class QdrantStore(BaseVectorStore):
    name = "qdrant"

    def __init__(
        self,
        dim: int,
        collection: str = "chunks",
        host: str | None = None,
        port: int | None = None,
        url: str | None = None,
    ):
        self.dim = dim
        self.collection = collection
        # Prefer URL if provided, else host:port
        url = url or os.getenv("QDRANT_URL")
        host = host or os.getenv("QDRANT_HOST", "localhost")
        port = port or int(os.getenv("QDRANT_PORT", "6333"))
        api_key = os.getenv("QDRANT_API_KEY")  # optional for local

        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            self.client = QdrantClient(host=host, port=port, api_key=api_key)

        self._ensure_collection()

    def _ensure_collection(self):
        cols = [c.name for c in self.client.get_collections().collections]
        if self.collection not in cols:
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
            )

    def build(
        self, ids: list[str], vectors: np.ndarray, payloads: list[dict] | None = None
    ) -> None:
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
        )
        self.upsert(ids, vectors, payloads)

    def upsert(
        self, ids: list[str], vectors: np.ndarray, payloads: list[dict] | None = None
    ) -> None:
        payloads = payloads or [{} for _ in ids]
        points = []
        for cid, vec, pl in zip(
            ids, vectors.astype(np.float32, copy=False), payloads, strict=False
        ):
            points.append(PointStruct(id=cid, vector=vec.tolist(), payload=pl))
        # batch insert
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vec: np.ndarray, top_k: int = 40) -> list[tuple[str, float]]:
        res = self.client.search(
            collection_name=self.collection,
            query_vector=query_vec.astype(np.float32, copy=False).tolist(),
            limit=top_k,
        )
        return [(r.id, float(r.score)) for r in res]

    def get(self, ids: list[str]) -> np.ndarray:
        # Qdrant can fetch by ids, but we keep SQLite as single source for vectors.
        return np.zeros((0, self.dim), dtype=np.float32)
