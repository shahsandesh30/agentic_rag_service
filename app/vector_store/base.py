# app/vector_store/base.py
from __future__ import annotations

import numpy as np


class BaseVectorStore:
    name: str = "base"

    def build(
        self, ids: list[str], vectors: np.ndarray, payloads: list[dict] | None = None
    ) -> None:
        """(Re)build index from scratch."""
        raise NotImplementedError

    def upsert(
        self, ids: list[str], vectors: np.ndarray, payloads: list[dict] | None = None
    ) -> None:
        """Insert or update vectors."""
        raise NotImplementedError

    def search(self, query_vec: np.ndarray, top_k: int = 40) -> list[tuple[str, float]]:
        """Return list of (chunk_id, score) by similarity (higher is better)."""
        raise NotImplementedError

    def get(self, ids: list[str]) -> np.ndarray:
        """Return vectors (rows correspond to ids order)."""
        raise NotImplementedError
