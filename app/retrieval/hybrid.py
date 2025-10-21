# app/retrieval/hybrid.py
from __future__ import annotations

import logging
from typing import Any

from app.config import get_settings

log = logging.getLogger(__name__)
_settings = get_settings()


def rrf_merge(
    faiss_hits: list[dict[str, Any]],
    bm25_hits: list[dict[str, Any]],
    k: int = 60,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Reciprocal Rank Fusion.
    Score(doc) = sum( 1 / (k + rank_i) ) over systems.
    Ranks start at 1.
    """
    top_k = int(top_k or _settings.top_k)

    def key(h):  # (chunk_id, source) pair uniquely identifies a chunk
        return (h["chunk_id"], h.get("source"))

    fused: dict[Any, dict[str, Any]] = {}

    def add_list(hits: list[dict[str, Any]]):
        for h in hits:
            kk = key(h)
            if kk not in fused:
                fused[kk] = dict(h)
                fused[kk]["_rrf"] = 0.0
            rank = int(h.get("rank", 999999))
            fused[kk]["_rrf"] += 1.0 / (k + rank)

    add_list(faiss_hits)
    add_list(bm25_hits)

    merged = list(fused.values())
    merged.sort(key=lambda x: x["_rrf"], reverse=True)
    # normalize output: keep original fields + final rank & fused_score
    out: list[dict[str, Any]] = []
    for i, h in enumerate(merged[:top_k], start=1):
        h2 = {k: v for k, v in h.items() if not k.startswith("_")}
        h2["rank"] = i
        h2["fused_score"] = h["_rrf"]
        out.append(h2)
    return out


class HybridSearcher:
    """
    Wraps FAISS and BM25 searchers; merges results via RRF.
    """

    def __init__(self, faiss_searcher, bm25_searcher, top_k_default: int | None = None):
        self.faiss = faiss_searcher
        self.bm25 = bm25_searcher
        self._top_k_default = int(top_k_default or _settings.top_k)

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        k = int(top_k or self._top_k_default)
        faiss_hits = self.faiss.search(query, top_k=k)
        bm25_hits = self.bm25.search(query, top_k=k)
        return rrf_merge(faiss_hits, bm25_hits, k=60, top_k=k)

    def close(self):
        # Close inner resources if available
        if hasattr(self.faiss, "close"):
            self.faiss.close()
        if hasattr(self.bm25, "close"):
            self.bm25.close()
