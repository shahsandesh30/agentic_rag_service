# app/rerank/reranker.py
from __future__ import annotations

import logging
from typing import Any

from app.config import get_settings

log = logging.getLogger(__name__)
_settings = get_settings()


class Reranker:
    """
    Cross-encoder reranker using sentence-transformers (e.g., BAAI/bge-reranker-base).
    Scores (query, passage) and sorts descending.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or _settings.reranker_model or "BAAI/bge-reranker-base"
        self._model = None
        self._lazy_loaded = False

    def _ensure_loaded(self):
        if self._lazy_loaded:
            return
        from sentence_transformers import CrossEncoder  # heavy import

        self._model = CrossEncoder(self.model_name, trust_remote_code=True)
        self._lazy_loaded = True
        log.info("Reranker loaded (%s)", self.model_name)

    def rerank(
        self, query: str, hits: list[dict[str, Any]], top_k: int | None = None
    ) -> list[dict[str, Any]]:
        if not hits:
            return hits
        self._ensure_loaded()
        k = int(top_k or min(_settings.rerank_k, len(hits)))
        pairs = [(query, h["text"]) for h in hits]
        scores = self._model.predict(pairs)  # higher is better
        # attach and sort
        enriched = []
        for h, s in zip(hits, scores, strict=False):
            h2 = dict(h)
            h2["rerank_score"] = float(s)
            enriched.append(h2)
        enriched.sort(key=lambda x: x["rerank_score"], reverse=True)
        # keep the top_k reranked, but preserve original fields
        out = []
        for i, h in enumerate(enriched[:k], start=1):
            h2 = {k: v for k, v in h.items() if k != "rerank_score"}
            h2["rank"] = i
            out.append(h2)
        return out
