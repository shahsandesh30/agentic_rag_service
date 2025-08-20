# app/retrieval/rerank.py
from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import CrossEncoder

def _truncate_chars(t: str, max_chars: int) -> str:
    if t is None:
        return ""
    return t if len(t) <= max_chars else t[:max_chars]

class Reranker:
    """
    Cross-encoder reranker (query, passage) -> relevance score.
    Default: BAAI/bge-reranker-base (English). For multilingual, try: BAAI/bge-reranker-v2-m3.
    """
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        max_passage_chars: int = 1200,
        batch_size: int = 16,
        device: str | None = None,   # e.g., "cuda" to force GPU
    ):
        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device, trust_remote_code=True)
        self.max_passage_chars = max_passage_chars
        self.batch_size = batch_size

    def rerank(self, query: str, hits: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        hits: [{'chunk_id','text',...}]  (text should be full chunk text; we truncate safely)
        returns new list sorted by cross-encoder score (desc), with 'ce_score' added
        """
        if not hits:
            return []

        pairs = [(query, _truncate_chars(h.get("text",""), self.max_passage_chars)) for h in hits]
        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False
        )  # higher is better

        for h, s in zip(hits, scores.tolist()):
            h["ce_score"] = float(s)

        # sort by ce_score desc and cut to top_k
        hits.sort(key=lambda x: x.get("ce_score", 0.0), reverse=True)
        return hits[: min(top_k, len(hits))]
