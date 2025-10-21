# app/retrieval/rerank.py
"""
Reranking functionality for search results using cross-encoder models.
"""

from __future__ import annotations

import logging

from sentence_transformers import CrossEncoder

from .interfaces import SearchResult

logger = logging.getLogger(__name__)


def _truncate_chars(text: str | None, max_chars: int) -> str:
    """
    Safely truncate text to maximum character length.

    Args:
        text: Text to truncate (can be None)
        max_chars: Maximum number of characters

    Returns:
        Truncated text string
    """
    if text is None:
        return ""
    return text if len(text) <= max_chars else text[:max_chars]


class Reranker:
    """
    Cross-encoder reranker for improving search result relevance.

    Uses a cross-encoder model to score query-document pairs and rerank results.
    Default model: BAAI/bge-reranker-base (English).
    For multilingual support, try: BAAI/bge-reranker-v2-m3.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        max_passage_chars: int = 1200,
        batch_size: int = 16,
        device: str | None = None,
    ):
        """
        Initialize the reranker.

        Args:
            model_name: Name of the cross-encoder model to use
            max_passage_chars: Maximum characters per passage for reranking
            batch_size: Batch size for model predictions
            device: Device to run model on (e.g., "cuda" for GPU)

        Raises:
            Exception: If model loading fails
        """
        self.model_name = model_name
        self.max_passage_chars = max_passage_chars
        self.batch_size = batch_size
        self.device = device

        try:
            logger.info(f"Loading reranker model: {model_name}")
            self.model = CrossEncoder(model_name, device=device, trust_remote_code=True)
            logger.info(f"Successfully loaded reranker model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load reranker model {model_name}: {e}")
            raise

    def rerank(self, query: str, hits: list[SearchResult], top_k: int = 10) -> list[SearchResult]:
        """
        Rerank search results using cross-encoder scoring.

        Args:
            query: The search query string
            hits: List of search results to rerank
            top_k: Maximum number of results to return after reranking

        Returns:
            New list of results sorted by cross-encoder score (descending)

        Raises:
            Exception: If reranking fails
        """
        if not hits:
            logger.debug("No hits to rerank")
            return []

        if not query or not query.strip():
            logger.warning("Empty query provided for reranking")
            return hits[:top_k]

        try:
            # Prepare query-passage pairs
            pairs = [
                (query, _truncate_chars(hit.get("text", ""), self.max_passage_chars))
                for hit in hits
            ]

            logger.debug(f"Reranking {len(pairs)} query-passage pairs")

            # Get cross-encoder scores
            scores = self.model.predict(
                pairs, batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=False
            )

            # Create a copy of hits to avoid modifying the original
            reranked_hits = []
            for hit, score in zip(hits, scores.tolist(), strict=False):
                hit_copy = hit.copy()
                hit_copy["ce_score"] = float(score)
                reranked_hits.append(hit_copy)

            # Sort by cross-encoder score (descending) and limit to top_k
            reranked_hits.sort(key=lambda x: x.get("ce_score", 0.0), reverse=True)
            result = reranked_hits[: min(top_k, len(reranked_hits))]

            logger.debug(f"Reranked {len(hits)} hits to {len(result)} results")
            return result

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Return original hits if reranking fails
            return hits[:top_k]
