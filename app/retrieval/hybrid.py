# app/retrieval/hybrid.py
from __future__ import annotations
from typing import List, Dict
from app.retrieval.vector import VectorSearcher
from app.retrieval.bm25 import BM25Searcher
from app.retrieval.store import connect, fetch_chunk_texts, fetch_full_chunks
from app.retrieval.rerank import Reranker

def rrf_fuse(v_hits: List[Dict], b_hits: List[Dict], rrf_k: int = 60, top_k: int = 10) -> List[Dict]:
    """
    Reciprocal Rank Fusion:
      score(id) = sum_over_lists( 1 / (rrf_k + rank(id)) )
    where rank starts at 1 inside each list.
    """
    ranks: dict[str, float] = {}
    slots: dict[str, Dict] = {}

    for lst in (v_hits, b_hits):
        for r, h in enumerate(lst, start=1):
            cid = h["chunk_id"]
            ranks[cid] = ranks.get(cid, 0.0) + 1.0 / (rrf_k + r)
            # keep first-seen meta (will fill missing later)
            slots.setdefault(cid, h)

    # pick top_k by fused score
    order = sorted(ranks.items(), key=lambda x: x[1], reverse=True)[:top_k]
    fused = []
    for cid, score in order:
        item = dict(slots.get(cid, {}))
        item["chunk_id"] = cid
        item["rrf"] = score
        fused.append(item)
    return fused

class HybridSearcher:
    def __init__(self, vector: VectorSearcher, bm25: BM25Searcher, db_path: str = "rag_local.db"):
        self.vector = vector
        self.bm25 = bm25
        self.db_path = db_path
        self._reranker = None

    def _ensure_reranker(self, model_name="BAAI/bge-reranker-base", max_passage_chars=1200, batch_size=16):
        if self._reranker is None:
            self._reranker = Reranker(model_name=model_name, max_passage_chars=max_passage_chars, batch_size=batch_size)

    def search(
        self,
        query: str,
        top_k: int = 10,
        k_vector: int = 40,
        k_bm25: int = 40,
        rrf_k: int = 60,
        rerank: bool = False,
        rerank_k: int = 20,
        reranker_model: str = "BAAI/bge-reranker-base",
        reranker_batch: int = 16,
        max_passage_chars: int = 1200,
    ) -> List[Dict]:
        v_hits = self.vector.search(query, top_k=k_vector)
        b_hits = self.bm25.search(query, top_k=k_bm25)
        fused = rrf_fuse(v_hits, b_hits, rrf_k=rrf_k, top_k=max(top_k, rerank_k) if rerank else top_k)

        # hydrate missing snippet fields (as before)
        missing = [h["chunk_id"] for h in fused if not h.get("text")]
        if missing:
            conn = connect(self.db_path)
            meta_map = fetch_chunk_texts(conn, missing)
            conn.close()
            for h in fused:
                if not h.get("text"):
                    m = meta_map.get(h["chunk_id"], {})
                    h.update({
                        "section": h.get("section") or m.get("section"),
                        "source": h.get("source") or m.get("source"),
                        "path": h.get("path") or m.get("path"),
                        "text": m.get("text", "")[:400]
                    })

        if rerank and fused:
            # fetch full texts for top rerank_k
            top_candidates = fused[: min(rerank_k, len(fused))]
            conn = connect(self.db_path)
            full_map = fetch_full_chunks(conn, [h["chunk_id"] for h in top_candidates])
            conn.close()
            for h in top_candidates:
                # replace snippet with full text for scoring; keep snippet for API payload later
                h["__full_text"] = full_map.get(h["chunk_id"], h.get("text",""))

            # build copies with full text for reranking
            to_score = [{**h, "text": h["__full_text"]} for h in top_candidates]

            self._ensure_reranker(reranker_model, max_passage_chars=max_passage_chars, batch_size=reranker_batch)
            reranked = self._reranker.rerank(query, to_score, top_k=top_k)

            # drop temp full text field from originals and return reranked list
            for h in fused:
                h.pop("__full_text", None)
            return reranked

        # no rerank → return fused top_k
        return fused[:top_k]
