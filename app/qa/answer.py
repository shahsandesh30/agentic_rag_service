# app/qa/answer.py
from __future__ import annotations
from typing import List, Dict, Optional
import math, json
from pydantic import ValidationError

from app.retrieval.hybrid import HybridSearcher
from app.retrieval.store import connect, fetch_full_chunks
from app.llm.hf import HFGenerator
from app.qa.prompt import SYSTEM_RULES, make_context_blocks, make_user_prompt
from app.qa.schema import AnswerPayload, Citation

def _normalize(scores: List[float]) -> List[float]:
    if not scores: return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]

def _confidence_from_hits(hits: List[Dict]) -> float:
    # Prefer cross-encoder scores when present, else fused scores
    ce = [h.get("ce_score") for h in hits if "ce_score" in h]
    base = ce if ce else [h.get("rrf", h.get("score", 0.0)) for h in hits]
    if not base: return 0.3
    norm = _normalize([float(x) for x in base[:5]])
    raw = sum(norm) / max(1, len(norm))
    # map to [0.35, 0.9] to avoid overconfidence
    return max(0.35, min(0.9, 0.35 + 0.55 * raw))

def _default_citations(hits: List[Dict], limit: int = 5) -> List[Dict]:
    out = []
    for h in hits[:limit]:
        out.append({
            "chunk_id": h["chunk_id"],
            "source": h.get("source"),
            "path": h.get("path"),
            "section": h.get("section"),
        })
    return out

def answer_question(
    question: str,
    searcher: HybridSearcher,
    generator: HFGenerator,
    *,
    top_k_ctx: int = 8,
    k_vector: int = 40,
    k_bm25: int = 40,
    rrf_k: int = 60,
    rerank: bool = True,
    rerank_k: int = 20,
) -> AnswerPayload:
    # 1) retrieve
    hits = searcher.search(
        question,
        top_k=max(top_k_ctx, 10),
        k_vector=k_vector,
        k_bm25=k_bm25,
        rrf_k=rrf_k,
        rerank=rerank,
        rerank_k=rerank_k,
    )
    if not hits:
        return AnswerPayload(
            answer="I don’t have enough information in the provided corpus to answer that.",
            citations=[],
            confidence=0.35,
        )

    # 2) hydrate full texts for context assembly
    conn = connect(searcher.db_path)
    full_map = fetch_full_chunks(conn, [h["chunk_id"] for h in hits[:top_k_ctx]])
    conn.close()

    # 3) build prompts
    context_blocks = make_context_blocks(hits, full_map, max_blocks=top_k_ctx)
    system = SYSTEM_RULES
    user = make_user_prompt(question)

    # 4) generate
    raw = generator.generate({"prompt": user, "context": context_blocks, "system": system})
    # 'raw' might already be a dict from our Step 1 adapter; otherwise coerce
    if isinstance(raw, dict) and "answer" in raw:
        data = raw
    else:
        try:
            data = json.loads(raw if isinstance(raw, str) else json.dumps(raw))
        except Exception:
            data = {"answer": str(raw), "citations": [], "confidence": 0.5, "safety": {"blocked": False}}

    # 5) validate & auto-repair
    # fill missing citations with top chunks; clamp confidence
    if not data.get("citations"):
        data["citations"] = _default_citations(hits, limit=min(5, top_k_ctx))
    if "confidence" not in data or not isinstance(data["confidence"], (int, float)):
        data["confidence"] = _confidence_from_hits(hits)
    else:
        data["confidence"] = float(max(0.0, min(1.0, data["confidence"])))

    try:
        payload = AnswerPayload(**data)
    except ValidationError:
        # Hard repair
        payload = AnswerPayload(
            answer=str(data.get("answer") or "I don’t know based on the supplied context."),
            citations=[Citation(**c) for c in _default_citations(hits)],
            confidence=_confidence_from_hits(hits),
            safety=data.get("safety") or {"blocked": False},
        )

    return payload
