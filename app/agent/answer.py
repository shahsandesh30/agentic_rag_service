from __future__ import annotations
from typing import Dict, Any, List
from app.qa.answer import answer_question
from app.retrieval.hybrid import HybridSearcher
from app.llm.hf import HFGenerator

def answer_with_rewrites(
    question: str,
    rewrites: List[str],
    searcher: HybridSearcher,
    generator: HFGenerator,
    *,
    confidence_gate: float = 0.65,
    top_k_ctx: int = 8,
) -> Dict[str, Any]:
    attempts: List[Dict[str, Any]] = []
    best = None
    best_c = -1.0

    for i, q in enumerate(rewrites):
        payload = answer_question(
            q,
            searcher,
            generator,
            top_k_ctx=top_k_ctx,
            rerank=True,
            rerank_k=20,
        )
        d = payload.dict()
        d["_rewrite"] = q
        attempts.append(d)
        c = float(d.get("confidence", 0.0))
        if c > best_c:
            best_c, best = c, d
        # Early exit if confident enough on the original or first rewrite
        if i == 0 and c >= confidence_gate:
            break

    return {"best": best, "answers": attempts}
