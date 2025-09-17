# app/agent/answer.py
from __future__ import annotations
from typing import Dict, Any, List
from app.qa.answer import answer_question
from app.retrieval.faiss_sqlite import FaissSqliteSearcher
from app.llm.groq_gen import GroqGenerator

def answer_with_rewrites(
    question: str,
    rewrites: List[str],
    searcher: FaissSqliteSearcher,
    generator: GroqGenerator,
    *,
    confidence_gate: float = 0.65,
    top_k_ctx: int = 8,
) -> Dict[str, Any]:
    """
    Try answering the original question and its rewrites.
    Pick the answer with the highest confidence.
    """
    attempts: List[Dict[str, Any]] = []
    best = None
    best_c = -1.0

    for i, q in enumerate([question] + rewrites):  # include original first
        payload = answer_question(
            q,
            searcher,
            generator,
            top_k_ctx=top_k_ctx,
            rerank=False,  # rerank disabled in FAISS-only baseline
        )
        d = payload.dict()
        d["_rewrite"] = q
        attempts.append(d)

        c = float(d.get("confidence", 0.0))
        if c > best_c:
            best_c, best = c, d

        # Early exit if confident enough on the original
        if i == 0 and c >= confidence_gate:
            break

    return {"best": best, "answers": attempts}
