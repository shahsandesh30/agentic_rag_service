# app/agent/answer.py
from __future__ import annotations
from typing import Dict, Any, List
from app.qa.answer import answer_question
from app.retrieval.faiss_sqlite import FaissSqliteSearcher
from app.llm.groq_gen import GroqGenerator
from app.qa.prompt import make_context_blocks, make_user_prompt, SYSTEM_RULES

def answer_with_rewrites(
    question: str,
    rewrites: List[str],
    searcher: FaissSqliteSearcher,
    generator: GroqGenerator,
    *,
    confidence_gate: float = 0.65,
    top_k_ctx: int = 8,
    mode: str = "merge",  # "multi" = current behavior, "merge" = new
) -> Dict[str, Any]:
    """
    Answer a question using original + rewrites.
    
    Modes:
    - multi: run retrieval + generation per rewrite, choose best answer.
    - merge: merge retrieval hits from all rewrites, one generation call.
    """

    # --- Mode 1: MULTI (existing behavior) ---
    if mode == "multi":
        attempts: List[Dict[str, Any]] = []
        best = None
        best_c = -1.0

        for i, q in enumerate([question] + rewrites):  # include original first
            payload = answer_question(
                q,
                searcher,
                generator,
                top_k_ctx=top_k_ctx,
            )
            d = payload.dict()
            d["_rewrite"] = q
            attempts.append(d)

            c = float(d.get("confidence", 0.0))
            if c > best_c:
                best_c, best = c, d

            # Early exit if original is confident enough
            if i == 0 and c >= confidence_gate:
                break

        return {"best": best, "answers": attempts}

    # --- Mode 2: MERGE (new optimization) ---
    elif mode == "merge":
        # Collect hits from all queries
        all_hits = []
        for q in [question] + rewrites:
            hits = searcher.search(q, top_k=top_k_ctx)
            all_hits.extend(hits)

        # Deduplicate by chunk_id
        seen = set()
        unique_hits = []
        for h in all_hits:
            if h["chunk_id"] not in seen:
                unique_hits.append(h)
                seen.add(h["chunk_id"])

        # Sort by score, keep top-N
        unique_hits.sort(key=lambda h: h["score"], reverse=True)
        merged_hits = unique_hits[:top_k_ctx]

        # Build context
        full_map = {h["chunk_id"]: h["text"] for h in merged_hits}
        context_blocks = make_context_blocks(merged_hits, full_map, max_blocks=top_k_ctx)

        # Generate once
        text = generator.generate(
            prompt=make_user_prompt(question),
            contexts=context_blocks,
            system=SYSTEM_RULES,
        )

        # Build answer payload
        best = {
            "answer": text.strip(),
            "citations": [
                {"source": h.get("source"), "path": h.get("path"), "section": h.get("section")}
                for h in merged_hits
            ],
            "confidence": min(0.9, max(h["score"] for h in merged_hits) if merged_hits else 0.35),
            "safety": {"blocked": False},
            "_mode": "merge",
        }

        return {"best": best, "answers": [best]}

    else:
        raise ValueError("mode must be either 'multi' or 'merge'")
