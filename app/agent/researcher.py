# app/agent/researcher.py
from __future__ import annotations
from typing import List, Optional
from app.llm.groq_gen import GroqGenerator
from app.retrieval.faiss_sqlite import FaissSqliteSearcher
import numpy as np

def make_rewrites(
    question: str,
    max_rewrites: int = 3,
    generator: Optional[GroqGenerator] = None,
    searcher: Optional[FaissSqliteSearcher] = None,
) -> List[str]:
    """
    Generate alternative rewrites of the user's question to improve recall.
    - Uses Groq LLM to propose rewrites.
    - Deduplicates semantically similar queries.
    - Optionally tests rewrites with FAISS search to keep only useful ones.
    """

    if generator is None:
        raise ValueError("GroqGenerator instance must be provided to make_rewrites")

    # --- Step 1: Ask LLM for rewrites ---
    prompt = (
        "Rewrite the following legal question into up to 5 retrieval-friendly queries. "
        "Use synonyms, alternative legal terms, or related formulations. "
        "Each rewrite must be short and standalone. "
        "Return each rewrite on a new line, without numbering."
        f"\nOriginal: {question}"
    )

    text = generator.generate(
        prompt=prompt,
        contexts=[],
        system="You produce terse search queries optimized for Australian legal document retrieval.",
        max_tokens=128,
        temperature=0.3,	
    )

    # --- Step 2: Clean + deduplicate ---
    lines = [ln.strip("- •\t ").strip() for ln in text.splitlines() if ln.strip()]
    uniq: List[str] = []
    for s in lines:
        if s.lower() not in {u.lower() for u in uniq}:
            uniq.append(s)

    # --- Step 3: Semantic deduplication (optional, if searcher provided) ---
    if searcher is not None:
        # Embed all queries (original + rewrites)
        all_queries = [question] + uniq
        vecs = searcher.embedder.encode(all_queries)

        keep = [0]  # always keep the original
        for i in range(1, len(all_queries)):
            sim_to_kept = max(
                np.dot(vecs[i], vecs[j]) / (np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j]) + 1e-12)
                for j in keep
            )
            if sim_to_kept < 0.92:  # similarity threshold → skip near-duplicates
                keep.append(i)
        uniq = [all_queries[i] for i in keep]

    # --- Step 4: Retrieval-aware filtering (optional) ---
    if searcher is not None:
        scored = []
        for q in uniq:
            hits = searcher.search(q, top_k=3)
            best_score = max((h["score"] for h in hits), default=0.0)
            scored.append((q, best_score))

        # Rank by retrieval strength
        scored.sort(key=lambda x: x[1], reverse=True)
        uniq = [q for q, _ in scored]

    # --- Step 5: Limit final rewrites ---
    return uniq[: max_rewrites + 1]  # +1 because original is always included
