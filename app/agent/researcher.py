# app/agent/researcher.py
from __future__ import annotations
from typing import List, Optional
from app.llm.groq_gen import GroqGenerator

def make_rewrites(
    question: str,
    max_rewrites: int = 3,
    generator: Optional[GroqGenerator] = None
) -> List[str]:
    """
    Generate alternative rewrites of the user's question to improve recall.
    Useful for retrieval in legal/RAG pipelines.
    """
    if generator is None:
        raise ValueError("GroqGenerator instance must be provided to make_rewrites")

    prompt = (
        "Rewrite the following legal question into up to 3 retrieval-friendly queries. "
        "Use synonyms, alternative legal terms, or related formulations. "
        "Each rewrite must be short and standalone. "
        "Return each rewrite on a new line, without numbering."
        f"\nOriginal: {question}"
    )

    text = generator.generate(
        prompt=prompt,
        contexts=[],
        system="You produce terse search queries optimized for legal document retrieval."
    )

    lines = [ln.strip("- •\t ").strip() for ln in text.splitlines() if ln.strip()]
    uniq = []
    for s in lines:
        if s.lower() not in {u.lower() for u in uniq}:
            uniq.append(s)
        if len(uniq) >= max_rewrites:
            break

    # Ensure original question comes first
    return [question] + uniq
