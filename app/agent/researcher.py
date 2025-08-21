from __future__ import annotations
from typing import List
from app.llm.hf import HFGenerator

def make_rewrites(question: str, max_rewrites: int = 3) -> List[str]:
    gen = HFGenerator()
    prompt = (
        "Rewrite the question into up to 3 retrieval-friendly queries, "
        "covering synonyms, likely terms, and missing specifics. "
        "Return each on a new line without numbering."
        f"\nOriginal: {question}"
    )
    text = gen.generate(prompt, contexts=[], system="You produce terse search queries.")
    lines = [ln.strip("- •\t ").strip() for ln in text.splitlines() if ln.strip()]
    uniq = []
    for s in lines:
        if s.lower() not in {u.lower() for u in uniq}:
            uniq.append(s)
        if len(uniq) >= max_rewrites:
            break
    # Ensure original first
    return [question] + uniq
