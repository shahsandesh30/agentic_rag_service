# app/agent/router.py
from __future__ import annotations
from typing import Literal, Optional
import re
from app.llm.groq_gen import GroqGenerator

RAG_HINTS = re.compile(
    r"(What are|policy|how do i|where is|procedure|regulation|guide|manual|refund|kyc|risk|terms|contract|compliance|step|install|configure|error|exception)",
    re.I,
)

def route(
    question: str,
    allow_llm_fallback: bool = False,
    generator: Optional[GroqGenerator] = None
) -> Literal["rag", "chitchat"]:
    """
    Lightweight intent classifier:
    - Heuristic first (regex & length).
    - Optionally falls back to Groq LLM for classification.
    """
    q = question.strip()

    # Simple heuristics
    if len(q.split()) <= 3 and not RAG_HINTS.search(q):
        return "chitchat"
    if RAG_HINTS.search(q):
        return "rag"

    # Optional Groq-based fallback
    if allow_llm_fallback:
        if generator is None:
            raise ValueError("GroqGenerator required when allow_llm_fallback=True")

        label = generator.generate(
            prompt=f"Classify the user question as 'rag' or 'chitchat'. "
                   f"Output exactly one word: rag or chitchat.\nQ: {q}",
            contexts=[],
            system="You are a strict classifier. Only respond with 'rag' or 'chitchat'."
        ).strip().lower()

        return "rag" if "rag" in label else "chitchat"

    # Default → assume retrieval needed
    return "rag"
