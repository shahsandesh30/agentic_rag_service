# app/agent/router.py
from __future__ import annotations
from typing import Literal, Optional
import re
from app.llm.groq_gen import GroqGenerator

# 🔎 Regex hints tuned for Australian legal documents
RAG_LEGAL_HINTS = re.compile(
    r"(law|legal|act|statute|crime|criminal|civil|constitution|case law|precedent|penalty|offence|offense|section|article|regulation|clause|court|justice|tribunal|appeal|ruling|decision)",
    re.I,
)

def route(
    question: str,
    allow_llm_fallback: bool = False,
    generator: Optional[GroqGenerator] = None
) -> Literal["rag", "chitchat"]:
    """
    Lightweight intent classifier:
    - Heuristic rules first (regex & length).
    - Optionally falls back to Groq LLM for classification.
    - Returns either 'rag' (legal/document QA) or 'chitchat'.
    """
    q = question.strip()

    # --- Rule 1: Very short queries are usually chit-chat ---
    if len(q.split()) <= 3 and not RAG_LEGAL_HINTS.search(q):
        return "chitchat"

    # --- Rule 2: Legal hints detected → RAG ---
    if RAG_LEGAL_HINTS.search(q):
        return "rag"

    # --- Optional LLM fallback for ambiguous cases ---
    if allow_llm_fallback:
        if generator is None:
            raise ValueError("GroqGenerator required when allow_llm_fallback=True")

        label = generator.generate(
            prompt=f"Classify the user question as 'rag' (legal/document QA) or 'chitchat'. "
                   f"Output exactly one word: rag or chitchat.\nQ: {q}",
            contexts=[],
            system="You are a strict classifier. Only respond with 'rag' or 'chitchat'.",
            max_tokens=16,
            temperature=0.0
        ).strip().lower()

        return "rag" if "rag" in label else "chitchat"

    # --- Default: Assume retrieval needed (safer for legal QA) ---
    return "rag"
