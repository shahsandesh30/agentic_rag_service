from __future__ import annotations
from typing import Literal
import re
from app.llm.hf import HFGenerator

RAG_HINTS = re.compile(
    r"(policy|how do i|where is|procedure|regulation|guide|manual|refund|kyc|risk|terms|contract|compliance|step|install|configure|error|exception)",
    re.I,
)

def route(question: str, allow_llm_fallback: bool = True) -> Literal["rag", "chitchat"]:
    # Simple heuristics first
    q = question.strip()
    if len(q.split()) <= 3 and not RAG_HINTS.search(q):
        return "chitchat"
    if RAG_HINTS.search(q):
        return "rag"
    # Optional LLM fallback (local, fast)
    if allow_llm_fallback:
        gen = HFGenerator()
        label = gen.generate(
            prompt=f"Classify the user question as 'rag' or 'chitchat'. Only output rag or chitchat.\nQ: {q}",
            contexts=[],
            system="You are a classifier. Output a single token: rag or chitchat."
        ).strip().lower()
        return "rag" if "rag" in label else "chitchat"
    return "rag"
