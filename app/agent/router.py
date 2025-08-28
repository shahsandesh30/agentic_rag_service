from __future__ import annotations
from typing import Literal
import re
from app.llm.hf import HFGenerator
from typing import Optional

RAG_HINTS = re.compile(
    r"(What are|policy|how do i|where is|procedure|regulation|guide|manual|refund|kyc|risk|terms|contract|compliance|step|install|configure|error|exception)",
    re.I,
)

def route(question: str, allow_llm_fallback: bool = False, generator: Optional[HFGenerator] = None) -> Literal["rag", "chitchat"]:
    # Simple heuristics first
    q = question.strip()
    if len(q.split()) <= 3 and not RAG_HINTS.search(q):
        print("router.py chitchatttttttttt")
        return "chitchat"
    if RAG_HINTS.search(q):
        print("router.py raggggggggggggggg1")
        return "rag"
    # Optional LLM fallback (local, fast)
    if allow_llm_fallback:
        print("router.py falbackkkkkkkkkk")
        gen = generator or HFGenerator()
        label = gen.generate(
            prompt=f"Classify the user question as 'rag' or 'chitchat'. Only output rag or chitchat.\nQ: {q}",
            contexts=[],
            system="You are a classifier. Output a single token: rag or chitchat."
        ).strip().lower()
        print("router.py falbackkkkk resultttt:", label )
        return "rag" if "rag" in label else "chitchat"
    return "rag"
