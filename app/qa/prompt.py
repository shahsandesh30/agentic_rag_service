# app/qa/prompt.py
from __future__ import annotations
from typing import List, Dict

SYSTEM_RULES = """You are a grounded QA assistant for enterprise RAG.
Rules:
- Use ONLY the supplied context blocks.
- If the answer is not in context, say you don’t know and suggest where it might be found.
- Always include citations by chunk_id for every claim drawn from context.
- Be concise and factual. No speculation. No web browsing.
- Output STRICT JSON with keys: answer, citations, confidence, safety.
- JSON only, no markdown, no extra text.
"""

def make_context_blocks(hits: List[Dict], full_texts: Dict[str, str], max_blocks: int = 8) -> List[str]:
    """
    Format the top-N results as independent, clearly delimited blocks.
    Each block includes chunk_id, source/path/section header and the full text body.
    """
    blocks = []
    for h in hits[:max_blocks]:
        cid = h["chunk_id"]
        section = (h.get("section") or "").strip()
        path = (h.get("path") or "").strip()
        source = (h.get("source") or "").strip()
        body = full_texts.get(cid, "")
        header = f"CHUNK_ID: {cid}\nSOURCE: {source}\nPATH: {path}\nSECTION: {section}".strip()
        blocks.append(header + "\n---\n" + body.strip())
    return blocks

def make_user_prompt(question: str) -> str:
    return (
        "Answer the user question using the context blocks above.\n"
        "Return STRICT JSON with keys exactly: answer, citations, confidence, safety.\n"
        f"User question: {question}"
    )
