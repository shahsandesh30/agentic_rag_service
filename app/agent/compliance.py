from __future__ import annotations
import re
from typing import Dict, Any

PII_RE = re.compile(r"\b(ssn|social security|passport|credit card|cvv)\b", re.I)
CRED_RE = re.compile(r"(password|api[_-]?key|secret)", re.I)

def check(payload: Dict[str, Any]) -> Dict[str, Any]:
    txt = (payload.get("answer") or "").lower()
    blocked = False
    reason = None
    if PII_RE.search(txt):
        blocked, reason = True, "Contains potential PII."
    elif CRED_RE.search(txt):
        # don’t block, but downgrade
        payload["confidence"] = max(0.2, min(0.6, float(payload.get("confidence", 0.5))))
        reason = "Credential mention detected; confidence reduced."
    payload.setdefault("safety", {})
    payload["safety"]["blocked"] = blocked
    payload["safety"]["reason"] = reason
    if blocked:
        payload["answer"] = "I can’t provide that."
        payload["citations"] = []
    return payload
