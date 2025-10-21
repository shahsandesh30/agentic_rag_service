from __future__ import annotations

import re

CC_RE = re.compile(r"\b((?:\d[ -]?){13,19})\b")
SECRET_RE = re.compile(r"(api[_-]?key|secret|password)\s*[:=]\s*([^\s,;]{6,})", re.I)


def mask_pii(text: str) -> str:
    if not text:
        return text
    t = CC_RE.sub("****-****-****-****", text)
    t = SECRET_RE.sub(lambda m: f"{m.group(1)}=********", t)
    return t


def redact_injection_lines(text: str) -> tuple[str, int]:
    """
    Remove lines containing common injection instructions but keep the rest.
    Returns (sanitized_text, redaction_count)
    """
    if not text:
        return text, 0
    lines = text.splitlines()
    kept, redactions = [], 0
    from .detectors import INJECTION_RE

    for ln in lines:
        if INJECTION_RE.search(ln):
            redactions += 1
            continue
        kept.append(ln)
    return "\n".join(kept), redactions
