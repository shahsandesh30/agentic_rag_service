from __future__ import annotations

import re

# Prompt-injection phrases commonly seen in malicious context
INJECTION_PATTERNS = [
    r"ignore (the )?(previous|above) instructions",
    r"disregard (the )?(system|prior) prompt",
    r"reveal (the )?system prompt",
    r"you are now (?:unchained|developer mode)",
    r"do not follow (?:the )?rules",
    r"act as (?:.+?) and do .*",
    r"erase memory|self[- ]?destruct|override guardrails",
    r"\[/?INST\]|###\s*instruction",
]
INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.I)

# Obvious PII / secrets indicators
PII_RES = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # US SSN (example)
    re.compile(r"\b(?:\d[ -]*?){13,19}\b"),  # credit card-ish
    re.compile(r"\b(passport|ssn|national id|aadhaar)\b", re.I),
]
SECRET_RES = [
    re.compile(r"\bapi[_-]?key\b", re.I),
    re.compile(r"\bsecret\b", re.I),
    re.compile(r"\bpassword\b", re.I),
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS Access Key-like
]

# Dangerous intent keywords (very small heuristic list; expand for your domain)
PROHIBITED_INTENT = re.compile(
    r"(build a bomb|make explosives|ransomware|keylogger|zero[- ]day|sql injection payload|xss payload|"
    r"bypass paywall|crack license|credential stuffing|disable security|backdoor)",
    re.I,
)


def detect_injection(text: str) -> bool:
    return bool(INJECTION_RE.search(text or ""))


def detect_pii_or_secrets(text: str) -> dict[str, int]:
    t = text or ""
    matches = 0
    for r in PII_RES + SECRET_RES:
        matches += len(r.findall(t))
    return {"hits": matches}


def detect_prohibited_intent(question: str) -> bool:
    return bool(PROHIBITED_INTENT.search(question or ""))


def flagged_context_chunks(blocks: dict[str, str]) -> list[str]:
    """Return chunk_ids whose text appears to contain injection signals."""
    bad = []
    for cid, body in blocks.items():
        if detect_injection(body):
            bad.append(cid)
    return bad
