from __future__ import annotations
from typing import Dict, List, Tuple
from .policy import DEFAULT_POLICY, SafetyPolicy
from .detectors import (
    detect_prohibited_intent, flagged_context_chunks, detect_pii_or_secrets
)
from .sanitize import mask_pii, redact_injection_lines

def preflight(
    question: str,
    context_full_map: Dict[str, str],
    policy: SafetyPolicy = DEFAULT_POLICY
) -> Tuple[bool, Dict, Dict[str, str]]:
    """
    Returns: (blocked, safety_info, sanitized_full_map)
    - blocked: True if request should be refused
    - safety_info: dict to be merged into final payload.safety
    - sanitized_full_map: possibly filtered/redacted context texts
    """
    safety = {"blocked": False, "reason": None, "flags": [], "risk_score": 0.0,
              "filtered_chunk_ids": [], "redactions": 0}

    # 1) hard intent block
    if policy.block_high_risk_intent and detect_prohibited_intent(question):
        safety.update({"blocked": True, "reason": "Prohibited intent detected", "flags": ["prohibited_intent"], "risk_score": 1.0})
        return True, safety, {}

    sanitized = dict(context_full_map)

    # 2) filter injected chunks
    if policy.filter_injected_chunks:
        bad = flagged_context_chunks(context_full_map)
        if bad:
            safety["flags"].append("context_injection")
            safety["filtered_chunk_ids"] = bad
            for cid in bad:
                sanitized.pop(cid, None)

    # 3) redact injection lines within remaining chunks
    redactions = 0
    for cid, body in list(sanitized.items()):
        new_body, n = redact_injection_lines(body)
        if n:
            redactions += n
            sanitized[cid] = new_body
    safety["redactions"] = redactions

    # 4) basic PII signal in context (for telemetry)
    pii_hits = 0
    for body in sanitized.values():
        pii_hits += detect_pii_or_secrets(body)["hits"]
    if pii_hits > 0:
        safety["flags"].append("pii_suspected_in_context")

    # crude risk score
    safety["risk_score"] = min(1.0, 0.2*len(safety["filtered_chunk_ids"]) + 0.05*redactions + (0.1 if pii_hits else 0.0))
    return False, safety, sanitized

def postflight(payload: Dict, safety_info: Dict, policy: SafetyPolicy = DEFAULT_POLICY) -> Dict:
    """
    Apply answer-side masking and confidence adjustments, then merge safety info.
    """
    payload.setdefault("safety", {})
    # Mask obvious PII/secrets in answer
    if policy.pii_mask_answer and isinstance(payload.get("answer"), str):
        payload["answer"] = mask_pii(payload["answer"])

    # If a lot of context was filtered, cap or floor confidence
    if safety_info.get("filtered_chunk_ids"):
        payload["confidence"] = min(policy.confidence_cap_for_risky, float(payload.get("confidence", 0.5)))
        payload["safety"]["reason"] = (payload["safety"].get("reason") or "Filtered suspicious context; confidence capped.")

    # Merge flags/risk details
    merged = {**payload["safety"], **{k: v for k, v in safety_info.items() if k not in ("blocked", "reason")}}
    payload["safety"] = {"blocked": bool(safety_info.get("blocked")), "reason": payload["safety"].get("reason") or safety_info.get("reason"), **merged}
    return payload
