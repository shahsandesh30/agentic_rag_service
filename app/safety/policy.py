# app/safety/policy.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SafetyPolicy:
    level: str = "standard"  # "permissive" | "standard" | "strict"
    pii_mask_answer: bool = True
    filter_injected_chunks: bool = True
    block_high_risk_intent: bool = True
    max_redactions_per_chunk: int = 6
    # if too many chunks are filtered, degrade confidence
    confidence_floor_after_filtering: float = 0.35
    confidence_cap_for_risky: float = 0.6


DEFAULT_POLICY = SafetyPolicy()
