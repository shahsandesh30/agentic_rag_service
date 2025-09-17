# app/qa/answer.py
from __future__ import annotations
from typing import List, Dict
import time

from app.llm.groq_gen import GroqGenerator  # using Groq now
from app.qa.prompt import SYSTEM_RULES, make_context_blocks, make_user_prompt
from app.qa.schema import AnswerPayload

# Safety
from app.safety.guard import preflight, postflight
from app.safety.policy import DEFAULT_POLICY

# Observability
from app.obs.metrics import GENERATION_LATENCY, TOKENS_INPUT, TOKENS_OUTPUT
from opentelemetry import trace
_tracer = trace.get_tracer(__name__)


def _normalize(nums: List[float]) -> List[float]:
    if not nums:
        return []
    lo, hi = min(nums), max(nums)
    if hi - lo < 1e-9:
        return [1.0 for _ in nums]
    return [(x - lo) / (hi - lo) for x in nums]


def _confidence_from_hits(hits: List[Dict]) -> float:
    base = [h.get("score", 0.0) for h in hits]
    if not base:
        return 0.35
    norm_top = _normalize([float(x) for x in base[:5]])
    raw = sum(norm_top) / max(1, len(norm_top))
    return max(0.35, min(0.90, 0.35 + 0.55 * raw))


def _default_citations(hits: List[Dict], limit: int = 5) -> List[Dict]:
    cites: List[Dict] = []
    for h in hits[:limit]:
        cites.append({
            "source": h.get("source"),
            "path": h.get("path"),
            "section": h.get("section"),
        })
    return cites


def answer_question(
    question: str,
    searcher,            # FaissSearcher only
    generator: GroqGenerator,
    *,
    top_k_ctx: int = 8,
) -> AnswerPayload:
    """
    Orchestrates: retrieve → guard.preflight → prompt → generate → postflight → validate
    """

    # 1) Retrieve candidates (directly from FAISS)
    hits = searcher.search(question, top_k=max(top_k_ctx, 10))

    if not hits:
        return AnswerPayload(
            answer="I don’t have enough information in the corpus to answer that.",
            citations=[],
            confidence=0.35,
            safety={"blocked": False},
        )

    # 2) Contexts: just take hit["text"] directly
    full_map = {i: h["text"] for i, h in enumerate(hits[:top_k_ctx])}

    # 3) Safety preflight
    blocked, safety_info, sanitized_full_map = preflight(
        question,
        full_map,
        policy=DEFAULT_POLICY
    )
    if blocked:
        return AnswerPayload(
            answer="I can’t help with that.",
            citations=[],
            confidence=0.2,
            safety={"blocked": True, **safety_info},
        )

    # 4) Prompt building
    context_blocks = make_context_blocks(hits, sanitized_full_map, max_blocks=top_k_ctx)
    system = SYSTEM_RULES
    user = make_user_prompt(question)

    try:
        tok = generator.tokenizer
        inp_tokens = len(tok(system + "\n\n" + "\n\n".join(context_blocks) + "\n\n" + user).input_ids)
        TOKENS_INPUT.inc(inp_tokens)
    except Exception:
        pass

    # 5) Generate
    with _tracer.start_as_current_span("generation.llm"):
        t0 = time.perf_counter()
        try:
            text = generator.generate(prompt=user, contexts=context_blocks, system=system)
        except Exception:
            text = ""
        finally:
            dt = time.perf_counter() - t0
            try:
                GENERATION_LATENCY.observe(dt)
            except Exception:
                pass

    # 6) Safety postflight
    try:
        final_text, blocked2, info2 = postflight(text, policy=DEFAULT_POLICY)
    except Exception:
        final_text, blocked2, info2 = text, False, {}

    if blocked2:
        return AnswerPayload(
            answer="I can’t help with that.",
            citations=[],
            confidence=0.2,
            safety={"blocked": True, **info2},
        )

    citations = _default_citations(hits, limit=min(5, top_k_ctx))
    confidence = _confidence_from_hits(hits)

    try:
        out_tokens = len(generator.tokenizer(final_text).input_ids)
        TOKENS_OUTPUT.inc(out_tokens)
    except Exception:
        pass

    return AnswerPayload(
        answer=final_text.strip() if isinstance(final_text, str) else str(final_text),
        citations=citations,
        confidence=confidence,
        safety={"blocked": False},
    )
