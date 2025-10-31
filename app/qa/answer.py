# app/qa/answer.py
from __future__ import annotations
from typing import List, Dict, Any
import time
import json

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


def _default_citations(hits: List[Dict], limit: int = 5) -> List[Dict[str, Any]]:
    cites: List[Dict[str, Any]] = []
    for h in hits[:limit]:
        cites.append({
            "source": h.get("source"),
            "path": h.get("path"),
            "section": h.get("section"),
        })
    return cites


def _parse_model_json(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        return {}
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _extract_answer(model_payload: Dict[str, Any], fallback: str) -> str:
    answer = model_payload.get("answer")
    if isinstance(answer, str) and answer.strip():
        return answer.strip()
    return fallback.strip() if isinstance(fallback, str) else str(fallback)


def _extract_citations(model_payload: Dict[str, Any], hits: List[Dict], *, limit: int) -> List[Dict[str, Any]]:
    raw = model_payload.get("citations")
    cites: List[Dict[str, Any]] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                cites.append({
                    "source": item.get("source"),
                    "path": item.get("path"),
                    "section": item.get("section"),
                })
    return cites or _default_citations(hits, limit=limit)


def _extract_confidence(model_payload: Dict[str, Any], hits: List[Dict]) -> float:
    value = model_payload.get("confidence")
    if isinstance(value, (int, float)):
        try:
            clamped = float(value)
        except (TypeError, ValueError):
            clamped = None
        else:
            if 0.0 <= clamped <= 1.0:
                return clamped
    return _confidence_from_hits(hits)


def _extract_safety(model_payload: Dict[str, Any], *, base: Dict[str, Any] | None = None) -> Dict[str, Any]:
    safety: Dict[str, Any] = {"blocked": False}
    if isinstance(base, dict):
        safety.update(base)
    raw = model_payload.get("safety")
    if isinstance(raw, dict):
        safety.update(raw)
    safety["blocked"] = bool(safety.get("blocked", False))
    return safety


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
            text = generator.generate(prompt=user, contexts=context_blocks, system=system, max_tokens=1024)
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

    model_payload = _parse_model_json(final_text if isinstance(final_text, str) else "")

    answer_text = _extract_answer(model_payload, final_text)
    citations = _extract_citations(model_payload, hits, limit=min(5, top_k_ctx))
    confidence = _extract_confidence(model_payload, hits)
    safety_payload = _extract_safety(model_payload, base=info2)

    try:
        out_tokens = len(generator.tokenizer(answer_text).input_ids)
        TOKENS_OUTPUT.inc(out_tokens)
    except Exception:
        pass

    return AnswerPayload(
        answer=answer_text,
        citations=citations,
        confidence=confidence,
        safety=safety_payload,
    )
