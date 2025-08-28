# app/qa/answer.py
from __future__ import annotations
from typing import List, Dict
import json
import math
import time

from pydantic import ValidationError

from app.retrieval.hybrid import HybridSearcher
from app.retrieval.store import connect, fetch_full_chunks
from app.llm.hf import HFGenerator
from app.qa.prompt import SYSTEM_RULES, make_context_blocks, make_user_prompt
from app.qa.schema import AnswerPayload, Citation

# Safety (Step 9)
from app.safety.guard import preflight, postflight
from app.safety.policy import DEFAULT_POLICY

# Observability (Step 11)
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
    """
    Heuristic: prefer cross-encoder scores; else use fused (rrf) / base scores.
    Cap to a safe range so we don't over-promise.
    """
    ce = [h.get("ce_score") for h in hits if "ce_score" in h]
    base = ce if ce else [h.get("rrf", h.get("score", 0.0)) for h in hits]
    if not base:
        return 0.35
    norm_top = _normalize([float(x) for x in base[:5]])
    raw = sum(norm_top) / max(1, len(norm_top))
    return max(0.35, min(0.90, 0.35 + 0.55 * raw))


def _default_citations(hits: List[Dict], limit: int = 5) -> List[Dict]:
    cites: List[Dict] = []
    for h in hits[:limit]:
        cites.append({
            "chunk_id": h["chunk_id"],
            "source": h.get("source"),
            "path": h.get("path"),
            "section": h.get("section"),
        })
    return cites


def answer_question(
    question: str,
    searcher: HybridSearcher,
    generator: HFGenerator,
    *,
    top_k_ctx: int = 8,
    k_vector: int = 40,
    k_bm25: int = 40,
    rrf_k: int = 60,
    rerank: bool = True,
    rerank_k: int = 20,
) -> AnswerPayload:
    """
    Orchestrates: retrieve → guard.preflight → prompt → generate → parse → guard.postflight → validate
    Returns a validated AnswerPayload.
    """

    # 1) Retrieve candidates (hybrid + optional rerank)
    hits = searcher.search(
        question,
        top_k=max(top_k_ctx, 10),
        k_vector=k_vector,
        k_bm25=k_bm25,
        rrf_k=rrf_k,
        rerank=rerank,
        rerank_k=rerank_k,
    )
    if not hits:
        return AnswerPayload(
            answer="I don’t have enough information in the provided corpus to answer that.",
            citations=[],
            confidence=0.35,
            safety={"blocked": False},
        )

    # 2) Hydrate full texts for context
    conn = connect(searcher.db_path)
    full_map = fetch_full_chunks(conn, [h["chunk_id"] for h in hits[:top_k_ctx]])
    conn.close()

    # 3) Safety preflight (filter injected chunks, compute risk, maybe block)
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
            safety={
                "blocked": True,
                "reason": safety_info.get("reason"),
                "flags": safety_info.get("flags"),
                "risk_score": safety_info.get("risk_score"),
                "filtered_chunk_ids": safety_info.get("filtered_chunk_ids", []),
                "redactions": safety_info.get("redactions", 0),
            },
        )

    # 4) Build prompt
    context_blocks = make_context_blocks(hits, sanitized_full_map, max_blocks=top_k_ctx)
    system = SYSTEM_RULES
    user = make_user_prompt(question)

    # Token estimation (approximate, for observability)
    try:
        tok = generator.tokenizer
        inp_tokens = len(tok(system + "\n\n" + "\n\n".join(context_blocks) + "\n\n" + user).input_ids)
        TOKENS_INPUT.inc(inp_tokens)
    except Exception:
        pass

    # 5) Generate with the local Transformers LLM
    with _tracer.start_as_current_span("generation.llm"):
        t0 = time.perf_counter()
        # HFGenerator.generate expects (prompt, contexts, system)
        raw = generator.generate
        try:
            text = raw(
                prompt=user,
                contexts=context_blocks,
                system=system,
            )
        except Exception:
            text = ""
        finally:
            dt = time.perf_counter() - t0
            try:
                GENERATION_LATENCY.observe(dt)
            except Exception:
                pass

    # 6) Safety postflight and finalize answer
    try:
        final_text, blocked2, info2 = postflight(text, policy=DEFAULT_POLICY)
    except Exception:
        final_text, blocked2, info2 = text, False, {}

    if blocked2:
        return AnswerPayload(
            answer="I can’t help with that.",
            citations=[],
            confidence=0.2,
            safety={
                "blocked": True,
                "reason": info2.get("reason"),
                "flags": info2.get("flags"),
                "risk_score": info2.get("risk_score"),
            },
        )

    citations = _default_citations(hits, limit=min(5, top_k_ctx))
    confidence = _confidence_from_hits(hits)

    # 7) Token output metric (best-effort)
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
