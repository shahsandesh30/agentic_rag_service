# app/agent/intent.py
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Literal

from app.config import get_settings

log = logging.getLogger(__name__)
settings = get_settings()

Label = Literal["rag", "web", "chitchat"]

# --- lightweight patterns (expand as desired) ---
RAG_HINTS = re.compile(
    r"(law|legal|act|statute|crime|criminal|civil|constitution|case law|precedent|"
    r"penalty|offence|offense|section|article|regulation|clause|court|justice|"
    r"tribunal|appeal|ruling|decision|legislation|statute|regulation|policy)",
    re.I,
)

WEB_HINTS = re.compile(
    r"(latest|today|news|update|current|recent|web|online|search|happening|"
    r"trending|breaking|now|live|real-time)",
    re.I,
)

CHITCHAT_HINTS = re.compile(
    r"(hello|hi|hey|thanks|thank you|bye|goodbye|how are you|what's up|"
    r"good morning|good afternoon|good evening)",
    re.I,
)


@dataclass
class IntentResult:
    label: Label
    confidence: float
    reasons: str


class IntentClassifier:
    """
    Hybrid intent classifier:
      1) Rules pass → returns (label, conf∈[0,1]) quickly
      2) Optional LLM pass (Groq) → soft prediction with rationale
      3) Fusion → pick label w/ higher confidence (or LLM tie-break)
    """

    def __init__(self, mode: str | None = None):
        self.mode = (mode or settings.intent_mode).lower()
        # Lazy import LLM provider only if needed
        self._llm = None

    # ---------------- rules ----------------
    def _rules(self, text: str) -> IntentResult:
        t = text.strip()
        if len(t) < settings.intent_min_query_len:
            return IntentResult("chitchat", 0.50, "very short query → chitchat default")

        # explicit hints
        if WEB_HINTS.search(t):
            return IntentResult("web", 0.80, "matched web/browse/latest keywords")
        if RAG_HINTS.search(t):
            return IntentResult("rag", 0.80, "matched doc/context/citation keywords")
        if CHITCHAT_HINTS.search(t):
            return IntentResult("chitchat", 0.75, "matched smalltalk keywords")

        # heuristic fallbacks
        if "http://" in t or "https://" in t:
            return IntentResult("web", 0.70, "contains URL → likely web")
        # Question-like → bias to rag; imperative/command → bias web
        if "?" in t:
            return IntentResult("rag", 0.60, "question form → likely retrieve/answer")
        if any(k in t.lower() for k in ("search", "find", "open", "check price", "watch")):
            return IntentResult("web", 0.60, "imperative verb → likely web")

        return IntentResult("rag", 0.55, "default to rag")

    # ---------------- llm ----------------
    def _ensure_llm(self):
        if self._llm is not None:
            return
        if settings.llm_provider != "groq" or not settings.groq_api_key:
            # leave None → LLM disabled
            return
        # Lightweight internal wrapper to avoid import cycles
        from groq import Groq  # type: ignore

        self._llm = Groq(api_key=settings.groq_api_key)

    def _llm_predict(self, text: str) -> IntentResult | None:
        try:
            self._ensure_llm()
            if self._llm is None:
                return None
            # prompt = (
            #     "You are an intent router. Classify the user's message as one of:\n"
            #     " - rag: needs retrieval from local documents / vector store\n"
            #     " - web: needs live web search / browsing the internet\n"
            #     " - chitchat: casual conversation without retrieval\n\n"
            #     "Return ONLY a compact JSON object with fields label (rag|web|chitchat) "
            #     'and confidence (0-1).\n'
            #     f"User: {text}\n"
            # )
            prompt = (
                f"Classify the user question as 'rag' (legal/document QA), 'web' (web search), or 'chitchat'. "
                f"Output exactly one word: rag, web, or chitchat.\nQ: {text}"
            )
            resp = self._llm.chat.completions.create(
                model=settings.groq_model,
                temperature=0.0,
                max_tokens=16,
                messages=[
                    {"role": "system", "content": "You return terse JSON only."},
                    {"role": "user", "content": prompt},
                ],
            )
            content = resp.choices[0].message.content or "{}"
            # very small JSON tolerance
            import json

            data = json.loads(content.strip().splitlines()[-1])
            lbl = str(data.get("label", "rag")).strip().lower()
            conf = float(data.get("confidence", 0.6))
            if lbl not in settings.intent_labels:
                lbl = "rag"
            return IntentResult(lbl, max(0.0, min(conf, 1.0)), "llm")
        except Exception as e:
            log.warning("LLM intent failed: %s", e)
            return None

    # ---------------- fusion ----------------
    def classify(self, text: str) -> IntentResult:
        rules_res = self._rules(text)

        if self.mode == "rules":
            return rules_res

        llm_res = self._llm_predict(text) if self.mode in ("llm", "hybrid") else None
        if llm_res is None:
            return rules_res

        # fusion: take higher confidence; if close, prefer llm
        if llm_res.confidence >= rules_res.confidence + 0.05:
            out = llm_res
        elif rules_res.confidence >= llm_res.confidence + 0.05:
            out = rules_res
        else:
            out = llm_res  # tie-break to LLM

        # gate very low confidence → fall back to rules
        if out.confidence < settings.intent_min_confidence:
            return rules_res
        return out
