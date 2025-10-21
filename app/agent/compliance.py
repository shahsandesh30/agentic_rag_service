# app/agent/compliance.py
"""
Safety and compliance checking for agent responses.

This module provides comprehensive safety checking functionality to ensure
agent responses comply with safety guidelines and don't contain sensitive information.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .interfaces import SafetyChecker
from .types import AnswerPayload

logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_CONFIDENCE_PENALTY = 0.3
DEFAULT_MIN_CONFIDENCE = 0.2
DEFAULT_MAX_CONFIDENCE = 0.6

# Compiled regex patterns for performance
PII_PATTERNS = re.compile(
    r"\b(ssn|social security|passport|credit card|cvv|bank account|routing number|"
    r"driver's license|license number|tax id|ein|ssn|ss#)\b",
    re.I,
)

CREDENTIAL_PATTERNS = re.compile(
    r"(password|api[_-]?key|secret|token|auth|login|credential|private key)", re.I
)

HARMFUL_CONTENT_PATTERNS = re.compile(
    r"\b(violence|threat|harm|illegal|unlawful|dangerous|weapon|drug|"
    r"hate speech|discrimination|harassment)\b",
    re.I,
)

MEDICAL_PATTERNS = re.compile(
    r"\b(medical advice|diagnosis|treatment|prescription|medication|"
    r"health condition|symptom|doctor|physician)\b",
    re.I,
)


class SafetyViolation(Enum):
    """Types of safety violations."""

    PII = "pii"
    CREDENTIALS = "credentials"
    HARMFUL_CONTENT = "harmful_content"
    MEDICAL_ADVICE = "medical_advice"
    NONE = "none"


@dataclass
class SafetyCheckResult:
    """Result of a safety check."""

    violation: SafetyViolation
    confidence_penalty: float
    blocked: bool
    reason: str
    suggested_action: str


class RegexSafetyChecker(SafetyChecker):
    """Regex-based safety checker."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.confidence_penalty = self.config.get("confidence_penalty", DEFAULT_CONFIDENCE_PENALTY)
        self.min_confidence = self.config.get("min_confidence", DEFAULT_MIN_CONFIDENCE)
        self.max_confidence = self.config.get("max_confidence", DEFAULT_MAX_CONFIDENCE)

    def check_safety(self, payload: AnswerPayload) -> AnswerPayload:
        """
        Check answer payload for safety violations.

        Args:
            payload: Answer payload to check

        Returns:
            Modified payload with safety information
        """
        if not payload.answer:
            logger.debug("Empty answer, skipping safety check")
            return payload

        try:
            # Perform safety checks
            result = self._perform_safety_checks(payload.answer)

            # Apply safety modifications
            modified_payload = self._apply_safety_modifications(payload, result)

            logger.debug(f"Safety check completed: {result.violation.value}")
            return modified_payload

        except Exception as e:
            logger.error(f"Safety check failed: {e}")
            # Return original payload with error flag
            payload.safety.error = str(e)
            return payload

    def _perform_safety_checks(self, text: str) -> SafetyCheckResult:
        """Perform comprehensive safety checks on text."""
        text_lower = text.lower()

        # Check for PII
        if PII_PATTERNS.search(text_lower):
            return SafetyCheckResult(
                violation=SafetyViolation.PII,
                confidence_penalty=0.8,
                blocked=True,
                reason="Contains potential personally identifiable information (PII).",
                suggested_action="Remove or redact PII information.",
            )

        # Check for harmful content
        if HARMFUL_CONTENT_PATTERNS.search(text_lower):
            return SafetyCheckResult(
                violation=SafetyViolation.HARMFUL_CONTENT,
                confidence_penalty=0.9,
                blocked=True,
                reason="Contains potentially harmful or inappropriate content.",
                suggested_action="Review and remove harmful content.",
            )

        # Check for medical advice
        if MEDICAL_PATTERNS.search(text_lower):
            return SafetyCheckResult(
                violation=SafetyViolation.MEDICAL_ADVICE,
                confidence_penalty=0.7,
                blocked=True,
                reason="Contains medical advice which should not be provided by AI.",
                suggested_action="Direct user to consult healthcare professionals.",
            )

        # Check for credentials (warning, not blocking)
        if CREDENTIAL_PATTERNS.search(text_lower):
            return SafetyCheckResult(
                violation=SafetyViolation.CREDENTIALS,
                confidence_penalty=self.confidence_penalty,
                blocked=False,
                reason="Contains credential-related information; confidence reduced.",
                suggested_action="Review for sensitive credential information.",
            )

        # No violations found
        return SafetyCheckResult(
            violation=SafetyViolation.NONE,
            confidence_penalty=0.0,
            blocked=False,
            reason="No safety violations detected.",
            suggested_action="Content appears safe.",
        )

    def _apply_safety_modifications(
        self, payload: AnswerPayload, result: SafetyCheckResult
    ) -> AnswerPayload:
        """Apply safety modifications to payload."""
        # Update safety information
        payload.safety.blocked = result.blocked
        payload.safety.reason = result.reason
        payload.safety.confidence_penalty = result.confidence_penalty

        if result.violation == SafetyViolation.NONE:
            payload.safety.level = "safe"
        elif result.blocked:
            payload.safety.level = "blocked"
        else:
            payload.safety.level = "warning"

        # Apply confidence penalty
        if result.confidence_penalty > 0:
            new_confidence = max(
                self.min_confidence,
                min(self.max_confidence, payload.confidence - result.confidence_penalty),
            )
            payload.confidence = new_confidence
            logger.debug(
                f"Confidence reduced from {payload.confidence + result.confidence_penalty} to {new_confidence}"
            )

        # Block content if necessary
        if result.blocked:
            payload.answer = "I can't provide that information due to safety concerns."
            payload.citations = []
            logger.warning(f"Content blocked due to: {result.reason}")

        return payload


class AdvancedSafetyChecker(SafetyChecker):
    """Advanced safety checker with multiple validation layers."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.regex_checker = RegexSafetyChecker(config)
        self.enable_llm_check = self.config.get("enable_llm_check", False)
        self.llm_generator = self.config.get("llm_generator")

    def check_safety(self, payload: AnswerPayload) -> AnswerPayload:
        """
        Perform advanced safety checking with multiple validation layers.

        Args:
            payload: Answer payload to check

        Returns:
            Modified payload with safety information
        """
        if not payload.answer:
            return payload

        try:
            # Step 1: Regex-based checking
            payload = self.regex_checker.check_safety(payload)

            # Step 2: Additional checks if not already blocked
            if not payload.safety.blocked:
                payload = self._perform_additional_checks(payload)

            # Step 3: LLM-based checking (if enabled)
            if self.enable_llm_check and self.llm_generator and not payload.safety.blocked:
                payload = self._perform_llm_safety_check(payload)

            logger.info(f"Advanced safety check completed: {payload.safety.level}")
            return payload

        except Exception as e:
            logger.error(f"Advanced safety check failed: {e}")
            payload.safety.error = str(e)
            return payload

    def _perform_additional_checks(self, payload: AnswerPayload) -> AnswerPayload:
        """Perform additional safety checks beyond regex."""
        # Add length-based checks
        if len(payload.answer) > 10000:  # Very long responses might be problematic
            payload.safety.level = "warning"
            payload.safety.reason = "Response is unusually long."
            payload.confidence = max(0.3, payload.confidence - 0.1)

        # Add citation checks
        if not payload.citations and payload.confidence > 0.8:
            payload.safety.level = "warning"
            payload.safety.reason = "High confidence without citations."
            payload.confidence = max(0.5, payload.confidence - 0.2)

        return payload

    def _perform_llm_safety_check(self, payload: AnswerPayload) -> AnswerPayload:
        """Perform LLM-based safety checking."""
        try:
            prompt = (
                f"Review this response for safety issues. Check for: "
                f"1. Personally identifiable information (PII)\n"
                f"2. Harmful or inappropriate content\n"
                f"3. Medical advice\n"
                f"4. Credential information\n"
                f"5. Legal advice beyond general information\n\n"
                f"Response: {payload.answer}\n\n"
                f"Rate safety from 1-10 (10 being safest) and explain any concerns."
            )

            # This would require LLM integration
            # For now, we'll skip this step
            logger.debug("LLM safety check skipped (not implemented)")
            return payload

        except Exception as e:
            logger.error(f"LLM safety check failed: {e}")
            return payload


def check(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Legacy safety check function for backward compatibility.

    Args:
        payload: Answer payload dictionary

    Returns:
        Modified payload dictionary
    """
    try:
        # Convert dict to AnswerPayload
        answer_payload = AnswerPayload.from_dict(payload)

        # Perform safety check
        checker = RegexSafetyChecker()
        result = checker.check_safety(answer_payload)

        # Convert back to dict
        return result.to_dict()

    except Exception as e:
        logger.error(f"Legacy safety check failed: {e}")
        # Return original payload with error
        payload.setdefault("safety", {})
        payload["safety"]["error"] = str(e)
        return payload
