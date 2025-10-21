# app/agent/interfaces.py
"""
Base interfaces and protocols for agent components.

This module defines the core interfaces for the agent system including:
- Intent classification
- Question rewriting
- Answer generation
- Safety checking
- State management
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Protocol

# Type aliases for better code clarity
Intent = Literal["rag", "web", "chitchat"]
AnswerPayload = dict[str, Any]
TraceEntry = dict[str, Any]
WebResult = dict[str, Any]


@dataclass
class AgentConfig:
    """Configuration for agent components."""

    max_rewrites: int = 3
    confidence_gate: float = 0.65
    top_k_context: int = 8
    similarity_threshold: float = 0.92
    max_tokens: int = 128
    temperature: float = 0.3
    enable_llm_fallback: bool = False
    enable_semantic_dedup: bool = True
    enable_retrieval_filtering: bool = True


class IntentClassifier(ABC):
    """Abstract base class for intent classification."""

    @abstractmethod
    def classify(self, question: str) -> Intent:
        """
        Classify the intent of a user question.

        Args:
            question: User question to classify

        Returns:
            Classified intent
        """
        pass


class QuestionRewriter(ABC):
    """Abstract base class for question rewriting."""

    @abstractmethod
    def rewrite(self, question: str, max_rewrites: int = 3) -> list[str]:
        """
        Generate alternative rewrites of a question.

        Args:
            question: Original question
            max_rewrites: Maximum number of rewrites to generate

        Returns:
            List of rewritten questions
        """
        pass


class AnswerGenerator(ABC):
    """Abstract base class for answer generation."""

    @abstractmethod
    def generate_answer(
        self, question: str, context: list[dict[str, Any]], mode: str = "merge"
    ) -> AnswerPayload:
        """
        Generate an answer for a question.

        Args:
            question: User question
            context: Retrieved context
            mode: Generation mode

        Returns:
            Answer payload with metadata
        """
        pass


class SafetyChecker(ABC):
    """Abstract base class for safety checking."""

    @abstractmethod
    def check_safety(self, payload: AnswerPayload) -> AnswerPayload:
        """
        Check answer payload for safety issues.

        Args:
            payload: Answer payload to check

        Returns:
            Modified payload with safety information
        """
        pass


class WebSearcher(ABC):
    """Abstract base class for web search."""

    @abstractmethod
    def search(self, query: str, num_results: int = 3) -> list[WebResult]:
        """
        Perform web search for a query.

        Args:
            query: Search query
            num_results: Number of results to return

        Returns:
            List of web search results
        """
        pass


class AgentState(Protocol):
    """Protocol for agent state management."""

    question: str
    intent: Intent | None
    rewrites: list[str]
    answers: list[AnswerPayload]
    best: AnswerPayload | None
    trace: list[TraceEntry]
    web_results: list[WebResult] | None


class AgentNode(ABC):
    """Abstract base class for agent graph nodes."""

    @abstractmethod
    def execute(self, state: AgentState) -> AgentState:
        """
        Execute the node logic.

        Args:
            state: Current agent state

        Returns:
            Updated agent state
        """
        pass


class AgentGraph(ABC):
    """Abstract base class for agent graph execution."""

    @abstractmethod
    def run(self, question: str) -> dict[str, Any]:
        """
        Run the agent graph with a question.

        Args:
            question: User question

        Returns:
            Final result with trace information
        """
        pass
