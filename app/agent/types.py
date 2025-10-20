# app/agent/types.py
"""
Type definitions for the agent system.

This module provides comprehensive type definitions for:
- Agent state management
- Intent classification
- Answer payloads
- Trace information
- Configuration structures
"""
from __future__ import annotations
from typing import List, Literal, TypedDict, Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum

# Core type aliases
Intent = Literal["rag", "web", "chitchat"]
AnswerMode = Literal["multi", "merge"]
SafetyLevel = Literal["safe", "warning", "blocked"]

# ----------------------------
# Agent configuration
# ----------------------------
@dataclass
class AgentConfig:
    """
    Runtime configuration for the agent and its classifiers.
    Keep defaults here self-contained to avoid importing from router.py.
    """
    min_query_length: int = 3
    similarity_threshold: float = 0.92
    llm_max_tokens: int = 16
    llm_temperature: float = 0.0
    use_llm_fallback: bool = False
    log_level: str = "INFO"


class ProcessingStatus(Enum):
    """Status of agent processing steps."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Citation:
    """Citation information for answers."""
    source: Optional[str] = None
    path: Optional[str] = None
    section: Optional[str] = None
    url: Optional[str] = None
    title: Optional[str] = None


@dataclass
class SafetyInfo:
    """Safety information for answers."""
    blocked: bool = False
    reason: Optional[str] = None
    level: SafetyLevel = "safe"
    confidence_penalty: float = 0.0


@dataclass
class AnswerPayload:
    """Structured answer payload."""
    answer: str
    citations: List[Citation]
    confidence: float
    safety: SafetyInfo
    mode: Optional[str] = None
    rewrite: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "answer": self.answer,
            "citations": [c.__dict__ for c in self.citations],
            "confidence": self.confidence,
            "safety": self.safety.__dict__,
            "mode": self.mode,
            "rewrite": self.rewrite,
            "metadata": self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnswerPayload":
        """Create from dictionary format."""
        citations = [Citation(**c) for c in data.get("citations", [])]
        safety = SafetyInfo(**data.get("safety", {}))
        return cls(
            answer=data.get("answer", ""),
            citations=citations,
            confidence=data.get("confidence", 0.0),
            safety=safety,
            mode=data.get("mode"),
            rewrite=data.get("rewrite"),
            metadata=data.get("metadata")
        )


@dataclass
class TraceEntry:
    """Trace entry for debugging and monitoring."""
    node: str
    timestamp: float
    status: ProcessingStatus
    duration: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class AgentState(TypedDict, total=False):
    """Agent state structure for graph execution."""
    question: str
    intent: Optional[Intent]
    rewrites: List[str]
    answers: List[Dict[str, Any]]  # List of AnswerPayload.dict()
    best: Optional[Dict[str, Any]]  # Chosen AnswerPayload.dict()
    trace: List[TraceEntry]
    web_results: Optional[List[Dict[str, Any]]]
    status: Optional[ProcessingStatus]
    error: Optional[str]
    metadata: Optional[Dict[str, Any]]


@dataclass
class AgentMetrics:
    """Metrics for agent performance monitoring."""
    total_duration: float
    node_durations: Dict[str, float]
    retrieval_count: int
    generation_count: int
    confidence_scores: List[float]
    safety_checks: int
    errors: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "total_duration": self.total_duration,
            "node_durations": self.node_durations,
            "retrieval_count": self.retrieval_count,
            "generation_count": self.generation_count,
            "confidence_scores": self.confidence_scores,
            "safety_checks": self.safety_checks,
            "errors": self.errors
        }
