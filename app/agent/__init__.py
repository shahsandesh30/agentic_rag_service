# app/agent/__init__.py
"""
Agent system for intelligent question answering and routing.

This module provides a comprehensive agent system including:
- Intent classification and routing
- Question rewriting and research
- Answer generation with multiple modes
- Safety and compliance checking
- Graph-based execution workflow
- Command-line interface

Main components:
- Router: Intent classification and routing
- Researcher: Question rewriting and research
- Answer: Answer generation with multiple strategies
- Compliance: Safety and compliance checking
- Graph: Workflow orchestration and execution
- CLI: Command-line interface for agent interaction
"""

from .answer import answer_with_rewrites
from .compliance import AdvancedSafetyChecker, RegexSafetyChecker, check
from .graph import build_graph, run_agent
from .interfaces import (
    AgentGraph,
    AgentNode,
    AnswerGenerator,
    IntentClassifier,
    QuestionRewriter,
    SafetyChecker,
    WebSearcher,
)
from .researcher import LLMQuestionRewriter, SemanticQuestionRewriter, make_rewrites
from .router import classify_intent
from .types import (
    AgentConfig,
    AgentMetrics,
    AgentState,
    AnswerMode,
    AnswerPayload,
    Citation,
    Intent,
    ProcessingStatus,
    SafetyInfo,
    SafetyLevel,
    TraceEntry,
)

__all__ = [
    # Main functions
    "router",
    "make_rewrites",
    "answer_with_rewrites",
    "check",
    "build_graph",
    "run_agent",
    # Classifiers
    "classify_intent",
    # Rewriters
    "LLMQuestionRewriter",
    "SemanticQuestionRewriter",
    # Safety checkers
    "RegexSafetyChecker",
    "AdvancedSafetyChecker",
    # Types and enums
    "Intent",
    "AnswerMode",
    "SafetyLevel",
    "ProcessingStatus",
    "Citation",
    "SafetyInfo",
    "AnswerPayload",
    "TraceEntry",
    "AgentState",
    "AgentMetrics",
    "AgentConfig",
    # Interfaces
    "IntentClassifier",
    "QuestionRewriter",
    "AnswerGenerator",
    "SafetyChecker",
    "WebSearcher",
    "AgentNode",
    "AgentGraph",
]
