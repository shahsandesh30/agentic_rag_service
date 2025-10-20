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

from .router import (
    route,
    RegexIntentClassifier,
    LLMIntentClassifier
)
from .researcher import (
    make_rewrites,
    LLMQuestionRewriter,
    SemanticQuestionRewriter
)
from .answer import answer_with_rewrites
from .compliance import (
    check,
    RegexSafetyChecker,
    AdvancedSafetyChecker
)
from .graph import (
    build_graph,
    run_agent
)
from .types import (
    Intent,
    AnswerMode,
    SafetyLevel,
    ProcessingStatus,
    Citation,
    SafetyInfo,
    AnswerPayload,
    TraceEntry,
    AgentState,
    AgentMetrics,
    AgentConfig
)
from .interfaces import (
    IntentClassifier,
    QuestionRewriter,
    AnswerGenerator,
    SafetyChecker,
    WebSearcher,
    AgentNode,
    AgentGraph
)

__all__ = [
    # Main functions
    "route",
    "make_rewrites", 
    "answer_with_rewrites",
    "check",
    "build_graph",
    "run_agent",
    
    # Classifiers
    "RegexIntentClassifier",
    "LLMIntentClassifier",
    
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
