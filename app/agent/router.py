# app/agent/router.py
"""
Intent classification and routing for agent system.

This module provides intent classification functionality to route user questions
to appropriate processing paths (RAG, web search, or chitchat).
"""
from __future__ import annotations
import re
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from app.llm.groq_gen import GroqGenerator
from .interfaces import IntentClassifier, Intent
from .types import AgentConfig

logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_MIN_QUERY_LENGTH = 3
DEFAULT_SIMILARITY_THRESHOLD = 0.92
DEFAULT_MAX_TOKENS = 16
DEFAULT_TEMPERATURE = 0.0

# Compiled regex patterns for performance
RAG_LEGAL_HINTS = re.compile(
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

class RegexIntentClassifier(IntentClassifier):
    """Regex-based intent classifier with optional LLM fallback."""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.min_query_length = DEFAULT_MIN_QUERY_LENGTH
        
    def classify(self, question: str) -> Intent:
        """
        Classify intent using regex patterns and heuristics.
        
        Args:
            question: User question to classify
            
        Returns:
            Classified intent
            
        Raises:
            ValueError: If question is empty or invalid
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
            
        q = question.strip()
        logger.debug(f"Classifying question: {q[:100]}...")
        
        try:
            # Rule 1: Check for chitchat patterns first
            if CHITCHAT_HINTS.search(q):
                logger.debug("Chitchat pattern detected")
                return "chitchat"
            
            # Rule 2: Very short queries are usually chitchat (unless legal)
            if len(q.split()) <= self.min_query_length and not RAG_LEGAL_HINTS.search(q):
                logger.debug("Short query without legal hints -> chitchat")
                return "chitchat"
            
            # Rule 3: Legal hints detected -> RAG
            if RAG_LEGAL_HINTS.search(q):
                logger.debug("Legal pattern detected -> RAG")
                return "rag"
            
            # Rule 4: Web search hints
            if WEB_HINTS.search(q):
                logger.debug("Web search pattern detected")
                return "web"
            
            # Rule 5: Default to RAG for longer queries
            if len(q.split()) > self.min_query_length:
                logger.debug("Long query without specific hints -> RAG")
                return "rag"
            
            # Rule 6: Default fallback
            logger.debug("No specific patterns -> chitchat")
            return "chitchat"
            
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            # Safe fallback
            return "chitchat"


class LLMIntentClassifier(IntentClassifier):
    """LLM-based intent classifier with regex fallback."""
    
    def __init__(self, generator: GroqGenerator, config: Optional[AgentConfig] = None):
        self.generator = generator
        self.config = config or AgentConfig()
        self.regex_classifier = RegexIntentClassifier(config)
        
    def classify(self, question: str) -> Intent:
        """
        Classify intent using LLM with regex fallback.
        
        Args:
            question: User question to classify
            
        Returns:
            Classified intent
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
            
        q = question.strip()
        logger.debug(f"LLM classifying question: {q[:100]}...")
        
        try:
            # First try regex classification
            regex_intent = self.regex_classifier.classify(question)
            
            # If regex is confident, use it
            if regex_intent in ["rag", "web"]:
                logger.debug(f"Regex classification confident: {regex_intent}")
                return regex_intent
            
            # For ambiguous cases, use LLM
            logger.debug("Using LLM for ambiguous classification")
            
            prompt = (
                f"Classify the user question as 'rag' (legal/document QA), 'web' (web search), or 'chitchat'. "
                f"Output exactly one word: rag, web, or chitchat.\nQ: {q}"
            )
            
            response = self.generator.generate(
                prompt=prompt,
                contexts=[],
                system="You are a strict classifier. Only respond with 'rag', 'web', or 'chitchat'.",
                max_tokens=DEFAULT_MAX_TOKENS,
                temperature=DEFAULT_TEMPERATURE
            ).strip().lower()
            
            # Parse LLM response
            if "rag" in response:
                intent = "rag"
            elif "web" in response:
                intent = "web"
            else:
                intent = "chitchat"
            
            logger.debug(f"LLM classification: {intent}")
            return intent
            
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            # Fallback to regex
            logger.debug("Falling back to regex classification")
            return self.regex_classifier.classify(question)


def route(
    question: str,
    allow_llm_fallback: bool = False,
    generator: Optional[GroqGenerator] = None,
    config: Optional[AgentConfig] = None
) -> Intent:
    """
    Route a question to appropriate processing path.
    
    Args:
        question: User question to route
        allow_llm_fallback: Whether to use LLM for ambiguous cases
        generator: LLM generator for fallback classification
        config: Agent configuration
        
    Returns:
        Classified intent
        
    Raises:
        ValueError: If LLM fallback is requested but generator is None
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
    
    if allow_llm_fallback and generator is None:
        raise ValueError("GroqGenerator required when allow_llm_fallback=True")
    
    try:
        if allow_llm_fallback:
            classifier = LLMIntentClassifier(generator, config)
        else:
            classifier = RegexIntentClassifier(config)
        
        intent = classifier.classify(question)
        logger.info(f"Question routed to: {intent}")
        return intent
        
    except Exception as e:
        logger.error(f"Routing failed: {e}")
        # Safe fallback
        return "chitchat"
