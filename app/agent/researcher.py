# app/agent/researcher.py
"""
Question rewriting and research functionality for agent system.

This module provides question rewriting capabilities to improve retrieval
performance by generating alternative formulations of user questions.
"""
from __future__ import annotations
import logging
from typing import List, Optional, Dict, Any, Tuple
import numpy as np

from app.llm.groq_gen import GroqGenerator
from app.retrieval.faiss_sqlite import FaissSqliteSearcher
from .interfaces import QuestionRewriter
from .types import AgentConfig

logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_MAX_REWRITES = 3
DEFAULT_SIMILARITY_THRESHOLD = 0.92
DEFAULT_MAX_TOKENS = 128
DEFAULT_TEMPERATURE = 0.3
DEFAULT_TOP_K = 3

class LLMQuestionRewriter(QuestionRewriter):
    """LLM-based question rewriter with semantic deduplication."""
    
    def __init__(self, generator: GroqGenerator, config: Optional[AgentConfig] = None):
        self.generator = generator
        self.config = config or AgentConfig()
        
    def rewrite(self, question: str, max_rewrites: int = DEFAULT_MAX_REWRITES) -> List[str]:
        """
        Generate alternative rewrites of a question using LLM.
        
        Args:
            question: Original question to rewrite
            max_rewrites: Maximum number of rewrites to generate
            
        Returns:
            List of rewritten questions including original
            
        Raises:
            ValueError: If question is empty or invalid
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
            
        logger.debug(f"Generating rewrites for: {question[:100]}...")
        
        try:
            # Generate rewrites using LLM
            rewrites = self._generate_llm_rewrites(question, max_rewrites)
            
            # Clean and deduplicate
            cleaned_rewrites = self._clean_and_deduplicate(rewrites)
            
            # Limit to requested number
            final_rewrites = cleaned_rewrites[:max_rewrites + 1]  # +1 for original
            
            logger.info(f"Generated {len(final_rewrites)} rewrites")
            return final_rewrites
            
        except Exception as e:
            logger.error(f"Question rewriting failed: {e}")
            # Return original question as fallback
            return [question]
    
    def _generate_llm_rewrites(self, question: str, max_rewrites: int) -> List[str]:
        """Generate rewrites using LLM."""
        prompt = (
            f"Rewrite the following legal question into up to {max_rewrites} retrieval-friendly queries. "
            "Use synonyms, alternative legal terms, or related formulations. "
            "Each rewrite must be short and standalone. "
            "Return each rewrite on a new line, without numbering."
            f"\nOriginal: {question}"
        )
        
        text = self.generator.generate(
            prompt=prompt,
            contexts=[],
            system="You produce terse search queries optimized for Australian legal document retrieval.",
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
        )
        
        # Parse response
        lines = [ln.strip("- •\t ").strip() for ln in text.splitlines() if ln.strip()]
        return lines
    
    def _clean_and_deduplicate(self, rewrites: List[str]) -> List[str]:
        """Clean and deduplicate rewrites."""
        uniq: List[str] = []
        for rewrite in rewrites:
            if rewrite and rewrite.lower() not in {u.lower() for u in uniq}:
                uniq.append(rewrite)
        return uniq


class SemanticQuestionRewriter(QuestionRewriter):
    """Question rewriter with semantic deduplication and retrieval filtering."""
    
    def __init__(
        self, 
        generator: GroqGenerator, 
        searcher: FaissSqliteSearcher,
        config: Optional[AgentConfig] = None
    ):
        self.generator = generator
        self.searcher = searcher
        self.config = config or AgentConfig()
        self.llm_rewriter = LLMQuestionRewriter(generator, config)
        
    def rewrite(self, question: str, max_rewrites: int = DEFAULT_MAX_REWRITES) -> List[str]:
        """
        Generate rewrites with semantic deduplication and retrieval filtering.
        
        Args:
            question: Original question to rewrite
            max_rewrites: Maximum number of rewrites to generate
            
        Returns:
            List of rewritten questions including original
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")
            
        logger.debug(f"Generating semantic rewrites for: {question[:100]}...")
        
        try:
            # Step 1: Generate basic rewrites
            basic_rewrites = self.llm_rewriter.rewrite(question, max_rewrites)
            
            # Step 2: Semantic deduplication
            deduplicated = self._semantic_deduplication([question] + basic_rewrites)
            
            # Step 3: Retrieval-aware filtering
            filtered = self._retrieval_filtering(deduplicated)
            
            # Step 4: Limit final results
            final_rewrites = filtered[:max_rewrites + 1]
            
            logger.info(f"Generated {len(final_rewrites)} semantic rewrites")
            return final_rewrites
            
        except Exception as e:
            logger.error(f"Semantic rewriting failed: {e}")
            # Fallback to basic rewriting
            return self.llm_rewriter.rewrite(question, max_rewrites)
    
    def _semantic_deduplication(self, queries: List[str]) -> List[str]:
        """Remove semantically similar queries."""
        if len(queries) <= 1:
            return queries
            
        try:
            # Embed all queries
            vecs = self.searcher.embedder.encode(queries)
            
            keep = [0]  # Always keep the original
            for i in range(1, len(queries)):
                # Calculate similarity to already kept queries
                similarities = []
                for j in keep:
                    sim = np.dot(vecs[i], vecs[j]) / (
                        np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j]) + 1e-12
                    )
                    similarities.append(sim)
                
                max_similarity = max(similarities) if similarities else 0.0
                
                # Keep if not too similar to existing queries
                if max_similarity < DEFAULT_SIMILARITY_THRESHOLD:
                    keep.append(i)
            
            return [queries[i] for i in keep]
            
        except Exception as e:
            logger.error(f"Semantic deduplication failed: {e}")
            return queries
    
    def _retrieval_filtering(self, queries: List[str]) -> List[str]:
        """Filter queries based on retrieval performance."""
        if not queries:
            return queries
            
        try:
            # Encode all queries
            vecs = self.searcher.embedder.encode(queries).astype("float32")
            
            # Normalize for cosine similarity
            vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
            
            # Perform batched FAISS search
            D, I = self.searcher.index.search(vecs, top_k=DEFAULT_TOP_K)
            
            # Score queries by best retrieval score
            scored = []
            for idx, query in enumerate(queries):
                best_score = max(D[idx]) if len(D[idx]) > 0 else 0.0
                scored.append((query, float(best_score)))
            
            # Sort by retrieval strength
            scored.sort(key=lambda x: x[1], reverse=True)
            return [query for query, _ in scored]
            
        except Exception as e:
            logger.error(f"Retrieval filtering failed: {e}")
            return queries


def make_rewrites(
    question: str,
    max_rewrites: int = DEFAULT_MAX_REWRITES,
    generator: Optional[GroqGenerator] = None,
    searcher: Optional[FaissSqliteSearcher] = None,
    config: Optional[AgentConfig] = None
) -> List[str]:
    """
    Generate alternative rewrites of a user question.
    
    Args:
        question: Original question to rewrite
        max_rewrites: Maximum number of rewrites to generate
        generator: LLM generator for rewriting
        searcher: Optional searcher for semantic filtering
        config: Agent configuration
        
    Returns:
        List of rewritten questions including original
        
    Raises:
        ValueError: If required parameters are missing
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty")
        
    if generator is None:
        raise ValueError("GroqGenerator instance must be provided")
    
    try:
        if searcher is not None:
            # Use semantic rewriter with retrieval filtering
            rewriter = SemanticQuestionRewriter(generator, searcher, config)
        else:
            # Use basic LLM rewriter
            rewriter = LLMQuestionRewriter(generator, config)
        
        return rewriter.rewrite(question, max_rewrites)
        
    except Exception as e:
        logger.error(f"Question rewriting failed: {e}")
        # Return original question as fallback
        return [question]
