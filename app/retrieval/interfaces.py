# app/retrieval/interfaces.py
"""
Base interfaces and protocols for retrieval components.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Protocol


class SearchResult(Dict[str, Any]):
    """Type alias for search result structure."""
    pass


class BaseSearcher(ABC):
    """Abstract base class for all searcher implementations."""
    
    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Search for relevant documents.
        
        Args:
            query: The search query string
            top_k: Maximum number of results to return
            
        Returns:
            List of search results with metadata
        """
        pass


class RerankerProtocol(Protocol):
    """Protocol for reranker implementations."""
    
    def rerank(self, query: str, hits: List[SearchResult], top_k: int = 10) -> List[SearchResult]:
        """
        Rerank search results based on query relevance.
        
        Args:
            query: The search query string
            hits: List of search results to rerank
            top_k: Maximum number of results to return after reranking
            
        Returns:
            Reranked list of search results
        """
        ...


class DatabaseConnection(Protocol):
    """Protocol for database connection objects."""
    
    def execute(self, query: str, params: tuple = ()) -> Any:
        """Execute a database query."""
        ...
    
    def close(self) -> None:
        """Close the database connection."""
        ...
