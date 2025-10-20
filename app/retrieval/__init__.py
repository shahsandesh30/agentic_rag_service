# app/retrieval/__init__.py
"""
Retrieval module for document search and ranking.

This module provides various search implementations including:
- FAISS-based vector search with SQLite metadata storage
- Cross-encoder reranking for improved relevance
- Database utilities for chunk and embedding management

Main components:
- FaissSqliteSearcher: Vector similarity search using FAISS
- Reranker: Cross-encoder based result reranking
- Database utilities: Functions for loading embeddings and chunk metadata
"""

from .faiss_sqlite import FaissSqliteSearcher
from .rerank import Reranker
from .store import (
    connect,
    load_embeddings,
    fetch_chunk_texts,
    load_all_chunks,
    fetch_full_chunks
)
from .interfaces import BaseSearcher, SearchResult, RerankerProtocol

__all__ = [
    # Main searcher classes
    "FaissSqliteSearcher",
    "Reranker",
    
    # Database utilities
    "connect",
    "load_embeddings", 
    "fetch_chunk_texts",
    "load_all_chunks",
    "fetch_full_chunks",
    
    # Interfaces and types
    "BaseSearcher",
    "SearchResult", 
    "RerankerProtocol",
]
