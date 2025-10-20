# app/retrieval/faiss_sqlite.py
"""
FAISS-based vector search implementation with SQLite metadata storage.
"""
from __future__ import annotations
import os
import logging
from typing import List, Dict, Optional
import faiss
import numpy as np

from app.embed.model import Embedder
from app.corpus.schema import (
    connect,
    chunk_ids_for_faiss_ids,
    fetch_chunk_texts
)
from .interfaces import BaseSearcher, SearchResult
from .store import connect as db_connect

logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_INDEX_DIR = "faiss_index"
DEFAULT_INDEX_FILE = os.path.join(DEFAULT_INDEX_DIR, "index.faiss")
DEFAULT_DB_PATH = "rag_local.db"
DEFAULT_TOP_K = 5
DEFAULT_TEXT_PREVIEW_LENGTH = 800


class FaissSqliteSearcher(BaseSearcher):
    """
    FAISS-based vector searcher with SQLite metadata storage.
    
    Combines FAISS for efficient vector similarity search with SQLite for
    storing and retrieving document metadata and text content.
    """
    
    def __init__(
        self, 
        embedder: Embedder, 
        db_path: str = DEFAULT_DB_PATH,
        index_file: str = DEFAULT_INDEX_FILE
    ):
        """
        Initialize the FAISS SQLite searcher.
        
        Args:
            embedder: Embedding model for query encoding
            db_path: Path to SQLite database file
            index_file: Path to FAISS index file
            
        Raises:
            FileNotFoundError: If FAISS index file doesn't exist
            Exception: If index loading fails
        """
        self.embedder = embedder
        self.db_path = db_path
        self.index_file = index_file
        self.index: Optional[faiss.IndexIDMap] = None
        self._load_faiss_index()

    def _load_faiss_index(self) -> None:
        """
        Load the FAISS index from disk.
        
        Raises:
            FileNotFoundError: If index file doesn't exist
            Exception: If index loading fails
        """
        if not os.path.exists(self.index_file):
            error_msg = f"FAISS index not found at {self.index_file}. Run ingest_and_index.py first."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
        try:
            logger.info(f"Loading FAISS index from {self.index_file}")
            index = faiss.read_index(self.index_file)

            # Ensure ID support for index.add_with_ids() and search with mapped IDs
            if not isinstance(index, faiss.IndexIDMap):
                index = faiss.IndexIDMap(index)

            self.index = index
            logger.info(f"Successfully loaded FAISS index with {index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Failed to load FAISS index from {self.index_file}: {e}")
            raise

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[SearchResult]:
        """
        Search for relevant documents using FAISS vector similarity.
        
        Args:
            query: The search query string
            top_k: Maximum number of results to return
            
        Returns:
            List of search results with metadata
            
        Raises:
            RuntimeError: If FAISS index is not loaded
            Exception: If search or database operations fail
        """
        if self.index is None:
            raise RuntimeError("FAISS index not loaded. Run ingest_and_index.py to build it.")

        if not query or not query.strip():
            logger.warning("Empty query provided for search")
            return []

        try:
            # Step 1: Embed and normalize the query
            query_vector = self._encode_query(query)
            
            # Step 2: Search in FAISS
            faiss_ids, scores = self._search_faiss(query_vector, top_k)
            
            if not faiss_ids:
                logger.info("No relevant FAISS results found")
                return []

            # Step 3: Fetch metadata from database
            hits = self._fetch_metadata(faiss_ids, scores)
            
            logger.debug(f"Found {len(hits)} search results for query")
            return hits
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            raise

    def _encode_query(self, query: str) -> np.ndarray:
        """
        Encode and normalize query for FAISS search.
        
        Args:
            query: Query string to encode
            
        Returns:
            Normalized query vector
        """
        try:
            # Encode the query
            q = self.embedder.encode([query]).astype("float32")
            
            # Normalize for cosine similarity
            q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
            
            return q
            
        except Exception as e:
            logger.error(f"Query encoding failed: {e}")
            raise

    def _search_faiss(self, query_vector: np.ndarray, top_k: int) -> tuple[List[int], List[float]]:
        """
        Perform FAISS vector search.
        
        Args:
            query_vector: Normalized query vector
            top_k: Number of results to retrieve
            
        Returns:
            Tuple of (faiss_ids, scores)
        """
        try:
            D, I = self.index.search(query_vector, top_k)
            faiss_ids = [int(x) for x in I[0] if x != -1]
            scores = [float(s) for s in D[0][:len(faiss_ids)]]
            
            return faiss_ids, scores
            
        except Exception as e:
            logger.error(f"FAISS search failed: {e}")
            raise

    def _fetch_metadata(self, faiss_ids: List[int], scores: List[float]) -> List[SearchResult]:
        """
        Fetch metadata for search results from database.
        
        Args:
            faiss_ids: List of FAISS IDs
            scores: List of similarity scores
            
        Returns:
            List of search results with metadata
        """
        try:
            # Connect to database and fetch metadata
            conn = db_connect(self.db_path)
            
            try:
                # Map FAISS IDs to chunk IDs
                id_map = chunk_ids_for_faiss_ids(conn, faiss_ids)
                chunk_ids = [id_map.get(fid, "") for fid in faiss_ids if id_map.get(fid)]
                
                if not chunk_ids:
                    logger.warning("No chunk IDs found for FAISS IDs")
                    return []
                
                # Fetch chunk metadata
                meta = fetch_chunk_texts(conn, chunk_ids)
                
                # Format results
                hits: List[SearchResult] = []
                for fid, score in zip(faiss_ids, scores):
                    cid = id_map.get(fid)
                    if not cid:
                        continue
                        
                    chunk_meta = meta.get(cid, {})
                    hits.append({
                        "chunk_id": cid,
                        "score": score,
                        "text": (chunk_meta.get("text") or "")[:DEFAULT_TEXT_PREVIEW_LENGTH],
                        "path": chunk_meta.get("path", ""),
                        "section": chunk_meta.get("section", ""),
                        "source": chunk_meta.get("source", ""),
                    })
                
                return hits
                
            finally:
                conn.close()
                
        except Exception as e:
            logger.error(f"Metadata fetching failed: {e}")
            raise
