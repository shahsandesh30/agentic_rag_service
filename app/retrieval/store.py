# app/retrieval/store.py
"""
Database operations and utilities for retrieval system.
"""

from __future__ import annotations

import json
import logging
import sqlite3

import numpy as np

logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_BATCH_SIZE = 999
DEFAULT_EMBEDDING_DIM = 1


def connect(db_path: str) -> sqlite3.Connection:
    """
    Create a database connection with row factory enabled.

    Args:
        db_path: Path to the SQLite database file

    Returns:
        SQLite connection object with row factory

    Raises:
        sqlite3.Error: If database connection fails
    """
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        logger.debug(f"Connected to database: {db_path}")
        return conn
    except sqlite3.Error as e:
        logger.error(f"Failed to connect to database {db_path}: {e}")
        raise


def load_embeddings(conn: sqlite3.Connection, model: str) -> tuple[list[str], np.ndarray]:
    """
    Load embeddings from database for a specific model.

    Args:
        conn: Database connection
        model: Model name to load embeddings for

    Returns:
        Tuple of (chunk_ids, embedding_matrix) where matrix shape is (N, dim)

    Raises:
        sqlite3.Error: If database query fails
        ValueError: If embeddings have inconsistent dimensions
    """
    try:
        rows = conn.execute(
            "SELECT chunk_id, dim, vec FROM embeddings WHERE model = ?", (model,)
        ).fetchall()

        if not rows:
            logger.warning(f"No embeddings found for model: {model}")
            return [], np.zeros((0, DEFAULT_EMBEDDING_DIM), dtype=np.float32)

        # Validate consistent dimensions
        dim = rows[0]["dim"]
        for row in rows:
            if row["dim"] != dim:
                raise ValueError(
                    f"Inconsistent embedding dimensions: expected {dim}, got {row['dim']}"
                )

        # Build matrix
        ids: list[str] = []
        vecs: list[np.ndarray] = []

        for row in rows:
            ids.append(row["chunk_id"])
            v = np.frombuffer(row["vec"], dtype=np.float32, count=dim)
            vecs.append(v)

        mat = np.vstack(vecs).astype(np.float32, copy=False)
        logger.info(f"Loaded {len(ids)} embeddings for model {model} with dimension {dim}")
        return ids, mat

    except sqlite3.Error as e:
        logger.error(f"Database error loading embeddings for model {model}: {e}")
        raise


def fetch_chunk_texts(
    conn: sqlite3.Connection, chunk_ids: list[str], batch_size: int = DEFAULT_BATCH_SIZE
) -> dict[str, dict]:
    """
    Fetch text and metadata for a list of chunk IDs.

    Args:
        conn: Database connection
        chunk_ids: List of chunk IDs to fetch
        batch_size: Batch size for database queries to avoid parameter limits

    Returns:
        Dictionary mapping chunk_id -> {text, section, source, path}

    Raises:
        sqlite3.Error: If database query fails
    """
    out: dict[str, dict] = {}
    if not chunk_ids:
        return out

    try:
        # Process in batches to avoid SQLite parameter limits
        for i in range(0, len(chunk_ids), batch_size):
            batch = chunk_ids[i : i + batch_size]
            placeholders = ",".join("?" * len(batch))
            query = f"SELECT c.id, c.text, c.section, c.meta_json FROM chunks c WHERE c.id IN ({placeholders})"

            for row in conn.execute(query, batch):
                meta = {}
                try:
                    meta = json.loads(row["meta_json"] or "{}")
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Failed to parse meta_json for chunk {row['id']}: {e}")
                    meta = {}

                out[row["id"]] = {
                    "text": row["text"] or "",
                    "section": row["section"] or "",
                    "source": meta.get("source", ""),
                    "path": meta.get("path", ""),
                }

        logger.debug(f"Fetched metadata for {len(out)} chunks")
        return out

    except sqlite3.Error as e:
        logger.error(f"Database error fetching chunk texts: {e}")
        raise


def load_all_chunks(conn: sqlite3.Connection) -> tuple[list[str], list[str]]:
    """
    Load all chunk IDs and texts from the database (used for BM25 indexing).

    Args:
        conn: Database connection

    Returns:
        Tuple of (chunk_ids, texts) lists

    Raises:
        sqlite3.Error: If database query fails
    """
    try:
        rows = conn.execute("SELECT id, text FROM chunks").fetchall()
        ids, texts = [], []

        for row in rows:
            ids.append(row["id"])
            texts.append(row["text"] or "")

        logger.info(f"Loaded {len(ids)} chunks for BM25 indexing")
        return ids, texts

    except sqlite3.Error as e:
        logger.error(f"Database error loading all chunks: {e}")
        raise


def fetch_full_chunks(
    conn: sqlite3.Connection, chunk_ids: list[str], batch_size: int = DEFAULT_BATCH_SIZE
) -> dict[str, str]:
    """
    Fetch full text content for a list of chunk IDs.

    Args:
        conn: Database connection
        chunk_ids: List of chunk IDs to fetch
        batch_size: Batch size for database queries

    Returns:
        Dictionary mapping chunk_id -> text content

    Raises:
        sqlite3.Error: If database query fails
    """
    out: dict[str, str] = {}
    if not chunk_ids:
        return out

    try:
        for i in range(0, len(chunk_ids), batch_size):
            batch = chunk_ids[i : i + batch_size]
            placeholders = ",".join("?" * len(batch))
            query = f"SELECT id, text FROM chunks WHERE id IN ({placeholders})"

            for row in conn.execute(query, batch):
                out[row["id"]] = row["text"] or ""

        logger.debug(f"Fetched full text for {len(out)} chunks")
        return out

    except sqlite3.Error as e:
        logger.error(f"Database error fetching full chunks: {e}")
        raise
