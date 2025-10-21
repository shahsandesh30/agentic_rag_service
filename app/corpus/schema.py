# app/corpus/schema.py
"""
Database schema and operations for corpus management.

This module provides functions for:
- Database connection management
- Schema initialization
- Document and chunk operations
- Embedding management
- Conversation memory storage
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_DB_PATH = "rag_local.db"
DEFAULT_BATCH_SIZE = 900
DEFAULT_FAISS_ID_RANGE = 2**63


def connect(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Create a database connection with proper configuration.

    Args:
        db_path: Path to SQLite database file

    Returns:
        SQLite connection object with row factory

    Raises:
        sqlite3.Error: If database connection fails
        OSError: If database file cannot be created
    """
    try:
        # Ensure directory exists
        db_path_obj = Path(db_path)
        db_path_obj.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row

        # Enable foreign key constraints
        conn.execute("PRAGMA foreign_keys = ON")

        logger.info(f"Connected to database: {db_path}")
        return conn

    except sqlite3.Error as e:
        logger.error(f"Database connection failed for {db_path}: {e}")
        raise
    except OSError as e:
        logger.error(f"Failed to create database directory for {db_path}: {e}")
        raise


def init_db(conn: sqlite3.Connection) -> None:
    """
    Initialize database schema with all required tables and indexes.

    Args:
        conn: Database connection

    Raises:
        sqlite3.Error: If schema creation fails
    """
    try:
        logger.info("Initializing database schema")

        # Documents table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            path TEXT,
            source TEXT,
            sha256 TEXT,
            mime TEXT,
            n_pages INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        logger.debug("Created documents table")

        # Chunks table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            doc_id TEXT,
            ordinal INTEGER,
            text TEXT,
            n_chars INTEGER,
            start_char INTEGER,
            end_char INTEGER,
            section TEXT,
            meta_json TEXT,
            FOREIGN KEY(doc_id) REFERENCES documents(id)
        )
        """)
        logger.debug("Created chunks table")

        # Embeddings table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            chunk_id TEXT,
            model TEXT,
            dim INTEGER,
            vec BLOB,
            faiss_id INTEGER,
            PRIMARY KEY(chunk_id, model),
            FOREIGN KEY(chunk_id) REFERENCES chunks(id)
        )
        """)
        logger.debug("Created embeddings table")

        # Conversation memory table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,         -- 'user' or 'assistant'
            content TEXT NOT NULL,
            embedding BLOB,             -- optional: store embedding for semantic memory
            created_at REAL DEFAULT (strftime('%s','now'))
        )
        """)
        logger.debug("Created conversations table")

        # Create indexes for performance
        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_conversations_session
        ON conversations(session_id)
        """)

        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_doc_id
        ON chunks(doc_id)
        """)

        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_embeddings_chunk_id
        ON embeddings(chunk_id)
        """)

        conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_embeddings_faiss_id
        ON embeddings(faiss_id)
        """)

        logger.info("Database schema initialization completed")

    except sqlite3.Error as e:
        logger.error(f"Database schema initialization failed: {e}")
        raise


def upsert_document(
    conn: sqlite3.Connection,
    doc_id: str,
    path: str,
    source: str,
    sha256_hex: str,
    mime: str,
    n_pages: int,
) -> None:
    """
    Upsert document record into database.

    Args:
        conn: Database connection
        doc_id: Document identifier
        path: Document file path
        source: Document source label
        sha256_hex: SHA256 hash of document content
        mime: MIME type of document
        n_pages: Number of pages in document

    Raises:
        sqlite3.Error: If database operation fails
    """
    try:
        conn.execute(
            """
        INSERT INTO documents(id, path, source, sha256, mime, n_pages, created_at)
        VALUES(?,?,?,?,?,?,?)
        ON CONFLICT(id) DO UPDATE SET path=excluded.path, source=excluded.source, mime=excluded.mime, n_pages=excluded.n_pages
        """,
            (doc_id, path, source, sha256_hex, mime, n_pages, time.time()),
        )
        logger.debug(f"Upserted document: {doc_id}")
    except sqlite3.Error as e:
        logger.error(f"Failed to upsert document {doc_id}: {e}")
        raise


def upsert_chunks(conn: sqlite3.Connection, rows: Iterable[tuple]) -> None:
    """
    Upsert chunk records into database.

    Args:
        conn: Database connection
        rows: Iterable of chunk tuples (id, doc_id, ordinal, text, n_chars, start_char, end_char, section, meta_json)

    Raises:
        sqlite3.Error: If database operation fails
    """
    try:
        rows_list = list(rows)
        if not rows_list:
            logger.warning("No chunks to upsert")
            return

        conn.executemany(
            """
        INSERT INTO chunks(id, doc_id, ordinal, text, n_chars, start_char, end_char, section, meta_json)
        VALUES(?,?,?,?,?,?,?,?,?)
        ON CONFLICT(id) DO NOTHING
        """,
            rows_list,
        )
        logger.info(f"Upserted {len(rows_list)} chunks")
    except sqlite3.Error as e:
        logger.error(f"Failed to upsert chunks: {e}")
        raise


def upsert_embeddings(conn, rows: Iterable[tuple[str, str, int, bytes, int]]) -> None:
    """
    Insert full embedding row: (chunk_id, model, dim, vec, faiss_id)
    """
    conn.executemany(
        """
    INSERT INTO embeddings(chunk_id, model, dim, vec, faiss_id)
    VALUES(?,?,?,?,?)
    ON CONFLICT(chunk_id, model) DO UPDATE SET vec=excluded.vec, faiss_id=excluded.faiss_id, dim=excluded.dim
    """,
        list(rows),
    )


def fetch_chunk_texts(
    conn: sqlite3.Connection, chunk_ids: list[str], batch_size: int = DEFAULT_BATCH_SIZE
) -> dict[str, dict]:
    """
    Fetch chunk texts and metadata from database.

    Args:
        conn: Database connection
        chunk_ids: List of chunk IDs to fetch
        batch_size: Batch size for database queries

    Returns:
        Dictionary mapping chunk_id to metadata

    Raises:
        sqlite3.Error: If database operation fails
    """
    if not chunk_ids:
        logger.debug("No chunk IDs provided")
        return {}

    try:
        out: dict[str, dict] = {}

        for i in range(0, len(chunk_ids), batch_size):
            batch = chunk_ids[i : i + batch_size]
            placeholders = ",".join("?" * len(batch))
            query = f"SELECT id, text, section, meta_json, doc_id FROM chunks WHERE id IN ({placeholders})"

            for row in conn.execute(query, batch):
                try:
                    meta = json.loads(row["meta_json"] or "{}")
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Failed to parse meta_json for chunk {row['id']}: {e}")
                    meta = {}

                # Fetch document metadata
                doc = conn.execute(
                    "SELECT path, source FROM documents WHERE id=?", (row["doc_id"],)
                ).fetchone()

                out[row["id"]] = {
                    "text": row["text"] or "",
                    "section": row["section"] or "",
                    "path": doc["path"] if doc else "",
                    "source": doc["source"] if doc else meta.get("source", ""),
                }

        logger.debug(f"Fetched metadata for {len(out)} chunks")
        return out

    except sqlite3.Error as e:
        logger.error(f"Failed to fetch chunk texts: {e}")
        raise


def fetch_full_chunks(
    conn: sqlite3.Connection, chunk_ids: list[str], batch_size: int = DEFAULT_BATCH_SIZE
) -> dict[str, str]:
    """
    Fetch full text content for chunk IDs.

    Args:
        conn: Database connection
        chunk_ids: List of chunk IDs to fetch
        batch_size: Batch size for database queries

    Returns:
        Dictionary mapping chunk_id to text content

    Raises:
        sqlite3.Error: If database operation fails
    """
    if not chunk_ids:
        logger.debug("No chunk IDs provided")
        return {}

    try:
        out: dict[str, str] = {}

        for i in range(0, len(chunk_ids), batch_size):
            batch = chunk_ids[i : i + batch_size]
            placeholders = ",".join("?" * len(batch))
            query = f"SELECT id, text FROM chunks WHERE id IN ({placeholders})"

            for row in conn.execute(query, batch):
                out[row["id"]] = row["text"] or ""

        logger.debug(f"Fetched full text for {len(out)} chunks")
        return out

    except sqlite3.Error as e:
        logger.error(f"Failed to fetch full chunks: {e}")
        raise


def missing_embedding_chunk_ids(conn: sqlite3.Connection) -> list[tuple[str, int]]:
    """
    Find chunks that don't have embeddings yet.

    Args:
        conn: Database connection

    Returns:
        List of (chunk_id, dimension_placeholder) tuples

    Raises:
        sqlite3.Error: If database operation fails
    """
    try:
        query = """
        SELECT c.id FROM chunks c
        LEFT JOIN embeddings e ON e.chunk_id = c.id
        WHERE e.chunk_id IS NULL
        """

        results = [(row["id"], 0) for row in conn.execute(query).fetchall()]
        logger.debug(f"Found {len(results)} chunks missing embeddings")
        return results

    except sqlite3.Error as e:
        logger.error(f"Failed to find missing embedding chunks: {e}")
        raise


def assign_faiss_ids(conn: sqlite3.Connection, pairs: list[tuple[str, int, int, str]]) -> None:
    """
    Assign FAISS IDs to chunks with safety checks.

    Args:
        conn: Database connection
        pairs: List of (chunk_id, faiss_id, dim, model) tuples

    Raises:
        sqlite3.Error: If database operation fails
    """
    if not pairs:
        logger.warning("No FAISS ID pairs provided")
        return

    try:
        safe_pairs = []
        for cid, fid, dim, model in pairs:
            # Ensure FAISS ID fits in signed int64 range
            if fid >= DEFAULT_FAISS_ID_RANGE or fid < -DEFAULT_FAISS_ID_RANGE:
                fid = fid % DEFAULT_FAISS_ID_RANGE
                logger.debug(f"Normalized FAISS ID for chunk {cid}: {fid}")
            safe_pairs.append((cid, model, fid, dim))

        conn.executemany(
            """
        INSERT INTO embeddings(chunk_id, model, faiss_id, dim)
        VALUES(?,?,?,?)
        ON CONFLICT(chunk_id, model) DO UPDATE SET faiss_id=excluded.faiss_id, dim=excluded.dim
        """,
            safe_pairs,
        )

        logger.info(f"Assigned FAISS IDs to {len(safe_pairs)} chunks")

    except sqlite3.Error as e:
        logger.error(f"Failed to assign FAISS IDs: {e}")
        raise


def chunk_ids_for_faiss_ids(
    conn: sqlite3.Connection, faiss_ids: list[int], batch_size: int = DEFAULT_BATCH_SIZE
) -> dict[int, str]:
    """
    Map FAISS IDs to chunk IDs.

    Args:
        conn: Database connection
        faiss_ids: List of FAISS IDs to look up
        batch_size: Batch size for database queries

    Returns:
        Dictionary mapping FAISS ID to chunk ID

    Raises:
        sqlite3.Error: If database operation fails
    """
    if not faiss_ids:
        logger.debug("No FAISS IDs provided")
        return {}

    try:
        out: dict[int, str] = {}

        for i in range(0, len(faiss_ids), batch_size):
            batch = faiss_ids[i : i + batch_size]
            placeholders = ",".join("?" * len(batch))
            query = f"SELECT faiss_id, chunk_id FROM embeddings WHERE faiss_id IN ({placeholders})"

            for row in conn.execute(query, batch):
                out[int(row["faiss_id"])] = row["chunk_id"]

        logger.debug(f"Mapped {len(out)} FAISS IDs to chunk IDs")
        return out

    except sqlite3.Error as e:
        logger.error(f"Failed to map FAISS IDs to chunk IDs: {e}")
        raise
