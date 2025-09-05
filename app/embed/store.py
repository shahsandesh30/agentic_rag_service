# app/embed/store.py
from __future__ import annotations
import sqlite3, time, hashlib, numpy as np
from typing import Iterable, Tuple, Optional, List

EMB_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings(
  chunk_id TEXT PRIMARY KEY,
  dim INTEGER NOT NULL,
  vec BLOB NOT NULL,
  model TEXT NOT NULL,
  text_sha256 TEXT NOT NULL,
  created_at INTEGER,
  FOREIGN KEY(chunk_id) REFERENCES chunks(id)
);
CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model);
"""

def init_embedding_table(conn: sqlite3.Connection):
    conn.executescript(EMB_SCHEMA)

def sha256_text(s: str) -> str:
    h = hashlib.sha256()
    h.update(s.encode("utf-8", errors="ignore"))
    return h.hexdigest()

def to_blob(arr: np.ndarray) -> bytes:
    assert arr.dtype == np.float32 and arr.ndim == 1
    return arr.tobytes(order="C")

def from_blob(b: bytes, dim: int) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32, count=dim)

def fetch_chunks_to_embed(
    conn: sqlite3.Connection,
    model: str,
    limit: int = 1000,
) -> List[Tuple[str, str]]:
    """
    Returns list of (chunk_id, text) where embedding is missing or stale for the given model.
    Logic: if there is no row in embeddings for chunk_id with same model and matching text_sha256 -> needs (re)compute.
    """
    # join chunks with embeddings on chunk_id & model, compare text_sha256 later in Python for clarity
    rows = conn.execute("""
        SELECT c.id, c.text, e.text_sha256, e.model
        FROM chunks c
        LEFT JOIN embeddings e ON e.chunk_id = c.id AND e.model = ?
        LIMIT ?;
    """, (model, limit)).fetchall()

    todo = []
    for cid, text, emb_hash, emb_model in rows:
        t_hash = sha256_text(text or "")
        # print(emb_hash, t_hash)
        # print(emb_hash != t_hash)
        # print(emb_model, model)
        if emb_hash is None or emb_model != model or emb_hash != t_hash:
            print("0--------")
            todo.append((cid, text))
    print("to doooooo", todo)
    return todo

def upsert_embeddings(
    conn: sqlite3.Connection,
    model: str,
    dim: int,
    batch: List[Tuple[str, np.ndarray, str]],
):
    """
    batch: list of (chunk_id, vec, text_sha256)
    """
    now = int(time.time())
    conn.executemany(
        """INSERT INTO embeddings(chunk_id, dim, vec, model, text_sha256, created_at)
           VALUES(?,?,?,?,?,?)
           ON CONFLICT(chunk_id) DO UPDATE SET
             dim=excluded.dim, vec=excluded.vec, model=excluded.model, text_sha256=excluded.text_sha256, created_at=excluded.created_at;""",
        [(cid, dim, to_blob(vec), model, th, now) for (cid, vec, th) in batch]
    )
