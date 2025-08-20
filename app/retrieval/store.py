# app/retrieval/store.py
from __future__ import annotations
import sqlite3, json
from typing import List, Tuple, Dict
import numpy as np

def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def load_embeddings(conn: sqlite3.Connection, model: str) -> tuple[list[str], np.ndarray]:
    """
    Returns (chunk_ids, matrix) where matrix shape is (N, dim), L2-normalized float32.
    """
    rows = conn.execute(
        "SELECT chunk_id, dim, vec FROM embeddings WHERE model = ?",
        (model,)
    ).fetchall()
    if not rows:
        return [], np.zeros((0, 1), dtype=np.float32)

    # assume consistent dim; build matrix
    dim = rows[0]["dim"]
    ids: list[str] = []
    vecs: list[np.ndarray] = []
    for r in rows:
        ids.append(r["chunk_id"])
        v = np.frombuffer(r["vec"], dtype=np.float32, count=dim)
        vecs.append(v)
    mat = np.vstack(vecs).astype(np.float32, copy=False)
    return ids, mat

def fetch_chunk_texts(conn: sqlite3.Connection, chunk_ids: list[str]) -> dict[str, dict]:
    """
    Fetch text + minimal metadata for a list of chunk IDs.
    Returns mapping: chunk_id -> {text, section, source, path}
    """
    out: dict[str, dict] = {}
    if not chunk_ids:
        return out

    # chunk IN() to avoid SQLite parameter limits
    for i in range(0, len(chunk_ids), 999):
        batch = chunk_ids[i:i+999]
        q = f"SELECT c.id, c.text, c.section, c.meta_json FROM chunks c WHERE c.id IN ({','.join('?'*len(batch))})"
        for row in conn.execute(q, batch):
            meta = {}
            try:
                meta = json.loads(row["meta_json"] or "{}")
            except Exception:
                meta = {}
            out[row["id"]] = {
                "text": row["text"],
                "section": row["section"],
                "source": meta.get("source"),
                "path": meta.get("path"),
            }
    return out

# Load all chunk ids + texts (for BM25)
def load_all_chunks(conn: sqlite3.Connection) -> tuple[list[str], list[str]]:
    rows = conn.execute("SELECT id, text FROM chunks").fetchall()
    ids, texts = [], []
    for r in rows:
        ids.append(r["id"])
        texts.append(r["text"] or "")

    return ids, texts

def fetch_full_chunks(conn: sqlite3.Connection, chunk_ids: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    if not chunk_ids:
        return out
    for i in range(0, len(chunk_ids), 999):
        batch = chunk_ids[i:i+999]
        q = f"SELECT id, text FROM chunks WHERE id IN ({','.join('?'*len(batch))})"
        for row in conn.execute(q, batch):
            out[row["id"]] = row["text"] or ""
    return out