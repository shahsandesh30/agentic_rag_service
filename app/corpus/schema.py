# app/corpus/schema.py
from __future__ import annotations
import sqlite3, json, time
from typing import Iterable, List, Dict, Tuple, Optional

def connect(db_path: str = "rag_local.db") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db(conn):
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

def upsert_document(conn: sqlite3.Connection, doc_id: str, path: str, source: str,
                    sha256_hex: str, mime: str, n_pages: int) -> None:
    conn.execute("""
    INSERT INTO documents(id, path, source, sha256, mime, n_pages, created_at)
    VALUES(?,?,?,?,?,?,?)
    ON CONFLICT(id) DO UPDATE SET path=excluded.path, source=excluded.source, mime=excluded.mime, n_pages=excluded.n_pages
    """, (doc_id, path, source, sha256_hex, mime, n_pages, time.time()))


def upsert_chunks(conn: sqlite3.Connection, rows: Iterable[Tuple]) -> None:
    """
    rows: (id, doc_id, ordinal, text, n_chars, start_char, end_char, section, meta_json)
    """
    conn.executemany("""
    INSERT INTO chunks(id, doc_id, ordinal, text, n_chars, start_char, end_char, section, meta_json)
    VALUES(?,?,?,?,?,?,?,?,?)
    ON CONFLICT(id) DO NOTHING
    """, list(rows))

def upsert_embeddings(conn, rows: Iterable[Tuple[str, str, int, bytes, int]]) -> None:
    """
    Insert full embedding row: (chunk_id, model, dim, vec, faiss_id)
    """
    conn.executemany("""
    INSERT INTO embeddings(chunk_id, model, dim, vec, faiss_id)
    VALUES(?,?,?,?,?)
    ON CONFLICT(chunk_id, model) DO UPDATE SET vec=excluded.vec, faiss_id=excluded.faiss_id, dim=excluded.dim
    """, list(rows))



def fetch_chunk_texts(conn: sqlite3.Connection, chunk_ids: List[str]) -> Dict[str, Dict]:
    if not chunk_ids:
        return {}
    out: Dict[str, Dict] = {}
    for i in range(0, len(chunk_ids), 900):
        batch = chunk_ids[i:i+900]
        q = f"SELECT id, text, section, meta_json, doc_id FROM chunks WHERE id IN ({','.join('?'*len(batch))})"
        for r in conn.execute(q, batch):
            meta = json.loads(r["meta_json"] or "{}")
            doc = conn.execute("SELECT path, source FROM documents WHERE id=?",(r["doc_id"],)).fetchone()
            out[r["id"]] = {
                "text": r["text"],
                "section": r["section"],
                "path": doc["path"] if doc else "",
                "source": doc["source"] if doc else meta.get("source"),
            }
    return out

def fetch_full_chunks(conn: sqlite3.Connection, chunk_ids: List[str]) -> Dict[str, str]:
    if not chunk_ids:
        return {}
    out: Dict[str, str] = {}
    for i in range(0, len(chunk_ids), 900):
        batch = chunk_ids[i:i+900]
        q = f"SELECT id, text FROM chunks WHERE id IN ({','.join('?'*len(batch))})"
        for r in conn.execute(q, batch):
            out[r["id"]] = r["text"]
    return out

def missing_embedding_chunk_ids(conn: sqlite3.Connection) -> List[Tuple[str,int]]:
    """
    Returns [(chunk_id, est_dim_placeholder=0)] for chunks that are not yet mapped in embeddings.
    """
    q = """
    SELECT c.id FROM chunks c
    LEFT JOIN embeddings e ON e.chunk_id = c.id
    WHERE e.chunk_id IS NULL
    """
    return [(r["id"], 0) for r in conn.execute(q).fetchall()]

def assign_faiss_ids(conn: sqlite3.Connection, pairs: List[Tuple[str,int,int,str]]) -> None:
    """
    pairs: List of (chunk_id, faiss_id, dim, model)
    Ensures faiss_id fits signed int64 range before inserting.
    """
    safe_pairs = []
    for cid, fid, dim, model in pairs:
        if fid >= 2**63 or fid < -(2**63):
            fid = fid % (2**63)   # normalize
        safe_pairs.append((cid, model, fid, dim))

    conn.executemany("""
    INSERT INTO embeddings(chunk_id, model, faiss_id, dim)
    VALUES(?,?,?,?)
    ON CONFLICT(chunk_id, model) DO UPDATE SET faiss_id=excluded.faiss_id, dim=excluded.dim
    """, safe_pairs)

def chunk_ids_for_faiss_ids(conn: sqlite3.Connection, faiss_ids: List[int]) -> Dict[int, str]:
    if not faiss_ids:
        return {}
    out: Dict[int, str] = {}
    for i in range(0, len(faiss_ids), 900):
        batch = faiss_ids[i:i+900]
        q = f"SELECT faiss_id, chunk_id FROM embeddings WHERE faiss_id IN ({','.join('?'*len(batch))})"
        for r in conn.execute(q, batch):
            out[int(r["faiss_id"])] = r["chunk_id"]
    return out
