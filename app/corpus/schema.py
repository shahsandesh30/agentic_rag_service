import sqlite3, json, time, os
from typing import Optional, Dict, Any, List, Tuple

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS documents(
  id TEXT PRIMARY KEY,
  path TEXT NOT NULL,
  source TEXT,
  sha256 TEXT NOT NULL,
  mime TEXT,
  n_pages INTEGER,
  created_at INTEGER
);
CREATE TABLE IF NOT EXISTS chunks(
  id TEXT PRIMARY KEY,
  doc_id TEXT NOT NULL,
  ord INTEGER NOT NULL,
  text TEXT NOT NULL,
  n_chars INTEGER NOT NULL,
  start_char INTEGER,
  end_char INTEGER,
  section TEXT,
  meta_json TEXT,
  FOREIGN KEY(doc_id) REFERENCES documents(id)
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""

def connect(db_path: str = "rag_local.db") -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True) if os.path.dirname(db_path) else None
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreigh_keys=ON;")
    return conn
    
def init_db(conn: sqlite3.Connection):
    conn.executescript(SCHEMA_SQL)

def upsert_document(conn, doc_id: str, path: str, source: str, sha256: str, mime: str, n_pages: int):
    conn.execute(
        """INSERT INTO documents(id, path, source, sha256, mime, n_pages, created_at)
           VALUES(?,?,?,?,?,?,?)
           ON CONFLICT(id) DO UPDATE SET path=excluded.path, source=excluded.source, sha256=excluded.sha256, mime=excluded.mime, n_pages=excluded.n_pages;""",
        (doc_id, path, source, sha256, mime, n_pages, int(time.time()))
    )

def upsert_chunks(conn, rows: List[Tuple]):
    """
    rows: (id, doc_id, ord, text, n_chars, start_char, end_char, section, meta_json)
    """
    conn.executemany(
        """INSERT INTO chunks(id, doc_id, ord, text, n_chars, start_char, end_char, section, meta_json)
           VALUES(?,?,?,?,?,?,?,?,?)
           ON CONFLICT(id) DO UPDATE SET text=excluded.text, n_chars=excluded.n_chars, start_char=excluded.start_char, end_char=excluded.end_char, section=excluded.section, meta_json=excluded.meta_json;""",
        rows
    )