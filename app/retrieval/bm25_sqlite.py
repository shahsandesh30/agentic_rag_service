# app/retrieval/bm25_sqlite.py
from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from app.config import get_settings

log = logging.getLogger(__name__)
_settings = get_settings()


class BM25SqliteSearcher:
    """
    BM25 over chunk texts stored in SQLite.
    - Lazy-loads all chunk texts (id, text, section, source) into memory once.
    - Builds a BM25Okapi index over tokenized terms (simple whitespace lower).
    - Returns the same hit shape as FAISS searcher: {chunk_id,text,section,source,score,rank}
    """

    def __init__(self, sqlite_path: str | None = None, top_k_default: int | None = None):
        self.sqlite_path = Path(sqlite_path or _settings.sqlite_path)
        self._top_k_default = int(top_k_default or _settings.top_k)

        self._conn: sqlite3.Connection | None = None
        self._docs: list[dict[str, Any]] | None = None
        self._bm25: BM25Okapi | None = None

        self._init_lock = threading.Lock()
        self._local = threading.local()

        log.info(
            "BM25SqliteSearcher configured (sqlite=%s, top_k_default=%s)",
            self.sqlite_path,
            self._top_k_default,
        )

    # ----- SQLite connection (per-thread) -----
    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            if not self.sqlite_path.exists():
                raise FileNotFoundError(f"SQLite DB not found: {self.sqlite_path}")
            conn = sqlite3.connect(str(self.sqlite_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def _ensure_loaded(self) -> None:
        if self._bm25 is not None and self._docs is not None:
            return
        with self._init_lock:
            if self._bm25 is not None and self._docs is not None:
                return
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT c.id AS chunk_id, c.text, c.section, d.path AS source "
                "FROM chunks c JOIN documents d ON d.id = c.doc_id"
            ).fetchall()
            self._docs = [
                {
                    "chunk_id": r["chunk_id"],
                    "text": r["text"],
                    "section": r["section"],
                    "source": r["source"],
                }
                for r in rows
            ]
            # Tokenize (very simple; swap in smarter tokenization if needed)
            corpus = [d["text"].lower().split() for d in self._docs]
            self._bm25 = BM25Okapi(corpus)
            log.info("BM25 index loaded (%s chunks)", len(self._docs))

    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        if not query or not query.strip():
            return []
        self._ensure_loaded()
        k = int(top_k or self._top_k_default)
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        # top-k by score
        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        hits: list[dict[str, Any]] = []
        for rank, i in enumerate(idxs, start=1):
            doc = self._docs[i]
            hits.append(
                {
                    "chunk_id": doc["chunk_id"],
                    "text": doc["text"],
                    "section": doc["section"],
                    "source": doc["source"],
                    "score": float(scores[i]),
                    "rank": rank,
                }
            )
        return hits

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            finally:
                self._local.conn = None
