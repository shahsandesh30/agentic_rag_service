# app/retrieval/faiss_sqlite.py
from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

import numpy as np

try:
    import faiss  # type: ignore
except Exception as e:  # pragma: no cover
    raise ImportError("faiss is required for vector search. pip install faiss-cpu") from e

from app.config import get_settings

log = logging.getLogger(__name__)
_settings = get_settings()


class FaissSqliteSearcher:
    """
    Thread-safe, lazily loaded FAISS + SQLite retriever.

    - Lazily reads the FAISS index (only on first search).
    - Uses one SQLite connection per thread via threading.local().
    - Maps FAISS ids <-> chunk metadata stored in SQLite.
    - Compatible with your existing encode() API on the provided embedder.

    Expected DB schema (abbrev):
      documents(id, path, ...)
      chunks(id, doc_id, text, section, faiss_id, ...)
      -- faiss_id (INTEGER) must map each chunk row to an entry in FAISS
    """

    def __init__(
        self,
        embedder: Any,
        db_path: str | None = None,  # backward-compat name
        sqlite_path: str | None = None,
        faiss_path: str | None = None,
        top_k_default: int | None = None,
    ):
        self._embedder = embedder

        # Respect passed args first, then config defaults
        self.sqlite_path = Path(sqlite_path or db_path or _settings.sqlite_path)
        self.faiss_path = Path(faiss_path or _settings.faiss_path)
        self._top_k_default = int(top_k_default or _settings.top_k)

        # Lazy-loaded state
        self._faiss_index: faiss.Index | None = None
        self._dim: int | None = None

        # Thread-local SQLite connection store
        self._local = threading.local()

        # One-time init locks
        self._faiss_lock = threading.Lock()
        self._closing = False

        log.info(
            "FaissSqliteSearcher configured (sqlite=%s, faiss=%s, top_k_default=%s)",
            self.sqlite_path,
            self.faiss_path,
            self._top_k_default,
        )

    # SQLite connection mgmt
    def _get_conn(self) -> sqlite3.Connection:
        """
        Returns a per-thread SQLite connection.
        """
        conn = getattr(self._local, "conn", None)
        if conn is None:
            if not self.sqlite_path.exists():
                raise FileNotFoundError(f"SQLite DB not found: {self.sqlite_path}")
            conn = sqlite3.connect(str(self.sqlite_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
            log.debug("Opened SQLite connection for thread %s", threading.get_ident())
        return conn

    def _close_conn(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:  # pragma: no cover
                pass
            finally:
                self._local.conn = None
                log.debug("Closed SQLite connection for thread %s", threading.get_ident())

    # FAISS index mgmt
    def _ensure_faiss_loaded(self) -> None:
        """
        Lazily loads the FAISS index once per process (thread-safe).
        """
        if self._faiss_index is not None:
            return
        with self._faiss_lock:
            if self._faiss_index is not None:
                return
            if not self.faiss_path.exists():
                raise FileNotFoundError(f"FAISS index not found: {self.faiss_path}")
            self._faiss_index = faiss.read_index(str(self.faiss_path))
            self._dim = int(self._faiss_index.d)
            log.info(
                "Loaded FAISS index (d=%s, path=%s)",
                self._dim,
                self.faiss_path,
            )

    # Public API
    def search(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        """
        Vector search + metadata lookup.

        Returns list of dicts:
          { chunk_id, text, section, source, score }
        """
        if not query or not query.strip():
            return []

        self._ensure_faiss_loaded()
        conn = self._get_conn()

        k = int(top_k or self._top_k_default)
        if k <= 0:
            return []

        # Encode and normalize query
        q_emb = self._encode_query(query)  # shape (1, d)
        scores, ids = self._faiss_index.search(q_emb, k)  # type: ignore

        faiss_ids = [int(i) for i in ids[0] if int(i) >= 0]
        if not faiss_ids:
            return []

        # Fetch chunk metadata, preserve order of FAISS results
        placeholders = ",".join(["?"] * len(faiss_ids))
        sql = (
            "SELECT c.id AS chunk_id, c.text, c.section, d.path AS source, c.faiss_id "
            "FROM chunks c "
            "JOIN documents d ON d.id = c.doc_id "
            f"WHERE c.faiss_id IN ({placeholders})"
        )
        rows = conn.execute(sql, faiss_ids).fetchall()

        # Map faiss_id -> row for quick lookup
        row_by_faiss = {int(r["faiss_id"]): r for r in rows}

        results: list[dict[str, Any]] = []
        for rank, (faiss_id, score) in enumerate(zip(ids[0], scores[0], strict=False)):
            fid = int(faiss_id)
            if fid < 0:
                continue
            row = row_by_faiss.get(fid)
            if not row:
                # If index/db are out of sync, just skip this hit
                continue
            results.append(
                {
                    "chunk_id": row["chunk_id"],
                    "text": row["text"],
                    "section": row["section"],
                    "source": row["source"],
                    "score": float(score),
                    "rank": rank + 1,
                }
            )
        return results

    def reload(self) -> None:
        """
        Hot-reload the FAISS index (e.g., after reindexing on disk).
        """
        with self._faiss_lock:
            self._faiss_index = None
            self._dim = None
        self._ensure_faiss_loaded()
        log.info("FAISS index reloaded")

    def close(self) -> None:
        """
        Close resources. Safe to call multiple times.
        """
        self._closing = True
        self._close_conn()

    # Helpers
    def _encode_query(self, query: str) -> np.ndarray:
        """
        Uses the provided embedder to encode the query and ensures float32, contiguous.
        """
        emb = self._embedder.encode([query])  # (1, d) or list -> np.ndarray
        arr = np.asarray(emb, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr[None, :]
        return np.ascontiguousarray(arr)

    # Ensure connections are closed when GC'd (best-effort)
    def __del__(self) -> None:  # pragma: no cover
        if not self._closing:
            self.close()
