# app/embed/compute.py
from __future__ import annotations
import sqlite3, math
from typing import List, Tuple
import numpy as np

from app.corpus.schema import connect, init_db
from .store import init_embedding_table, fetch_chunks_to_embed, upsert_embeddings, sha256_text
from .model import Embedder

def compute_embeddings(
    db_path: str = "rag_local.db",
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 64,
    limit_per_pass: int = 5000,
):
    conn = connect(db_path)
    init_db(conn)
    init_embedding_table(conn)

    emb = Embedder(model_name=model_name)
    total_done = 0

    while True:
        todo = fetch_chunks_to_embed(conn, model=model_name, limit=limit_per_pass)
        if not todo:
            break

        # batch over 'todo'
        for i in range(0, len(todo), batch_size):
            batch_ids, batch_texts = zip(*todo[i:i+batch_size])
            vecs: np.ndarray = emb.encode(list(batch_texts), batch_size=batch_size)
            rows = []
            for cid, text, vec in zip(batch_ids, batch_texts, vecs):
                rows.append((cid, vec, sha256_text(text)))
            upsert_embeddings(conn, model=model_name, dim=emb.dim, batch=rows)
            conn.commit()
            total_done += len(rows)

    return {"embedded": total_done, "model": model_name, "db": db_path}
