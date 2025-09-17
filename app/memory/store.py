# app/memory/store.py
import sqlite3, time
from typing import List, Dict

DB_PATH = "rag_local.db"

def connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def save_message(session_id: str, role: str, content: str):
    conn = connect()
    conn.execute(
        "INSERT INTO conversations(session_id, role, content, created_at) VALUES(?,?,?,?)",
        (session_id, role, content, time.time())
    )
    conn.commit()
    conn.close()

def get_recent_messages(session_id: str, limit: int = 5) -> List[Dict]:
    conn = connect()
    rows = conn.execute(
        "SELECT role, content FROM conversations WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
        (session_id, limit)
    ).fetchall()
    conn.close()
    # reverse to chronological
    return [dict(r) for r in rows[::-1]]
