import tempfile, sqlite3, numpy as np, os, pytest

@pytest.fixture
def tmp_db():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = tmp.name
    tmp.close()  # release handle on Windows

    conn = sqlite3.connect(db_path)
    conn.executescript("""
    CREATE TABLE chunks (
        id TEXT PRIMARY KEY,
        doc_id TEXT,
        section TEXT,
        text TEXT,
        meta_json TEXT
    );
    CREATE TABLE embeddings (
        chunk_id TEXT,
        model TEXT,
        dim INTEGER,
        vec BLOB
    );
    """)
    # Insert one chunk + embedding
    text = "Refunds are allowed within 30 days."
    vec = np.ones((384,), dtype=np.float32) / np.sqrt(384)
    conn.execute("INSERT INTO chunks VALUES (?,?,?,?,?)",
                 ("chunk1","doc1","policy",text,"{}"))
    conn.execute("INSERT INTO embeddings VALUES (?,?,?,?)",
                 ("chunk1","BAAI/bge-small-en-v1.5",384,vec.tobytes()))
    conn.commit()
    conn.close()

    yield db_path

    # cleanup after tests
    try:
        os.remove(db_path)
    except PermissionError:
        pass
