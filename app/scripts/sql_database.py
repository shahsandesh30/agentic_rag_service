import sqlite3

conn = sqlite3.connect("rag_local.db")
cur = conn.cursor()

# 1) Inspect duplicates
cur.execute("""
SELECT sha256, COUNT(*) AS n
FROM documents
GROUP BY sha256
HAVING n > 1;
""")
print("Dup hashes:", cur.fetchall())

# 2) Delete embeddings for duplicates (keeping source='policies' as example)
cur.executescript("""
DELETE FROM embeddings
WHERE chunk_id IN (
  SELECT c.id
  FROM chunks c
  WHERE c.doc_id IN (
    SELECT d.id FROM documents d
    WHERE d.sha256 IN (SELECT sha256 FROM documents GROUP BY sha256 HAVING COUNT(*) > 1)
      AND d.source <> 'policies'
  )
);

DELETE FROM chunks
WHERE doc_id IN (
  SELECT d.id FROM documents d
  WHERE d.sha256 IN (SELECT sha256 FROM documents GROUP BY sha256 HAVING COUNT(*) > 1)
    AND d.source <> 'policies'
);

DELETE FROM documents
WHERE sha256 IN (SELECT sha256 FROM documents GROUP BY sha256 HAVING COUNT(*) > 1)
  AND source <> 'policies';
""")

conn.commit()
conn.close()
print("Cleanup done.")
