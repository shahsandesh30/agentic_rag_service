# Agentic RAG Services (Production Pipeline)

A modular, production-style Retrieval-Augmented Generation (RAG) service with agentic capabilities.  
Documents (PDF/TXT/Markdown) are ingested into **SQLite** (for metadata + chunks),  
embeddings are stored in **FAISS** (fast vector search),  
and answers are generated via **Groq-hosted LLMs** with grounded citations.  

An **agent workflow** (LangGraph) provides intent routing, query rewriting, and compliance checks.

---

## ✨ Features
- **Vector retrieval with FAISS** (fast ANN search)  
- **SQLite metadata store** (documents, chunks, faiss_id mapping)  
- **RAG QA** with citations, confidence scoring, and safety checks  
- **Groq LLM API integration** (e.g., `llama-3.1-8b-instant`)  
- **Agentic workflow**:  
  - intent routing (RAG vs chit-chat)  
  - query rewriting for better recall  
  - compliance filtering  
- **Observability**: Prometheus metrics + OpenTelemetry tracing  
- **Frontend demo**: API console to test `/search`, `/ask`, `/agent/ask`, `/chat`  

---

## 📂 Repository Layout
- `app/corpus/` → ingest files, normalize, chunk, persist to SQLite  
- `app/embed/` → embedding model wrapper (SentenceTransformers)  
- `app/retrieval/faiss_sqlite.py` → FAISS+SQLite searcher (production pipeline)  
- `app/qa/` → prompt templates, QA orchestration, schema  
- `app/llm/` → Groq API client + optional local HF wrapper  
- `app/agent/` → LangGraph-based agent (router, researcher, answerer, compliance)  
- `app/safety/` → injection filtering, PII detection, risk scoring  
- `app/obs/` → observability middleware, metrics, tracing  
- `app/api.py` → FastAPI app exposing `/healthz`, `/chat`, `/search`, `/ask`, `/agent/ask`, `/metrics`  
- `scripts/ingest_and_index.py` → end-to-end ingestion (SQLite + FAISS indexing)  
- `frontend/` → minimal web console for interacting with the API  
- `rag_local.db` → SQLite database (documents & chunks)  
- `faiss_index/index.faiss` → FAISS index (embeddings)  
- `data/` → sample corpus  

---

## 🚀 Quickstart

### 1) Environment
```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2) Configure `.env`
Create a `.env` in the repo root:

```ini
GROQ_API_KEY=your_api_key_here
MODEL_NAME=BAAI/bge-small-en-v1.5
```

### 3) Ingest + Index
Run the ingestion pipeline to process PDFs/TXT/MD into SQLite & FAISS:

```bash
python -m app.scripts.ingest_and_index data/
```

### 4) Run the API
```bash
uvicorn app.api:app --reload --port 8000
```

### 5) Frontend Demo
```bash
python -m http.server 5173 -d frontend
# open http://localhost:5173
```

---

## 🔌 API Endpoints

### POST `/search`
Vector search using FAISS.
```bash
curl -X POST http://localhost:8000/search   -H "Content-Type: application/json"   -d '{"query":"refund policy", "top_k":5}'
```

### POST `/ask`
RAG QA with context + Groq LLM.
```bash
curl -X POST http://localhost:8000/ask   -H "Content-Type: application/json"   -d '{"question":"What does the policy say about refunds?"}'
```

### POST `/agent/ask`
Agentic workflow (routing, rewrites, compliance).
```bash
curl -X POST http://localhost:8000/agent/ask   -H "Content-Type: application/json"   -d '{"question":"What are the key findings?", "trace":true}'
```

### POST `/chat`
Direct LLM chat (no retrieval).
```bash
curl -X POST http://localhost:8000/chat   -H "Content-Type: application/json"   -d '{"prompt":"Hello"}'
```

### GET /metrics
Prometheus metrics for monitoring and observability.

```bash
curl http://localhost:8000/metrics
```

## CLI utilities

### Ingest
```bash
python -m app.corpus.ingest_cli path/to/folder --db rag_local.db --source my_corpus
```

### Compute embeddings
```bash
python -m app.embed.compute_cli --db rag_local.db --model BAAI/bge-small-en-v1.5
```

### Search from terminal
```bash
python -m app.retrieval.search_cli "your query" --mode hybrid --rerank --rerank-k 20
```

### Run agent workflow
```bash
python -m app.agent.agent_cli "your question" --trace
```

### Evaluate RAG performance
```bash
python -m app.eval.runner --dataset app/eval/samples.yaml --db rag_local.db
```

## Agent Workflow

The agent module implements a LangGraph-based workflow with the following components:

1. **Router**: Determines intent (RAG vs chitchat) and routes accordingly
2. **Researcher**: For RAG queries, generates query rewrites to improve retrieval
3. **Answerer**: Executes RAG or direct LLM generation based on intent
4. **Compliance**: Applies safety checks and content filtering

The workflow automatically handles:
- Intent classification between information-seeking and conversational queries
- Query rewriting for better retrieval performance
- Safety compliance and content filtering
- Confidence scoring and answer selection

## Safety & Compliance

The safety module provides comprehensive content protection:

- **Intent detection**: Blocks prohibited queries and high-risk intents
- **Content filtering**: Removes injected or suspicious context chunks
- **PII detection**: Identifies and masks personal information
- **Injection prevention**: Redacts prompt injection attempts
- **Risk scoring**: Dynamic confidence adjustment based on safety signals

## Observability

Built-in monitoring and observability features:

- **Prometheus metrics**: HTTP request counts, latency, and custom metrics
- **OpenTelemetry tracing**: Distributed tracing for request flows
- **Structured logging**: Request IDs, timing, and error tracking
- **Middleware**: Automatic request/response logging and metrics collection

## Evaluation Framework

Comprehensive evaluation capabilities for RAG systems:

- **Retrieval metrics**: Recall@K, MRR, NDCG for search quality
- **Answer quality**: Semantic similarity, faithfulness, citation alignment
- **Safety evaluation**: Blocking rates and risk assessment
- **Batch evaluation**: Process multiple test cases and generate reports

## Data model (SQLite)
- `documents(id, path, source, sha256, mime, n_pages, created_at)`
- `chunks(id, doc_id, ord, text, n_chars, start_char, end_char, section, meta_json)`
- `embeddings(chunk_id, dim, vec, model, text_sha256, created_at)` where `vec` is a float32 BLOB

## Tuning tips
- Adjust chunk `max-chars` and `overlap` during ingestion for your corpus
- Use `--rerank` on `/search` or increase `top_k_ctx` on `/ask` for tougher queries
- Swap models via `.env` (LLM, embedder, reranker)
- Configure safety policies in `app/safety/policy.py` for your use case
- Use the agent workflow (`/agent/ask`) for intelligent query handling
- Monitor performance with `/metrics` and evaluation framework

## Known issues (to be fixed)
- `app/llm/hf.py`: `torch.cuda.is_available` should be called as a function: `torch.cuda.is_available()`
- `app/qa/answer.py`: `HFGenerator.generate` is called with a dict; it expects `(prompt: str, contexts: List[str], system: str)`
- `app/corpus/schema.py`: PRAGMA typo `foreigh_keys` → `foreign_keys` (foreign key enforcement currently off)

## Troubleshooting
- NLTK: If tokenizers are missing, BM25 falls back to a regex tokenizer automatically
- PDFs: `pdfplumber` extracts text; scanned/image-only PDFs may need OCR (e.g., Tesseract) before ingestion
- Memory: The service loads embeddings/chunks in memory for search; consider sharding or a vector DB for very large corpora
- Agent workflow: Use `--trace` flag to debug agent execution flow
- Safety: Check `/metrics` for safety-related metrics and blocking rates

## License
TBD


