## Agentic RAG Services

A local, modular Retrieval-Augmented Generation (RAG) service with agentic capabilities. Ingest PDFs/TXT/Markdown into SQLite, compute embeddings, perform hybrid search (vector + BM25 with optional cross-encoder reranking), and answer questions with grounded citations via a FastAPI API. Features an intelligent agent workflow using LangGraph for intent routing, query rewriting, and safety compliance.

### Features
- **Hybrid retrieval**: vector (SentenceTransformers) + BM25 with Reciprocal Rank Fusion (RRF)
- **Optional cross-encoder reranking** for higher precision
- **Lightweight storage**: SQLite for documents, chunks, and embeddings (float32 BLOB)
- **FastAPI endpoints**: chat, search, grounded QA with citations, and agentic workflow
- **Agentic workflow**: LangGraph-based agent with intent routing, query rewriting, and safety checks
- **Safety & compliance**: Content filtering, PII detection, injection prevention
- **Observability**: Prometheus metrics, OpenTelemetry tracing, structured logging
- **Evaluation framework**: Comprehensive RAG evaluation with multiple metrics
- **Simple CLIs** for ingestion, embedding, search, and agent execution

### Repository layout
- `app/corpus/`: ingest files, clean text, chunk, and persist to SQLite
- `app/embed/`: embedding model wrapper, compute + store embeddings
- `app/retrieval/`: vector/BM25/hybrid retrieval, reranking, and data access
- `app/qa/`: prompts, schemas, and RAG QA orchestration
- `app/llm/`: Hugging Face causal LM wrapper
- `app/agent/`: LangGraph-based agent workflow with intent routing, query rewriting, and safety
- `app/safety/`: content filtering, PII detection, injection prevention, and compliance checks
- `app/obs/`: observability middleware, metrics, tracing, and logging
- `app/eval/`: evaluation framework with retrieval and answer quality metrics
- `app/api.py`: FastAPI app exposing `/healthz`, `/chat`, `/search`, `/ask`, `/agent/ask`, `/metrics`
- `rag_local.db`: default SQLite database (with WAL files)
- `data/`: sample data (`data/policies/final_report.pdf`)

## Quickstart

### 1) Environment
Requirements are in `requirements.txt`. A virtual environment is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

GPU is optional but recommended. To install CUDA-enabled PyTorch, use the official selector on the PyTorch site: [PyTorch installation page](https://pytorch.org/get-started/locally/).

### 2) Optional: configure `.env`
Create a `.env` in the repo root to override defaults from `app/config.py`:

```ini
MODEL_NAME=microsoft/phi-2
DEVICE_MAP=auto
MAX_NEW_TOKENS=128
TEMPERATURE=0.2
TOP_P=0.9
DO_SAMPLE=false
PORT=8000
```

### 3) Ingest your corpus
Ingest PDFs/TXT/MD from a folder or a single file into `rag_local.db`.

```powershell
python -m app.corpus.ingest_cli data/policies --db rag_local.db --source policies --max-chars 1200 --overlap 150
```

### 4) Compute embeddings
Compute (or refresh) embeddings for chunks and store them in SQLite.

```powershell
python -m app.embed.compute_cli --db rag_local.db --model BAAI/bge-small-en-v1.5 --batch-size 64
```

### 5) Run the API

```powershell
uvicorn app.api:app --reload --port 8000
```

### 6) Serve the frontend

```powershell
python -m http.server 5173 -d frontend
# open http://localhost:5173
```

## API usage

### POST /search
Hybrid search (vector | bm25 | hybrid) with optional reranking.

```powershell
$body = @{ query = "what's in the report?"; mode = "hybrid"; top_k = 5; rerank = $true } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/search -ContentType 'application/json' -Body $body
```

```bash
curl -X POST http://localhost:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"what\u0027s in the report?","mode":"hybrid","top_k":5,"rerank":true}'
```

### POST /ask
RAG QA: retrieves context, optionally reranks, and asks the LLM to return strict JSON with citations and confidence.

```powershell
$body = @{ question = "Summarize the report" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/ask -ContentType 'application/json' -Body $body
```

```bash
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Summarize the report"}'
```

### POST /agent/ask
Agentic workflow: intelligent intent routing, query rewriting, and safety compliance using LangGraph.

```powershell
$body = @{ question = "What are the key findings in the report?"; trace = $false } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/agent/ask -ContentType 'application/json' -Body $body
```

```bash
curl -X POST http://localhost:8000/agent/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What are the key findings in the report?","trace":false}'
```

### POST /chat
Direct LLM generation (no retrieval).

```powershell
$body = @{ prompt = "Hello" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://localhost:8000/chat -ContentType 'application/json' -Body $body
```

```bash
curl -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello"}'
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


