# app/api.py
import json
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.agent.graph import run_agent
from app.config import get_settings
from app.embed.model import Embedder
from app.http_errors import (
    generic_exception_handler,
    http_exception_handler,
    request_validation_exception_handler,
)
from app.llm.groq_gen import GroqGenerator
from app.memory.store import get_recent_messages, save_message
from app.obs.middleware import ObservabilityMiddleware
from app.obs.request_log import RequestLogMiddleware
from app.obs.tracing import setup_logging, setup_tracing
from app.qa.answer import answer_question
from app.rerank.reranker import Reranker
from app.retrieval.bm25_sqlite import BM25SqliteSearcher
from app.retrieval.faiss_sqlite import FaissSqliteSearcher
from app.retrieval.hybrid import HybridSearcher

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

# --- FastAPI App ---
app = FastAPI(title="Agentic RAG Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev/demo only
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(RequestLogMiddleware)

app.add_exception_handler(RequestValidationError, request_validation_exception_handler)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

setup_logging()
setup_tracing(app)
app.add_middleware(ObservabilityMiddleware)
# expose /metrics (Prometheus text format)
app.mount("/metrics", make_asgi_app())

load_dotenv()

# --- Components ---
embedder = Embedder(model_name=settings.embed_model)

# Build base FAISS searcher
# Build base FAISS searcher
_faiss = FaissSqliteSearcher(
    embedder=embedder,
    sqlite_path=settings.sqlite_path,
    faiss_path=settings.faiss_path,
    top_k_default=settings.top_k,
)

# If HYBRID_RETRIEVAL is on → wrap FAISS with BM25 and hybrid
if settings.hybrid:
    _bm25 = BM25SqliteSearcher(sqlite_path=settings.sqlite_path, top_k_default=settings.top_k)
    _searcher = HybridSearcher(_faiss, _bm25, top_k_default=settings.top_k)
else:
    _searcher = _faiss

_reranker = Reranker(settings.reranker_model) if settings.use_rerank else None

llm_api_key = os.getenv("GROQ_API_KEY")
_gen = GroqGenerator(model="llama-3.1-8b-instant", api_key=llm_api_key)


# --- Request Models ---
class ChatReq(BaseModel):
    prompt: str


class ChatResp(BaseModel):
    output: str


class SearchReq(BaseModel):
    query: str
    top_k: int = 5


class AskReq(BaseModel):
    session_id: str = "default"  # 👈 Added session tracking
    question: str
    top_k_ctx: int = 8


class AgentAskReq(BaseModel):
    session_id: str = "default"  # 👈 Added session tracking
    question: str
    trace: bool = False


# --- Endpoints ---
@app.get("/healthz")
def health():
    return {"ok": True, "env": settings.env}


@app.post("/chat", response_model=ChatResp)
def chat(req: ChatReq):
    text = _gen.generate(req.prompt, system="You are a fast assistant.")
    return ChatResp(output=text)


@app.post("/search")
def search(req: SearchReq):
    hits = _searcher.search(req.query, top_k=req.top_k)
    if _reranker:
        # Rerank the retrieved hits for demo; keep the same response shape
        hits = _reranker.rerank(req.query, hits, top_k=req.top_k)
    return {"hits": hits}


@app.post("/ask")
def ask(req: AskReq):
    # --- Fetch conversation history ---
    history = get_recent_messages(req.session_id, limit=5)
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])

    # --- Combine history + new question ---
    user_input = f"{history_text}\nUser: {req.question}" if history_text else req.question

    # --- Run RAG QA ---
    payload = answer_question(
        user_input,  # 👈 enriched with history
        _searcher,
        _gen,
        top_k_ctx=req.top_k_ctx,
    )

    # --- Save both Q and A to memory ---
    save_message(req.session_id, "user", req.question)
    save_message(req.session_id, "assistant", payload.answer)

    answer_dict = json.loads(payload.dict()["answer"])

    return answer_dict["answer"]
    # return payload.dict()


@app.post("/agent/ask")
def agent_ask(req: AgentAskReq):
    # --- Fetch conversation history ---
    history = get_recent_messages("agent_" + req.session_id, limit=5)
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])

    # --- Combine history + new question ---
    enriched_q = f"{history_text}\nUser: {req.question}" if history_text else req.question

    # --- Run agent ---
    out = run_agent(enriched_q)

    # --- Save memory ---
    save_message("agent_" + req.session_id, "user", req.question)
    save_message("agent_" + req.session_id, "assistant", out["final"]["answer"])

    return out if req.trace else out["final"]
