from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# from app.llm.hf import HFGenerator, get_shared_generator
from app.llm.groq_gen import GroqGenerator
from app.retrieval.vector import VectorSearcher
from app.retrieval.bm25 import BM25Searcher
from app.retrieval.hybrid import HybridSearcher
from app.qa.answer import answer_question
from app.agent.graph import run_agent

from prometheus_client import make_asgi_app
from app.obs.middleware import ObservabilityMiddleware
from app.obs.tracing import setup_tracing, setup_logging

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="RAG-Agentic-AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev/demo only
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

setup_logging()
setup_tracing(app)
app.add_middleware(ObservabilityMiddleware)
# expose /metrics (Prometheus text format)
app.mount("/metrics", make_asgi_app())

load_dotenv()

# Single shared generator instance
_vector = VectorSearcher(db_path="rag_local.db", model_name="BAAI/bge-small-en-v1.5")
_bm25   = BM25Searcher(db_path="rag_local.db")
_hybrid = HybridSearcher(_vector, _bm25, db_path="rag_local.db")
# _gen    = get_shared_generator()
llm_api_key = os.getenv("GROQ_API_KEY")
_gen = GroqGenerator(model="llama-3.1-8b-instant", api_key=llm_api_key)

class ChatReq(BaseModel):
    prompt: str

class ChatResp(BaseModel):
    output: str

class SearchReq(BaseModel):
    query: str
    top_k: int = 3
    mode: str = "hybrid"       # "vector" | "bm25" | "hybrid"
    k_vector: int = 20         # only used for hybrid
    k_bm25: int = 20           # only used for hybrid
    rrf_k: int = 60            # only used for hybrid
    rerank: bool = False
    rerank_k: int = 10
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_batch: int = 4
    max_passage_chars: int = 800

class AskReq(BaseModel):
    question: str
    top_k_ctx: int = 8
    rerank: bool = True
    rerank_k: int = 20
    mode: str = "hybrid"      # "vector" | "bm25" | "hybrid"
    k_vector: int = 40
    k_bm25: int = 40
    rrf_k: int = 60

class AgentAskReq(BaseModel):
    question: str
    trace: bool = False

@app.get("/healthz")
def health():
    return {"ok": True}

@app.post("/chat", response_model=ChatResp)
def chat(req: ChatReq):
    text = _gen.generate(req.prompt, system="You are a fast assistant.")
    return ChatResp(output=text)

@app.post("/search")
def search(req: SearchReq):
    mode = req.mode.lower()
    if mode == "vector":
        hits = _vector.search(req.query, top_k=req.top_k)
    elif mode == "bm25":
        hits = _bm25.search(req.query, top_k=req.top_k)
    else:
        hits = _hybrid.search(
            req.query,
            top_k=req.top_k,
            k_vector=req.k_vector,
            k_bm25=req.k_bm25,
            rrf_k=req.rrf_k,
            rerank=req.rerank,
            rerank_k=req.rerank_k,
            reranker_model=req.reranker_model,
            reranker_batch=req.reranker_batch,
            max_passage_chars=req.max_passage_chars,
        )
    return {"hits": hits}

@app.post("/ask")
def ask(req: AskReq):
    print("Received question:", req.question)
    mode = (req.mode or "hybrid").lower()
    if mode == "vector":
        _searcher = _vector
    elif mode == "bm25":
        _searcher = _bm25
    else:
        _searcher = _hybrid
    payload = answer_question(
        req.question,
        _searcher,
        _gen,
        top_k_ctx=req.top_k_ctx,
        k_vector=req.k_vector,
        k_bm25=req.k_bm25,
        rrf_k=req.rrf_k,
        rerank=req.rerank,
        rerank_k=req.rerank_k,
    )
    # Already a pydantic model → dict
    return payload.dict()

@app.post("/agent/ask")
def agent_ask(req: AgentAskReq):
    out = run_agent(req.question)
    return out if req.trace else out["final"]