# app/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os

from app.llm.groq_gen import GroqGenerator
from app.retrieval.faiss_sqlite import FaissSqliteSearcher
from app.qa.answer import answer_question
from app.agent.graph import run_agent

from prometheus_client import make_asgi_app
from app.obs.middleware import ObservabilityMiddleware
from app.obs.tracing import setup_tracing, setup_logging

from fastapi.middleware.cors import CORSMiddleware
from app.embed.model import Embedder

# 👇 memory functions
from app.memory.store import save_message, get_recent_messages

# --- FastAPI App ---
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


def _session_key(session_id: str, prefix: str = "") -> str:
    """Consistently build a memory session key with an optional prefix."""
    return f"{prefix}{session_id}" if prefix else session_id


def _build_enriched_prompt(session_id: str, question: str, *, history_limit: int = 5) -> str:
    """Return the user question enriched with recent chat history if available."""
    history = get_recent_messages(session_id, limit=history_limit)
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    return f"{history_text}\nUser: {question}" if history_text else question


def _record_exchange(session_id: str, question: str, answer: str) -> None:
    """Persist the latest user and assistant messages to memory."""
    save_message(session_id, "user", question)
    save_message(session_id, "assistant", answer)

# --- Components ---
embedder = Embedder(model_name="BAAI/bge-small-en-v1.5")
_searcher = FaissSqliteSearcher(embedder, db_path="rag_local.db")

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
    session_id: str = "default"   # 👈 Added session tracking
    question: str
    top_k_ctx: int = 8

class AgentAskReq(BaseModel):
    session_id: str = "default"   # 👈 Added session tracking
    question: str
    trace: bool = False

# --- Endpoints ---
@app.get("/healthz")
def health():
    return {"ok": True}

@app.post("/chat", response_model=ChatResp)
def chat(req: ChatReq):
    text = _gen.generate(req.prompt, system="You are a fast assistant.")
    return ChatResp(output=text)

@app.post("/search")
def search(req: SearchReq):
    hits = _searcher.search(req.query, top_k=req.top_k)
    return {"hits": hits}

@app.post("/ask")
def ask(req: AskReq):
    session_id = _session_key(req.session_id)
    user_input = _build_enriched_prompt(session_id, req.question)

    # --- Run RAG QA ---
    payload = answer_question(
        user_input,   # 👈 enriched with history
        _searcher,
        _gen,
        top_k_ctx=req.top_k_ctx,
    )

    # --- Save both Q and A to memory ---
    _record_exchange(session_id, req.question, payload.answer)

    return payload.dict()
    

@app.post("/agent/ask")
def agent_ask(req: AgentAskReq):
    session_id = _session_key(req.session_id, prefix="agent_")
    enriched_q = _build_enriched_prompt(session_id, req.question)

    # --- Run agent ---
    out = run_agent(enriched_q)

    final_section = out.get("final") if isinstance(out, dict) else {}
    final_answer = ""
    if isinstance(final_section, dict):
        ans = final_section.get("answer")
        if isinstance(ans, str):
            final_answer = ans

    # --- Save memory ---
    _record_exchange(session_id, req.question, final_answer)

    if req.trace:
        return out

    return final_section or {"answer": final_answer}

