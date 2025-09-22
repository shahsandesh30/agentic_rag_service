# app/api.py
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import json

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
    # --- Fetch conversation history ---
    history = get_recent_messages(req.session_id, limit=5)
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])

    # --- Combine history + new question ---
    user_input = f"{history_text}\nUser: {req.question}" if history_text else req.question

    # --- Run RAG QA ---
    payload = answer_question(
        user_input,   # 👈 enriched with history
        _searcher,
        _gen,
        top_k_ctx=req.top_k_ctx,
    )

    # --- Save both Q and A to memory ---
    save_message(req.session_id, "user", req.question)
    save_message(req.session_id, "assistant", payload.answer)

    answer_dict = json.loads(payload.dict()['answer'])

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

