from __future__ import annotations
from typing import Dict, Any
from langgraph.graph import StateGraph, END

from app.agent.types import AgentState
from app.agent.router import route
from app.agent.researcher import make_rewrites
from app.agent.answer import answer_with_rewrites
from app.agent.compliance import check

from app.retrieval.vector import VectorSearcher
from app.retrieval.bm25 import BM25Searcher
from app.retrieval.hybrid import HybridSearcher
from app.llm.hf import HFGenerator

# singletons
_V = VectorSearcher(db_path="rag_local.db", model_name="BAAI/bge-small-en-v1.5")
_B = BM25Searcher(db_path="rag_local.db")
_H = HybridSearcher(_V, _B, db_path="rag_local.db")
_GEN = HFGenerator()

def node_router(state: AgentState) -> AgentState:
    intent = route(state["question"])
    state["intent"] = intent
    state.setdefault("trace", []).append({"node": "router", "intent": intent})
    return state

def node_research(state: AgentState) -> AgentState:
    if state.get("intent") != "rag":
        state["rewrites"] = []
        state.setdefault("trace", []).append({"node": "researcher", "skipped": True})
        return state
    rw = make_rewrites(state["question"], max_rewrites=3)
    state["rewrites"] = rw
    state.setdefault("trace", []).append({"node": "researcher", "rewrites": rw})
    return state

def node_answer(state: AgentState) -> AgentState:
    if state.get("intent") == "chitchat":
        # direct LLM, no context
        out = _GEN.generate(
            prompt=state["question"],
            contexts=[],
            system=(
                "Casual assistant. Be brief. "
                "If user asks for private or dangerous info, decline."
            ),
        )
        payload = {
            "answer": out.strip(),
            "citations": [],
            "confidence": 0.5,
            "safety": {"blocked": False}
        }
        state["best"] = payload
        state.setdefault("trace", []).append({"node": "answerer", "mode": "chitchat"})
        return state

    bundle = answer_with_rewrites(
        state["question"],
        state.get("rewrites", [state["question"]]),
        searcher=_H,
        generator=_GEN,
        top_k_ctx=8,
        confidence_gate=0.65,
    )
    state["answers"] = bundle["answers"]
    state["best"] = bundle["best"]
    state.setdefault("trace", []).append({
        "node": "answerer",
        "tried": len(bundle["answers"]),
        "best_conf": bundle["best"].get("confidence", 0.0) if bundle["best"] else 0.0
    })
    return state

def node_compliance(state: AgentState) -> AgentState:
    state["best"] = check(state["best"])
    state.setdefault("trace", []).append({"node": "compliance", "blocked": state["best"]["safety"]["blocked"]})
    return state

# Build the graph
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("router", node_router)
    g.add_node("researcher", node_research)
    g.add_node("answerer", node_answer)
    g.add_node("compliance", node_compliance)

    g.set_entry_point("router")
    g.add_edge("router", "researcher")
    g.add_edge("researcher", "answerer")
    g.add_edge("answerer", "compliance")
    g.add_edge("compliance", END)
    return g.compile()

# Convenience runner
def run_agent(question: str) -> Dict[str, Any]:
    app = build_graph()
    state: AgentState = {"question": question}
    final = app.invoke(state)
    return {"final": final.get("best"), "trace": final.get("trace", [])}
