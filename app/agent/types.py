# app/agent/types.py
from __future__ import annotations
from typing import List, Literal, TypedDict, Optional, Dict, Any

Intent = Literal["rag", "chitchat"]

class AgentState(TypedDict, total=False):
    question: str
    intent: Intent
    rewrites: List[str]                                                                                           
    answers: List[Dict[str, Any]]   # list of AnswerPayload.dict()
    best: Dict[str, Any]            # chosen AnswerPayload.dict()
    trace: List[Dict[str, Any]]     # breadcrumbs for debugging
