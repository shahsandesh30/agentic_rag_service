# tests/conftest.py
import json

import pytest
from fastapi.testclient import TestClient

import app.api as api


class _StubGen:
    def generate(self, prompt: str, system: str = "") -> str:
        return f"[stubbed LLM] {prompt[:20]}..."


class _StubSearcher:
    def __init__(self):
        self.calls = []

    def search(self, query: str, top_k: int = 5):
        self.calls.append((query, top_k))
        # Minimal shape your API returns: {"hits": [...]} in /search
        return [
            {
                "chunk_id": "c1",
                "text": "Example chunk text",
                "section": "Intro",
                "source": "tests/sample.txt",
                "score": 42.0,
                "rank": 1,
            }
        ]


class _StubAnswerPayload:
    """Mimics your answer_question return so api.py can json.loads(payload.dict()['answer'])."""

    def __init__(self, text: str = "stubbed answer"):
        # Your /ask uses `json.loads(payload.dict()['answer'])["answer"]`
        self._text = text

    def dict(self):
        # payload.dict()['answer'] -> json string
        return {"answer": json.dumps({"answer": self._text})}


@pytest.fixture(autouse=True)
def patch_heavy_components(monkeypatch):
    """
    Replace heavy components with stubs so tests are fast and offline.
    This fixture is autouse → applies to all tests automatically.
    """
    # Stub LLM generator
    monkeypatch.setattr(api, "_gen", _StubGen(), raising=True)

    # Stub Searcher
    monkeypatch.setattr(api, "_searcher", _StubSearcher(), raising=True)

    # Stub memory store (no DB needed)
    monkeypatch.setattr(api, "get_recent_messages", lambda session_id, limit=5: [], raising=True)
    monkeypatch.setattr(api, "save_message", lambda session_id, role, content: None, raising=True)

    # Stub answer_question function
    def _stub_answer_question(user_input, searcher, gen, top_k_ctx=8):
        return _StubAnswerPayload("stubbed answer")

    monkeypatch.setattr(api, "answer_question", _stub_answer_question, raising=True)

    yield


@pytest.fixture(scope="session")
def client():
    # Important: import app.api AFTER monkeypatch in other tests if needed
    return TestClient(api.app)
