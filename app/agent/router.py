# app/agent/router.py
from app.agent.intent import IntentClassifier

_intent = IntentClassifier()  # module-level reuse


def classify_intent(user_text: str) -> dict:
    """
    Returns: {"label": "rag|web|chitchat", "confidence": float, "reasons": str}
    """
    res = _intent.classify(user_text)
    return {"label": res.label, "confidence": res.confidence, "reasons": res.reasons}
