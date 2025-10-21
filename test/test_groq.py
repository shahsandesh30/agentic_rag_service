from app.llm.groq_gen import GroqGenerator


class DummyCompletions:
    def create(self, **kwargs):
        class R:
            pass

        r = R()
        r.choices = [type("Msg", (), {"message": type("M", (), {"content": "Mock answer"})()})]
        return r


class DummyChat:
    def __init__(self):
        self.completions = DummyCompletions()


class DummyClient:
    def __init__(self):
        self.chat = DummyChat()


def test_groq_generate(monkeypatch):
    g = GroqGenerator(model="llama-3.1-8b-instant", api_key="dummy")
    monkeypatch.setattr(g, "client", DummyClient())
    out = g.generate("What is refund policy?", ["Refunds within 30 days."])
    assert "Mock answer" in out
