# app/llm/groq_gen.py
from __future__ import annotations

import os

from groq import Groq


class GroqGenerator:
    """
    Groq API wrapper for text generation.
    """

    def __init__(self, model: str | None = None, api_key: str | None = None):
        self.model = model or os.getenv("GEN_MODEL", "llama-3.1-8b-instant")
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing GROQ_API_KEY in env or passed explicitly")
        self.client = Groq(api_key=self.api_key)

    def generate(
        self,
        prompt: str,
        contexts: list[str] = [],
        system: str = "You are a grounded QA assistant.",
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str:
        msgs = [{"role": "system", "content": system}]
        if contexts:
            ctx = "\n\n".join(contexts)
            msgs.append({"role": "user", "content": f"Context:\n{ctx}"})
        msgs.append({"role": "user", "content": prompt})

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
