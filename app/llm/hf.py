import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.config import settings


def _format_prompt(tokenizer, system: str, user: str, contexts: list[str]) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})

    ctx = "\n\n".join([f"[CTX{i + 1}]\n{c}" for i, c in enumerate(contexts)])
    user_msg = user if not ctx else f"{ctx}\n\n User:\n{user}"
    msgs.append({"role": "user", "content": user_msg})
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"{system}\n\n{ctx}\n\nUser:\n{user}"


class HFGenerator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(settings.model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=settings.device_map,
            trust_remote_code=True,
        )

    @torch.inference_mode()
    def generate(
        self, prompt: str, contexts: list[str] | None = None, system: str | None = None
    ) -> str:
        system = system or "You are a concise helpful assistant."
        contexts = contexts or []
        text = _format_prompt(self.tokenizer, system, prompt, contexts)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=settings.max_new_tokens,
            do_sample=settings.do_sample,
            temperature=settings.temperature,
            top_p=settings.top_p,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        completion = self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return completion.strip()


# Shared singleton accessor to avoid multiple heavy initializations
_SHARED_GENERATOR: HFGenerator | None = None


def get_shared_generator() -> HFGenerator:
    global _SHARED_GENERATOR
    if _SHARED_GENERATOR is None:
        _SHARED_GENERATOR = HFGenerator()
    return _SHARED_GENERATOR
