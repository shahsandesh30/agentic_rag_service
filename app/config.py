# app/config.py
from functools import lru_cache
from typing import Literal

from pydantic import (
    Field,
    field_validator,  # v2 validator
)
from pydantic_settings import BaseSettings, SettingsConfigDict  # moved in v2


class Settings(BaseSettings):
    """
    Central configuration for the service (Pydantic v2).
    Loads from environment variables and a .env file (if present).
    """

    # --- Runtime / env ---
    env: Literal["dev", "prod", "test"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # --- Models / LLM ---
    llm_provider: Literal["groq", "hf"] = Field("groq", alias="LLM_PROVIDER")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = Field("llama-3.1-8b-instant", alias="GROQ_MODEL")

    # Embeddings & reranking
    embed_model: str = Field("BAAI/bge-small-en-v1.5", alias="MODEL_NAME")
    use_rerank: bool = Field(False, alias="USE_RERANK")
    reranker_model: str | None = Field(default=None, alias="RERANKER_MODEL")
    rerank_k: int = Field(20, ge=1, le=200)

    # --- Retrieval ---
    top_k: int = Field(5, ge=1, le=50)
    hybrid: bool = Field(False, alias="HYBRID_RETRIEVAL")  # (bm25 + vector) toggle

    # --- Agent / Intent ---
    intent_mode: Literal["rules", "llm", "hybrid"] = Field("hybrid", alias="INTENT_MODE")
    intent_min_query_len: int = Field(3, alias="INTENT_MIN_QUERY_LEN")
    intent_min_confidence: float = Field(0.55, ge=0.0, le=1.0, alias="INTENT_MIN_CONFIDENCE")
    intent_labels: tuple[str, ...] = ("rag", "web", "chitchat")

    # --- Data paths ---
    sqlite_path: str = Field("rag_local.db", alias="SQLITE_PATH")
    faiss_path: str = Field("faiss_index/index.faiss", alias="FAISS_PATH")

    # --- Safety ---
    enable_safety: bool = Field(True, alias="ENABLE_SAFETY")

    # Pydantic v2 config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  # env var names are case-insensitive
        extra="ignore",  # ignore extra envs silently
        populate_by_name=True,  # allow using field names as env keys too
    )

    @field_validator("groq_api_key")
    @classmethod
    def _warn_if_missing_groq(cls, v, info):
        # Warn (don’t crash) if GROQ_API_KEY is missing while provider=groq
        data = info.data if hasattr(info, "data") else {}
        provider = data.get("llm_provider", "groq")
        if provider == "groq" and not v:
            import logging

            logging.getLogger(__name__).warning(
                "GROQ_API_KEY missing; LLM calls will fail until it’s set."
            )
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings instance so every import doesn’t re-parse the env."""
    return Settings()


# Example of usage in other modules:
# from app.config import get_settings
# settings = get_settings()

# # Example usage:
# dim = settings.top_k
# embed_model = settings.embed_model
