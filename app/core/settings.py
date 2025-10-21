# app/core/settings.py

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    # --- Database & Storage ---
    db_path: str = Field("rag_local.db", description="Path to SQLite database")
    vector_db: str = Field(
        "sqlite", description="Vector backend: sqlite | faiss | qdrant | pgvector"
    )

    # --- Embeddings ---
    embed_model: str = Field("BAAI/bge-small-en-v1.5", description="Embedding model ID")

    # --- LLM Backend ---
    gen_backend: str = Field("groq", description="LLM backend: hf | groq | together")
    gen_model: str = Field("llama-3.1-8b-instant", description="Default generation model")
    groq_api_key: str | None = Field(None, description="Groq API key")
    hf_model: str = Field("meta-llama/Llama-2-7b-hf", description="HF local model (if backend=hf)")

    # --- API Server ---
    api_host: str = Field("0.0.0.0", description="FastAPI host")
    api_port: int = Field(8000, description="FastAPI port")
    cors_allow_origins: str = Field("*", description="Allowed CORS origins (comma separated)")

    # --- Observability ---
    prometheus_enabled: bool = True
    log_level: str = Field("INFO", description="Log level")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# global settings instance
settings = Settings()
