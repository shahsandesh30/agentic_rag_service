# app/core/settings.py
from pydantic import BaseSettings

class Settings(BaseSettings):
    db_path: str = "rag_local.db"
    vector_db: str = "sqlite"
    gen_model: str = "llama-3.1-8b-instant"
    groq_api_key: str | None = None

    class Config:
        env_file = ".env"

settings = Settings()
