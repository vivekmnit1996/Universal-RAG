"""Centralized Configuration Module for Universal RAG."""

import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable override support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Application metadata
    APP_NAME: str = "Universal RAG"
    APP_VERSION: str = "2.0.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # Directories and Storage
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    UPLOAD_DIR: str = str(BASE_DIR / "uploads")
    DATA_DIR: str = str(BASE_DIR / "data")
    DB_PATH: str = str(BASE_DIR / "data" / "metadata.db")
    VECTOR_STORE_DIR: str = str(BASE_DIR / "data" / "vectorstore")

    # API Keys & Models
    OPENAI_API_KEY: str = ""
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.1
    EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Ingestion & Chunking parameters
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 100
    MAX_UPLOAD_SIZE_MB: int = 25
    ALLOWED_EXTENSIONS: List[str] = [
        ".pdf", ".csv", ".txt", ".md", ".png", ".jpg", ".jpeg"
    ]

    # Retrieval & Memory parameters
    TOP_K: int = 4
    SIMILARITY_THRESHOLD: float = 0.15  # Minimum normalized similarity score cutoff for knowledge retrieval
    MAX_RETRIES: int = 1  # LangGraph query rewrite retries on low relevance
    MAX_HISTORY_MESSAGES: int = 10  # Bounded sliding window for conversation history (5 turns)

    # Security & CORS
    CORS_ORIGINS: List[str] = ["*"]

    def ensure_directories(self) -> None:
        """Create necessary data, upload, and vectorstore directories."""
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.VECTOR_STORE_DIR, exist_ok=True)


# Singleton instance
settings = Settings()
settings.ensure_directories()
