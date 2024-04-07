"""Application configuration using Pydantic Settings."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_name: str = "RAG Chatbot"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "llama3.1:8b"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_timeout: float = 120.0

    # Vector Store (FAISS)
    vector_store_path: Path = Field(default=Path("./data/vector_store"))
    index_name: str = "rag_index"

    # RAG Pipeline
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    similarity_threshold: float = 0.7
    rerank_top_n: int = 3

    # Document Processing
    data_dir: Path = Field(default=Path("./data/documents"))
    supported_extensions: list[str] = Field(
        default=[".pdf", ".txt", ".md", ".py", ".js", ".ts", ".json", ".yaml", ".yml"]
    )

    # API
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = Field(default=["http://localhost:3000", "http://localhost:8501"])

    # Streaming
    stream_response: bool = True


settings = Settings()
