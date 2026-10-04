"""Environment settings (secrets, paths, providers). Loaded once via pydantic-settings.

Runtime-tunable pipeline defaults and guardrail settings live separately in
``app.core.runtime_config`` because they are editable through ``PUT /api/config``.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Process-level settings sourced from environment variables / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        env_ignore_empty=True,  # "OPENAI_API_KEY=" means unset, not an empty key
        extra="ignore",
    )

    app_env: Literal["development", "production", "test"] = "development"
    log_level: str = "INFO"
    data_dir: Path = BACKEND_ROOT / "data"
    cors_origins: str = "http://localhost:3000"  # comma-separated

    # LLM provider (generation, KG extraction, guardrail classifier, eval judge)
    llm_provider: Literal["openai", "anthropic", "ollama"] = "openai"
    openai_api_key: SecretStr | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-4.1-mini"
    openai_model_fast: str = "gpt-4.1-mini"
    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-sonnet-5-5"
    anthropic_model_fast: str = "claude-haiku-4-5-20251001"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # Embeddings / reranker / tokenizer (local by default, independent of llm_provider)
    embedding_provider: Literal["local", "openai"] = "local"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    openai_embedding_model: str = "text-embedding-3-small"
    reranker_model: str = "BAAI/bge-reranker-base"
    tokenizer_encoding: str = "cl100k_base"

    # Stores
    chroma_collection: str = "chunks"
    neo4j_uri: str | None = None
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr | None = None

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "graphrag.db"

    @property
    def chroma_path(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def runtime_config_path(self) -> Path:
        return self.data_dir / "runtime_config.json"

    @property
    def uploads_dir(self) -> Path:
        """Original uploaded files, by SHA-256 (eval re-chunks them at other chunk sizes)."""
        return self.data_dir / "uploads"

    @property
    def eval_indexes_dir(self) -> Path:
        """Throwaway per-chunk_size indexes built by eval runs."""
        return self.data_dir / "eval_indexes"

    @property
    def graph_backend(self) -> Literal["neo4j", "networkx"]:
        return "neo4j" if self.neo4j_uri else "networkx"


@lru_cache
def get_settings() -> Settings:
    """Return the cached process settings."""
    return Settings()
