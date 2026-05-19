from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-3-5-haiku-20241022"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    # Local model settings (used when RunRequest.mode == 'local')
    local_model: str | None = "gpt2"
    allow_local_generation: bool = False
    max_concurrent_api_calls: int = 8
    run_storage_dir: Path = Path("backend/data/runs")
    model_storage_dir: Path = Path("backend/data/models")
    allow_embedding_model_download: bool = False
    enable_umap_hdbscan: bool = False

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.run_storage_dir.mkdir(parents=True, exist_ok=True)
    settings.model_storage_dir.mkdir(parents=True, exist_ok=True)
    return settings
