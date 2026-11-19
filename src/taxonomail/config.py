"""Settings, read from an optional TOML file. Every field has a default."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LLMConfig(_Section):
    """One LLM step: which backend and model."""

    provider: Literal["anthropic", "openai"] = "anthropic"
    model: str
    base_url: str | None = Field(
        default=None, description="For OpenAI-compatible servers, e.g. http://localhost:11434/v1"
    )
    api_key_env: str | None = Field(
        default=None,
        description="Environment variable with the API key; default ANTHROPIC_API_KEY or "
        "OPENAI_API_KEY.",
    )
    max_tokens: int = 8000


class EmbeddingConfig(_Section):
    """An OpenAI-compatible embeddings endpoint (``POST /v1/embeddings``)."""

    model: str = "text-embedding-3-small"
    base_url: str | None = Field(
        default=None, description="Default: OpenAI. E.g. http://localhost:11434/v1 for Ollama."
    )
    api_key_env: str = Field(
        default="OPENAI_API_KEY", description="Environment variable with the API key."
    )
    batch_size: int = 64


class ClusteringConfig(_Section):
    """BERTopic: UMAP and HDBSCAN settings."""

    n_neighbors: int = 15
    n_components: int = 5
    min_cluster_size: int = 10
    representatives: int = 8
    keywords: int = 10
    seed: int = 0


class TaxonomyConfig(_Section):
    """How many folders the LLM should make."""

    min_folders: int = 8
    max_folders: int = 25


class Settings(_Section):
    """All settings."""

    naming: LLMConfig = LLMConfig(model="claude-sonnet-5-5", max_tokens=1000)
    consolidation: LLMConfig = LLMConfig(model="claude-opus-5-5", max_tokens=16000)
    embedding: EmbeddingConfig = EmbeddingConfig()
    clustering: ClusteringConfig = ClusteringConfig()
    taxonomy: TaxonomyConfig = TaxonomyConfig()


def load_settings(path: Path | None) -> Settings:
    """Read settings from ``path``, or return the defaults when there is no file."""
    if path is None:
        return Settings()
    with path.open("rb") as handle:
        return Settings.model_validate(tomllib.load(handle))
