"""Settings, read from an optional TOML file. Every field has a default."""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


OPENAI_URL = "https://api.openai.com/v1"


class LLMConfig(_Section):
    """One LLM step: an OpenAI-compatible chat endpoint and model."""

    model: str
    base_url: str = OPENAI_URL
    max_tokens: int = 8000


class NamingConfig(LLMConfig):
    """Defaults for the folder-naming step."""

    model: str = "gpt-6.1-terra"
    max_tokens: int = 1000


class ConsolidationConfig(LLMConfig):
    """Defaults for the taxonomy-consolidation step."""

    model: str = "gpt-6.2-astra"
    max_tokens: int = 16000


class EmbeddingConfig(_Section):
    """An OpenAI-compatible embeddings endpoint (``POST /v1/embeddings``)."""

    model: str = "text-embedding-3-small"
    base_url: str = OPENAI_URL
    batch_size: int = Field(default=64, ge=1)


class ClusteringConfig(_Section):
    """BERTopic: UMAP and HDBSCAN settings."""

    n_neighbors: int = Field(default=15, ge=2)
    n_components: int = Field(default=5, ge=1)
    min_cluster_size: int = Field(default=10, ge=2)
    representatives: int = 8
    keywords: int = 10
    seed: int = 0


# Words of letters, joined by single spaces, hyphens or apostrophes (' and U+2019).
LANGUAGE_PATTERN = "^[^\\W\\d_]+(?:['\u2019 -][^\\W\\d_]+)*$"


class TaxonomyConfig(_Section):
    """How many folders the LLM should make, and the language of their names."""

    min_folders: int = 8
    max_folders: int = 25
    language: str = Field(default="English", max_length=40, pattern=LANGUAGE_PATTERN)
    """Language of folder and cluster names and descriptions, e.g. "German"."""

    @model_validator(mode="after")
    def _check_range(self) -> TaxonomyConfig:
        if self.min_folders > self.max_folders:
            raise ValueError("min_folders must not exceed max_folders")
        return self


class Settings(_Section):
    """All settings."""

    naming: NamingConfig = NamingConfig()
    consolidation: ConsolidationConfig = ConsolidationConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    clustering: ClusteringConfig = ClusteringConfig()
    taxonomy: TaxonomyConfig = TaxonomyConfig()


def load_settings(path: Path | None) -> Settings:
    """Read settings from ``path``, or return the defaults when there is no file."""
    if path is None:
        return Settings()
    with path.open("rb") as handle:
        return Settings.model_validate(tomllib.load(handle))
