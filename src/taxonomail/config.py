"""Settings, read from an optional TOML file. Every field has a default."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)


OPENAI_URL = "https://api.openai.com/v1"
DEFAULT_KEY_ENV = "OPENAI_API_KEY"
# Upper-case variable names only, so a pasted key (mixed case, "-") fails validation instead
# of ending up in an error message.
ENV_NAME = r"^[A-Z_][A-Z0-9_]{0,63}$"


class LLMConfig(_Section):
    """One LLM step: an OpenAI-compatible chat endpoint and model."""

    section: ClassVar[str] = "naming/consolidation"
    model: str
    base_url: str = OPENAI_URL
    api_key_env: str = Field(default=DEFAULT_KEY_ENV, pattern=ENV_NAME, repr=False)
    """Name of the environment variable that holds the key, never the key itself."""
    max_tokens: int = 8000


class NamingConfig(LLMConfig):
    """Defaults for the folder-naming step."""

    section: ClassVar[str] = "naming"
    model: str = "gpt-6.1-terra"
    max_tokens: int = 1000


class ConsolidationConfig(LLMConfig):
    """Defaults for the taxonomy-consolidation step."""

    section: ClassVar[str] = "consolidation"
    model: str = "gpt-6.2-astra"
    max_tokens: int = 16000


class EmbeddingConfig(_Section):
    """An OpenAI-compatible embeddings endpoint (``POST /v1/embeddings``)."""

    section: ClassVar[str] = "embedding"
    model: str = "text-embedding-3-small"
    base_url: str = OPENAI_URL
    api_key_env: str = Field(default=DEFAULT_KEY_ENV, pattern=ENV_NAME, repr=False)
    """Name of the environment variable that holds the key, never the key itself."""
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
