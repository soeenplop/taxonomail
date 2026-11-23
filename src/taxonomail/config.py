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


class ClassificationConfig(_Section):
    """Jev (TypeSafe System One): picks one folder per mail for ``classify``."""

    section: ClassVar[str] = "classification"
    model: str = "jev-latest"
    base_url: str = "https://api.typesafe.ai/v1"
    api_key_env: str = Field(default="TYPESAFE_API_KEY", pattern=ENV_NAME, repr=False)
    """Name of the environment variable that holds the key, never the key itself."""
    min_probability: float = Field(default=0.5, ge=0, le=1)
    """A mail whose best folder is less likely than this stays where it is."""
    timeout: float = Field(default=10, gt=0)
    """Seconds per request."""
    workers: int = Field(default=8, ge=1, le=64)
    """Parallel requests in ``classify``."""


class RelevanceConfig(_Section):
    """When ``classify --stale-folder`` moves a mail that is no longer worth keeping."""

    min_probability: float = Field(default=0.8, ge=0.5, le=1)
    """How sure Jev must be that the mail is no longer relevant; in doubt it stays."""
    min_age_days: int = Field(default=90, ge=0)
    """Only mails at least this old (by their Date header) are moved."""
    question: str = Field(
        default="Is this mail no longer worth keeping once it is a few weeks old?", min_length=1
    )
    stale: str = Field(
        default=(
            "Short-lived or automated mail nobody needs to read again later: one-time and "
            "two-factor codes, login and new-device notifications, newsletters, promotions "
            "and ads, payment notifications from payment services such as PayPal, app beta "
            "and TestFlight notifications, registration and e-mail confirmations, updated "
            "terms of service or privacy policies."
        ),
        min_length=1,
    )
    """What a mail that is no longer relevant looks like."""
    relevant: str = Field(
        default=(
            "Mail that may still matter later: invoices, contracts, tax and official "
            "documents, bookings and tickets, orders and warranties, account statements, "
            "access data or credentials sent to you, and mail written personally by a person."
        ),
        min_length=1,
    )
    """What a mail worth keeping looks like."""


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
    classification: ClassificationConfig = ClassificationConfig()
    relevance: RelevanceConfig = RelevanceConfig()


def load_settings(path: Path | None) -> Settings:
    """Read settings from ``path``, or return the defaults when there is no file."""
    if path is None:
        return Settings()
    with path.open("rb") as handle:
        return Settings.model_validate(tomllib.load(handle))
