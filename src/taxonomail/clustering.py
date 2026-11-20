"""Embeddings (any OpenAI-compatible service) and clustering (BERTopic)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from taxonomail.config import ClusteringConfig, EmbeddingConfig
from taxonomail.llm import openai_client

Vectors = npt.NDArray[np.float32]
NOISE = -1
MAX_TEXT_BYTES = 8000
MAX_BATCH_BYTES = 100_000
EMPTY = "(empty)"


class Embedder(Protocol):
    """Turns texts into L2-normalized vectors, one row per text."""

    def embed(self, texts: Sequence[str]) -> Vectors:
        """Embed ``texts``."""
        ...


def _bound(text: str) -> str:
    """At most ``MAX_TEXT_BYTES`` UTF-8 bytes (cut at a character boundary), never blank."""
    cut = text.encode()[:MAX_TEXT_BYTES].decode(errors="ignore")
    return cut.strip() or EMPTY


class OpenAIEmbedder:
    """Embeddings from OpenAI or any OpenAI-compatible server (Ollama, vLLM, TEI, ...)."""

    def __init__(
        self,
        config: EmbeddingConfig,
        client: Any = None,
        client_factory: Callable[..., Any] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> None:
        self._config = config
        if client is None:
            client = openai_client(config.base_url, config.api_key_env, env, client_factory)
        self._client: Any = client

    def embed(self, texts: Sequence[str]) -> Vectors:
        """Embed ``texts`` in batches; rows are L2-normalized and in input order."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        # Bounded, never blank, so every text yields exactly one row.
        clean = [_bound(text) for text in texts]
        rows: list[list[float]] = []
        for batch in self._batches(clean):
            response = self._client.embeddings.create(model=self._config.model, input=batch)
            data = sorted(response.data, key=lambda item: item.index)
            if len(data) != len(batch):
                raise RuntimeError(f"got {len(data)} embeddings for {len(batch)} texts")
            rows.extend(item.embedding for item in data)
        vectors = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return (vectors / np.where(norms == 0, 1, norms)).astype(np.float32)

    def _batches(self, texts: list[str]) -> list[list[str]]:
        """Batches of at most ``batch_size`` texts and ``MAX_BATCH_BYTES`` UTF-8 bytes."""
        batches: list[list[str]] = [[]]
        size = 0
        for text in texts:
            if batches[-1] and (
                len(batches[-1]) >= self._config.batch_size
                or size + len(text.encode()) > MAX_BATCH_BYTES
            ):
                batches.append([])
                size = 0
            batches[-1].append(text)
            size += len(text.encode())
        return batches


@dataclass(frozen=True)
class Clusters:
    """The clustering of all mails."""

    labels: list[int]
    """Cluster per mail, ``-1`` for noise."""
    keywords: dict[int, list[str]]
    """Most distinctive words per cluster."""

    @property
    def ids(self) -> list[int]:
        """Cluster ids without noise, sorted."""
        return sorted({label for label in self.labels if label != NOISE})


class Clusterer(Protocol):
    """Clusters documents with precomputed embeddings."""

    def fit(self, docs: Sequence[str], vectors: Vectors) -> Clusters:
        """Cluster ``docs``; row ``i`` of ``vectors`` belongs to ``docs[i]``."""
        ...


class BERTopicClusterer:
    """BERTopic with UMAP and HDBSCAN on the given embeddings."""

    def __init__(self, config: ClusteringConfig) -> None:
        self._config = config

    def fit(self, docs: Sequence[str], vectors: Vectors) -> Clusters:
        """Fit BERTopic and return labels and keywords."""
        c = self._config
        n = len(docs)
        # Too few mails for a cluster (and for UMAP's neighbors).
        if n < max(3, c.min_cluster_size):
            return Clusters(labels=[NOISE] * n, keywords={})
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from umap import UMAP

        # UMAP's spectral init needs more points than neighbors and components.
        n_neighbors = min(c.n_neighbors, n - 1)
        n_components = max(1, min(c.n_components, n - 2))
        bounded = (n_neighbors, n_components) != (c.n_neighbors, c.n_components)
        model = BERTopic(
            umap_model=UMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=0.0,
                metric="cosine",
                init="random" if bounded else "spectral",
                random_state=c.seed,
            ),
            hdbscan_model=HDBSCAN(min_cluster_size=c.min_cluster_size, prediction_data=True),
            top_n_words=c.keywords,
            calculate_probabilities=False,
            verbose=False,
        )
        topics, _ = model.fit_transform(list(docs), embeddings=vectors)
        labels = [int(t) for t in topics]
        keywords = {
            cluster: [str(word) for word, _ in (model.get_topic(cluster) or []) if str(word)]
            for cluster in sorted(set(labels) - {NOISE})
        }
        return Clusters(labels=labels, keywords=keywords)


def representatives(vectors: Vectors, labels: Sequence[int], count: int) -> dict[int, list[int]]:
    """Up to ``count`` members per cluster, closest to the cluster's mean first."""
    label_array = np.asarray(labels)
    result: dict[int, list[int]] = {}
    for cluster in sorted(set(labels) - {NOISE}):
        members = np.flatnonzero(label_array == cluster)
        similarity = vectors[members] @ vectors[members].mean(axis=0)
        order = np.argsort(-similarity, kind="stable")
        result[cluster] = [int(members[i]) for i in order[:count]]
    return result


def nearest(vectors: Vectors, centroids: Vectors) -> list[int]:
    """Index of the most similar centroid (cosine) for every row of ``vectors``."""
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    unit = centroids / np.where(norms == 0, 1, norms)
    return [int(i) for i in np.argmax(vectors @ unit.T, axis=1)]
