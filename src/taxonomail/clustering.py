"""Embeddings (any OpenAI-compatible service) and clustering (BERTopic)."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from taxonomail.config import ClusteringConfig, EmbeddingConfig

Vectors = npt.NDArray[np.float32]
NOISE = -1


class Embedder(Protocol):
    """Turns texts into L2-normalized vectors, one row per text."""

    def embed(self, texts: Sequence[str]) -> Vectors:
        """Embed ``texts``."""
        ...


class OpenAIEmbedder:
    """Embeddings from OpenAI or any OpenAI-compatible server (Ollama, vLLM, TEI, ...)."""

    def __init__(self, config: EmbeddingConfig, client: Any = None) -> None:
        import openai

        self._config = config
        self._client: Any = client or openai.OpenAI(
            api_key=os.environ.get(config.api_key_env) or "unused", base_url=config.base_url
        )

    def embed(self, texts: Sequence[str]) -> Vectors:
        """Embed ``texts`` in batches; rows are L2-normalized."""
        rows: list[list[float]] = []
        size = self._config.batch_size
        for start in range(0, len(texts), size):
            batch = list(texts[start : start + size])
            response = self._client.embeddings.create(model=self._config.model, input=batch)
            data = sorted(response.data, key=lambda item: item.index)
            if len(data) != len(batch):
                raise RuntimeError(f"got {len(data)} embeddings for {len(batch)} texts")
            rows.extend(item.embedding for item in data)
        vectors = np.asarray(rows, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return (vectors / np.where(norms == 0, 1, norms)).astype(np.float32)


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
        from bertopic import BERTopic
        from hdbscan import HDBSCAN
        from umap import UMAP

        c = self._config
        model = BERTopic(
            umap_model=UMAP(
                n_neighbors=c.n_neighbors,
                n_components=c.n_components,
                min_dist=0.0,
                metric="cosine",
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
