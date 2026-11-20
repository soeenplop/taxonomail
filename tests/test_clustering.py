from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from taxonomail.clustering import BERTopicClusterer, OpenAIEmbedder
from taxonomail.config import ClusteringConfig, EmbeddingConfig
from taxonomail.llm import LLMError

DIRECTIONS = {"a": [1.0, 0.0, 0.0], "bb": [0.0, 2.0, 0.0], "ccc": [0.0, 0.0, 3.0]}


def test_openai_embedder_batches_orders_and_normalizes() -> None:
    calls: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        items = [
            SimpleNamespace(index=i, embedding=DIRECTIONS.get(text, [1.0, 1.0, 1.0]))
            for i, text in enumerate(kwargs["input"])
        ]
        return SimpleNamespace(data=list(reversed(items)))

    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    embedder = OpenAIEmbedder(EmbeddingConfig(model="emb", batch_size=2), client)
    assert embedder.embed([]).shape[0] == 0
    assert calls == []
    vectors = embedder.embed(["a", "bb", "ccc"])
    assert [c["input"] for c in calls] == [["a", "bb"], ["ccc"]]
    assert {c["model"] for c in calls} == {"emb"}
    assert np.array_equal(vectors, np.eye(3, dtype=np.float32))

    calls.clear()
    embedder.embed(["  ", "x" * 20_000, "y" * 60_000])
    sent = [text for c in calls for text in c["input"]]
    assert sent[0] == "(empty)" and len(sent) == 3
    assert max(len(text) for text in sent) <= 8000
    assert all(sum(map(len, c["input"])) <= 100_000 for c in calls)


def test_embedder_credentials_and_small_corpora() -> None:
    seen: list[dict[str, Any]] = []

    def factory(**kwargs: Any) -> Any:
        seen.append(kwargs)
        return object()

    official = "https://api.openai.com/v1"
    env = {"OPENAI_API_KEY": "sk-default", "OPENAI_BASE_URL": "https://evil.example", "K": "k"}
    OpenAIEmbedder(EmbeddingConfig(), client_factory=factory, env=env)
    OpenAIEmbedder(EmbeddingConfig(base_url=official + "/"), client_factory=factory, env=env)
    local = EmbeddingConfig(base_url="http://localhost:11434/v1")
    OpenAIEmbedder(local, client_factory=factory, env=env)
    OpenAIEmbedder(local.model_copy(update={"api_key_env": "K"}), client_factory=factory, env=env)
    assert seen == [
        {"api_key": "sk-default", "base_url": official},
        {"api_key": "sk-default", "base_url": official + "/"},
        {"api_key": "unused", "base_url": "http://localhost:11434/v1"},
        {"api_key": "k", "base_url": "http://localhost:11434/v1"},
    ]
    with pytest.raises(LLMError, match="MISSING") as info:
        OpenAIEmbedder(EmbeddingConfig(api_key_env="MISSING"), client_factory=factory, env=env)
    assert "sk-default" not in str(info.value)
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        OpenAIEmbedder(EmbeddingConfig(), client_factory=factory, env={})

    clusterer = BERTopicClusterer(ClusteringConfig(min_cluster_size=5))
    empty = np.zeros((0, 3), dtype=np.float32)
    assert clusterer.fit([], empty).labels == []
    small = clusterer.fit(["a", "b", "c"], np.eye(3, dtype=np.float32))
    assert small.labels == [-1, -1, -1] and small.keywords == {}
