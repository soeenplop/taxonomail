from types import SimpleNamespace
from typing import Any

import numpy as np

from taxonomail.clustering import OpenAIEmbedder
from taxonomail.config import EmbeddingConfig


def test_openai_embedder_batches_orders_and_normalizes() -> None:
    calls: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> Any:
        calls.append(kwargs)
        batch = kwargs["input"]
        items = [
            SimpleNamespace(index=i, embedding=[float(len(text)), 0.0])
            for i, text in enumerate(batch)
        ]
        return SimpleNamespace(data=list(reversed(items)))

    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    embedder = OpenAIEmbedder(EmbeddingConfig(model="emb", batch_size=2), client)
    vectors = embedder.embed(["a", "bb", "ccc"])
    assert [c["input"] for c in calls] == [["a", "bb"], ["ccc"]]
    assert {c["model"] for c in calls} == {"emb"}
    assert vectors.shape == (3, 2)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
