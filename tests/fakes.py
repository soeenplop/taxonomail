"""Synthetic mail and fakes for the embedder, the clusterer and the LLM."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from email.message import EmailMessage
from pathlib import Path

import numpy as np
import yaml

from taxonomail.classify import ClassifyError, Verdict
from taxonomail.clustering import NOISE, Clusters, Vectors
from taxonomail.mail import Mail

TOPICS = ("invoices", "travel", "club")


def raw_mail(
    subject: str,
    sender: str = "shop@example.org",
    body: str = "Hello.",
    n: int = 0,
    date: str | None = None,
) -> bytes:
    """A synthetic RFC 822 message."""
    message = EmailMessage()
    if date is not None:
        message["Date"] = date
    message["From"] = sender
    message["To"] = "me@example.net"
    message["Subject"] = subject
    message["Message-ID"] = f"<m{n}@example.org>"
    message.set_content(body)
    return bytes(message)


def topic_of(text: str) -> str:
    """The first word of the subject line in an embedded text."""
    match = re.search(r"Subject: (\w+)", text)
    return match.group(1) if match else ""


class FakeEmbedder:
    """One axis per topic; mails without a known topic get a mixed vector."""

    def embed(self, texts: Sequence[str]) -> Vectors:
        rows = []
        for text in texts:
            topic = topic_of(text)
            row = np.full(len(TOPICS), 0.1, dtype=np.float32)
            if topic in TOPICS:
                row[TOPICS.index(topic)] = 1.0
            rows.append(row / np.linalg.norm(row))
        return np.stack(rows)


class FakeClusterer:
    """Cluster = index of the subject's topic word; anything else is noise."""

    def fit(self, docs: Sequence[str], vectors: Vectors) -> Clusters:
        labels = [TOPICS.index(t) if (t := topic_of(d)) in TOPICS else NOISE for d in docs]
        keywords = {i: [TOPICS[i], "mail"] for i in set(labels) - {NOISE}}
        return Clusters(labels=labels, keywords=keywords)


Plan = Callable[[list[int]], dict[str, object]]


def one_folder_per_cluster(cluster_ids: list[int]) -> dict[str, object]:
    return {
        "folders": [
            {"id": TOPICS[c], "name": TOPICS[c].title(), "description": "d", "clusters": [c]}
            for c in cluster_ids
        ]
    }


class FakeLLM:
    """Answers naming and consolidation requests from the prompt text; records every call."""

    def __init__(self, plans: Sequence[Plan] = (one_folder_per_cluster,)) -> None:
        self.plans = list(plans)
        self.calls: list[tuple[str, list[dict[str, str]]]] = []

    def chat(self, system: str, messages: list[dict[str, str]]) -> str:
        self.calls.append((system, messages))
        user = messages[0]["content"]
        if "Representative mails" in user:
            topic = re.search(r"Keywords: <mail>\"(\w+)", user)
            word = topic.group(1) if topic else "unknown"
            return json.dumps({"name": f"{word} mails", "description": f"Mails about {word}."})
        listed = re.search(r"exactly one folder: ([\d, ]+)", user)
        assert listed is not None
        ids = [int(i) for i in listed.group(1).split(",")]
        plan = self.plans.pop(0) if len(self.plans) > 1 else self.plans[0]
        return "```json\n" + json.dumps(plan(ids)) + "\n```"


class FakeClassifier:
    """Picks the folder named by the subject's first word; "unsure" mails get no folder.

    With ``relevance``, a subject containing "promo" is surely stale (0.95), "maybe" is in
    doubt (0.6) and anything else is relevant (0.05).
    """

    def __init__(self, fail_on: str = "", relevance: bool = False, topics: bool = True) -> None:
        self.fail_on = fail_on
        self.relevance = relevance
        self.topics = topics
        self.seen: list[Mail] = []

    def classify(self, mail: Mail) -> Verdict:
        self.seen.append(mail)
        topic = mail.subject.split(" ")[0] if mail.subject else ""
        if self.fail_on and topic == self.fail_on:
            raise ClassifyError("Jev answered HTTP 503")
        stale = None
        if self.relevance:
            stale = 0.95 if "promo" in mail.subject else 0.6 if "maybe" in mail.subject else 0.05
        if not self.topics:
            return Verdict(folder=None, best=None, probability=None, stale=stale)
        if topic in TOPICS:
            return Verdict(folder=topic, best=topic, probability=0.9, stale=stale)
        return Verdict(folder=None, best=TOPICS[0], probability=0.3, stale=stale)


def write_taxonomy(directory: Path) -> Path:
    """A run's output for the three TOPICS, as ``taxonomail run`` writes it."""
    directory.mkdir(parents=True, exist_ok=True)
    folders = [
        {"id": t, "name": f"{t.title()}/{t}", "description": f"All {t}.", "mails": 3}
        for t in TOPICS
    ]
    (directory / "taxonomy.yaml").write_text(yaml.safe_dump({"folders": folders}), "utf-8")
    clusters = [
        {"cluster": i, "name": t, "description": f"{t} mails", "folder": t}
        for i, t in enumerate(TOPICS)
    ]
    (directory / "clusters.json").write_text(json.dumps(clusters), "utf-8")
    return directory
