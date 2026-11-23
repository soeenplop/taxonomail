"""Sort mail into the folders of a finished run with Jev (TypeSafe System One).

Jev answers a Choice question: it can only return one of the folder ids it is given, with a
probability for each, so a mail cannot make it invent a folder or produce text.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Protocol

import httpx2
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from taxonomail.config import ClassificationConfig
from taxonomail.mail import Mail

MAX_FOLDERS = 255
"""Most options a Jev Choice question accepts."""
QUESTION = "Which folder of this person's mailbox does this mail belong in?"
RETRY_STATUS = {408, 429, 500, 502, 503, 504}
FATAL_STATUS = {401, 402, 403, 404}
"""Answers that every further request would get too: bad key, no credit, wrong URL."""


class ClassifyError(Exception):
    """A mail could not be classified. The message never contains mail or response text."""


class FatalClassifyError(ClassifyError):
    """Every further request would fail too (missing key, rejected key, bad request)."""


@dataclass(frozen=True)
class Category:
    """One folder of a taxonomy, as Jev sees it."""

    id: str
    name: str
    description: str
    includes: tuple[str, ...] = ()
    """Descriptions of the clusters merged into the folder."""


class _Folder(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)
    id: str = Field(min_length=1, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    name: str = Field(min_length=1)
    description: str = ""


class _Taxonomy(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)
    folders: list[_Folder] = Field(min_length=1, max_length=MAX_FOLDERS)


class _Cluster(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)
    description: str
    folder: str


def load_taxonomy(directory: Path) -> list[Category]:
    """Folders from ``taxonomy.yaml``, with cluster descriptions from ``clusters.json`` if any."""
    try:
        data = yaml.safe_load((directory / "taxonomy.yaml").read_text("utf-8"))
        folders = _Taxonomy.model_validate(data).folders
    except (OSError, yaml.YAMLError, ValidationError):
        raise ValueError(f"no valid taxonomy.yaml in {directory}") from None
    if len({f.id for f in folders}) != len(folders):
        raise ValueError("taxonomy.yaml has duplicate folder ids")
    includes: dict[str, list[str]] = {}
    clusters_file = directory / "clusters.json"
    if clusters_file.exists():
        try:
            raw = json.loads(clusters_file.read_text("utf-8"))
            clusters = [_Cluster.model_validate(c) for c in raw]
        except (OSError, ValueError, TypeError):
            raise ValueError(f"clusters.json in {directory} is not valid") from None
        for cluster in clusters:
            includes.setdefault(cluster.folder, []).append(cluster.description)
    return [Category(f.id, f.name, f.description, tuple(includes.get(f.id, ()))) for f in folders]


def folder_path(category: Category, prefix: str = "") -> str:
    """The mailbox name for a folder: its readable name, without hierarchy separators."""
    if any(char in prefix for char in "\r\n\0"):
        raise ValueError("invalid folder prefix")
    name = re.sub(r"[/.\\%*\x00-\x1f\x7f]+", "-", category.name).strip(" -")
    return prefix + (name or category.id)


def folder_paths(categories: list[Category], prefix: str = "") -> dict[str, str]:
    """Mailbox name per folder id; fails when two folders would share a mailbox."""
    paths = {c.id: folder_path(c, prefix) for c in categories}
    seen: set[str] = set()
    for path in paths.values():
        if path.casefold() in seen:
            raise ValueError("two folders get the same mailbox name; rename one in taxonomy.yaml")
        seen.add(path.casefold())
    return paths


@dataclass(frozen=True)
class Verdict:
    """The folder Jev picked, or ``None`` when its probability is below the threshold."""

    folder: str | None
    best: str
    probability: float


class Classifier(Protocol):
    """Picks a folder for one mail."""

    def classify(self, mail: Mail) -> Verdict:
        """Return the folder for ``mail``; raise ``ClassifyError`` when that fails."""
        ...


class _Answer(BaseModel):
    model_config = ConfigDict(extra="ignore", hide_input_in_errors=True)
    choice: str
    probabilities: dict[str, Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]]


def _criterion(category: Category) -> dict[str, Any]:
    criterion: dict[str, Any] = {"folder": category.name, "what": category.description}
    if category.includes:
        criterion["includes"] = list(category.includes)
    return criterion


class JevClassifier:
    """Calls ``POST {base_url}/systemone`` with one Choice question per mail."""

    def __init__(
        self,
        config: ClassificationConfig,
        categories: list[Category],
        *,
        retries: int = 3,
        env: Mapping[str, str] | None = None,
        transport: httpx2.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        key = (os.environ if env is None else env).get(config.api_key_env)
        if not key:
            raise FatalClassifyError(
                "the environment variable named by [classification] api_key_env is not set"
            )
        self._config = config
        self._ids = {c.id for c in categories}
        self._criteria = {c.id: _criterion(c) for c in categories}
        self._retries = retries
        self._sleep = sleep
        self._client = httpx2.Client(
            base_url=config.base_url.rstrip("/") + "/",
            headers={"Authorization": f"Bearer {key}"},
            timeout=config.timeout,
            transport=transport,
        )

    def close(self) -> None:
        """Close the HTTP connection pool."""
        self._client.close()

    def classify(self, mail: Mail) -> Verdict:
        """Ask Jev for the folder of ``mail``."""
        body = {
            "model": self._config.model,
            "state": mail.text(),
            "questions": {
                "folder": {"type": "choice", "instructions": QUESTION, "criteria": self._criteria}
            },
        }
        answer = _parse(self._post(body), self._ids)
        probability = answer.probabilities[answer.choice]
        folder = answer.choice if probability >= self._config.min_probability else None
        return Verdict(folder=folder, best=answer.choice, probability=probability)

    def _post(self, body: dict[str, Any]) -> httpx2.Response:
        for attempt in range(self._retries + 1):
            last = attempt == self._retries
            try:
                response = self._client.post("systemone", json=body)
            except httpx2.TransportError as exc:
                if last:
                    raise ClassifyError(f"request failed ({type(exc).__name__})") from None
            else:
                if response.status_code not in RETRY_STATUS:
                    break
                if last:
                    raise ClassifyError(f"Jev answered HTTP {response.status_code}")
            self._sleep(2**attempt)
        if response.status_code in FATAL_STATUS:
            raise FatalClassifyError(f"Jev answered HTTP {response.status_code}")
        if response.status_code >= 400:  # e.g. 413: this mail only
            raise ClassifyError(f"Jev answered HTTP {response.status_code}")
        return response


def _parse(response: httpx2.Response, ids: set[str]) -> _Answer:
    try:
        answer = _Answer.model_validate(response.json()["answers"]["folder"])
    except (ValueError, KeyError, TypeError):
        raise ClassifyError("Jev sent an answer that is not valid") from None
    if answer.choice not in ids:
        raise ClassifyError("Jev chose a folder that is not in the taxonomy")
    if answer.choice not in answer.probabilities:
        raise ClassifyError("Jev sent no probability for its choice")
    return answer
