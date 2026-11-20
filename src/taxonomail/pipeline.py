"""The pipeline: embed, cluster, name the clusters, merge them into folders, assign every mail."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field, field_validator

from taxonomail import prompts
from taxonomail.clustering import NOISE, Clusterer, Clusters, Embedder, nearest, representatives
from taxonomail.config import Settings
from taxonomail.llm import LLM, ask_json, wrap_mail
from taxonomail.mail import Mail

EXCERPT_CHARS = 500
NAMING_WORKERS = 8

Log = Callable[[str], None]


class ClusterName(BaseModel):
    """The LLM's name for one cluster."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)


def slug(text: str) -> str:
    """Lowercase ASCII slug: accents dropped, other runs of non-alphanumerics become "-"."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")


class FolderDraft(BaseModel):
    """One folder as the LLM proposes it."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str
    clusters: list[int]

    @field_validator("id")
    @classmethod
    def _slug(cls, value: str) -> str:
        """Folder ids are ASCII slugs whatever the language of the names."""
        if not (cleaned := slug(value)):
            raise ValueError("id must contain letters a-z or digits")
        return cleaned


class FolderPlan(BaseModel):
    """The LLM's answer to the consolidation step."""

    folders: list[FolderDraft] = Field(min_length=1)


@dataclass(frozen=True)
class Folder:
    """A folder of the final taxonomy."""

    id: str
    name: str
    description: str
    size: int


@dataclass(frozen=True)
class Assignment:
    """Which folder a mail goes to, and how that was decided."""

    mail: Mail
    folder: str
    method: str
    """``cluster`` (the mail's cluster belongs to the folder) or ``nearest`` (noise mail)."""


@dataclass(frozen=True)
class Result:
    """The taxonomy and the folder of every mail."""

    folders: list[Folder]
    assignments: list[Assignment]
    cluster_names: dict[int, ClusterName]
    cluster_folders: dict[int, str]


def name_clusters(
    mails: Sequence[Mail], vectors: np.ndarray, clusters: Clusters, llm: LLM, settings: Settings
) -> dict[int, ClusterName]:
    """Ask the LLM for a name and description of every cluster."""
    reps = representatives(vectors, clusters.labels, settings.clustering.representatives)
    system = prompts.NAMING_SYSTEM.format(language=settings.taxonomy.language)

    def name(cluster: int) -> ClusterName:
        shown = "\n\n".join(
            "- "
            + wrap_mail(
                f"From: {mails[i].sender_domain}\nSubject: {mails[i].subject}\n"
                f"Body: {mails[i].body[:EXCERPT_CHARS]}"
            )
            for i in reps[cluster]
        )
        user = prompts.NAMING_USER.format(
            cluster_id=cluster,
            size=clusters.labels.count(cluster),
            keywords=wrap_mail(", ".join(clusters.keywords.get(cluster, []))),
            mails=shown,
        )
        return ask_json(llm, system, user, ClusterName)

    with ThreadPoolExecutor(max_workers=NAMING_WORKERS) as pool:
        names = list(pool.map(name, clusters.ids))
    return dict(zip(clusters.ids, names, strict=True))


def _min_folders(settings: Settings, cluster_ids: Sequence[int]) -> int:
    return min(settings.taxonomy.min_folders, len(cluster_ids))


def _folder_count(plan: FolderPlan, cluster_ids: Sequence[int]) -> int:
    """Distinct folders left after ``_repair``: each known cluster goes to its first folder."""
    known = set(cluster_ids)
    owner: dict[int, str] = {}
    for folder in plan.folders:
        for cluster in folder.clusters:
            if cluster in known:
                owner.setdefault(cluster, folder.id)
    return len(set(owner.values()))


def _count_problem(plan: FolderPlan, cluster_ids: Sequence[int], settings: Settings) -> str | None:
    count = _folder_count(plan, cluster_ids)
    low = _min_folders(settings, cluster_ids)
    high = settings.taxonomy.max_folders
    if low <= count <= high:
        return None
    return f"the plan has {count} folders with clusters, but it must have {low} to {high}"


def _problems(plan: FolderPlan, cluster_ids: Sequence[int], settings: Settings) -> list[str]:
    seen: dict[int, int] = {}
    for folder in plan.folders:
        for cluster in folder.clusters:
            seen[cluster] = seen.get(cluster, 0) + 1
    problems = [f"cluster {c} is in no folder" for c in cluster_ids if c not in seen]
    problems += [f"cluster {c} is in {n} folders" for c, n in seen.items() if n > 1]
    problems += [f"cluster {c} does not exist" for c in seen if c not in set(cluster_ids)]
    # Positions, not ids: ids are model-generated text.
    first: dict[str, int] = {}
    for position, folder in enumerate(plan.folders, 1):
        if folder.id in first:
            problems.append(f"folders {first[folder.id]} and {position} use the same id")
        first.setdefault(folder.id, position)
    if (problem := _count_problem(plan, cluster_ids, settings)) is not None:
        problems.append(problem)
    return problems


def _unique_ids(plan: FolderPlan) -> FolderPlan:
    """Give folders that share an id a numeric suffix: "cafe", "cafe-2", ..."""
    seen: set[str] = set()
    folders = []
    for folder in plan.folders:
        unique, n = folder.id, 2
        while unique in seen:
            unique, n = f"{folder.id}-{n}", n + 1
        seen.add(unique)
        folders.append(folder.model_copy(update={"id": unique}))
    return FolderPlan(folders=folders)


def consolidate(
    clusters: Clusters, names: dict[int, ClusterName], llm: LLM, settings: Settings, log: Log
) -> FolderPlan:
    """Ask the LLM to merge the clusters into folders; one retry if the mapping is off."""
    rows = [
        wrap_mail(
            json.dumps(
                {
                    "id": cluster,
                    "mails": clusters.labels.count(cluster),
                    "name": names[cluster].name,
                    "description": names[cluster].description,
                    "keywords": clusters.keywords.get(cluster, [])[:6],
                },
                ensure_ascii=False,
            )
        )
        for cluster in clusters.ids
    ]
    system = prompts.CONSOLIDATION_SYSTEM.format(
        min_folders=_min_folders(settings, clusters.ids),
        max_folders=settings.taxonomy.max_folders,
        language=settings.taxonomy.language,
    )
    user = prompts.CONSOLIDATION_USER.format(
        clustered=sum(1 for label in clusters.labels if label != NOISE),
        cluster_count=len(clusters.ids),
        noise=clusters.labels.count(NOISE),
        clusters="\n".join(rows),
        cluster_ids=", ".join(str(c) for c in clusters.ids),
    )
    plan = ask_json(llm, system, user, FolderPlan)
    problems = _problems(plan, clusters.ids, settings)
    if problems:
        log(f"folder plan has {len(problems)} problems, asking again")
        retry = f"{user}\n\nA previous answer had these problems:\n" + "\n".join(problems)
        plan = _unique_ids(ask_json(llm, system, retry, FolderPlan))
        if (problem := _count_problem(plan, clusters.ids, settings)) is not None:
            raise ValueError(problem)
    return plan


def _folder_centroids(
    vectors: np.ndarray, clusters: Clusters, mapping: dict[int, str]
) -> tuple[list[str], np.ndarray]:
    """Folder ids and the mean vector of each folder's clustered mails."""
    labels = np.asarray(clusters.labels)
    folder_ids = sorted(set(mapping.values()))
    centroids = np.stack(
        [
            vectors[np.isin(labels, [c for c, f in mapping.items() if f == fid])].mean(axis=0)
            for fid in folder_ids
        ]
    )
    return folder_ids, centroids


def _repair(plan: FolderPlan, clusters: Clusters, vectors: np.ndarray) -> dict[int, str]:
    """Cluster -> folder id: first mention wins, unknown ids are dropped, and clusters the
    plan left out go to the folder whose clusters are most similar."""
    mapping: dict[int, str] = {}
    known = set(clusters.ids)
    for folder in plan.folders:
        for cluster in folder.clusters:
            if cluster in known and cluster not in mapping:
                mapping[cluster] = folder.id
    if not mapping:
        raise ValueError("the folder plan assigns no known cluster to any folder")
    missing = [c for c in clusters.ids if c not in mapping]
    if missing:
        labels = np.asarray(clusters.labels)
        folder_ids, centroids = _folder_centroids(vectors, clusters, mapping)
        means = np.stack([vectors[labels == c].mean(axis=0) for c in missing])
        for cluster, index in zip(missing, nearest(means, centroids), strict=True):
            mapping[cluster] = folder_ids[index]
    return mapping


def assign(
    mails: Sequence[Mail], vectors: np.ndarray, clusters: Clusters, mapping: dict[int, str]
) -> list[Assignment]:
    """Clustered mails get their cluster's folder, noise mails the most similar folder."""
    folder_ids, centroids = _folder_centroids(vectors, clusters, mapping)
    noise = [i for i, label in enumerate(clusters.labels) if label == NOISE]
    near = dict(zip(noise, nearest(vectors[noise], centroids), strict=True)) if noise else {}
    return [
        Assignment(mail, folder_ids[near[i]], "nearest")
        if label == NOISE
        else Assignment(mail, mapping[label], "cluster")
        for i, (mail, label) in enumerate(zip(mails, clusters.labels, strict=True))
    ]


def build_taxonomy(
    mails: Sequence[Mail],
    *,
    embedder: Embedder,
    clusterer: Clusterer,
    naming: LLM,
    consolidation: LLM,
    settings: Settings,
    log: Log = print,
) -> Result:
    """Run the whole pipeline on ``mails``."""
    log(f"embedding {len(mails)} mails")
    vectors = embedder.embed([m.text() for m in mails])
    log("clustering")
    clusters = clusterer.fit([m.text() for m in mails], vectors)
    if not clusters.ids:
        raise ValueError("clustering found no clusters; try more mails or a smaller cluster size")
    log(f"{len(clusters.ids)} clusters, {clusters.labels.count(NOISE)} mails in none")
    log("naming clusters")
    names = name_clusters(mails, vectors, clusters, naming, settings)
    log("merging clusters into folders")
    plan = consolidate(clusters, names, consolidation, settings, log)
    mapping = _repair(plan, clusters, vectors)
    assignments = assign(mails, vectors, clusters, mapping)
    sizes = {f.id: 0 for f in plan.folders}
    for a in assignments:
        sizes[a.folder] += 1
    seen: set[str] = set()
    folders = []
    for draft in plan.folders:
        if draft.id in seen or sizes[draft.id] == 0:
            continue
        seen.add(draft.id)
        folders.append(Folder(draft.id, draft.name, draft.description, sizes[draft.id]))
    return Result(folders, assignments, names, mapping)
