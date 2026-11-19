"""The pipeline: embed, cluster, name the clusters, merge them into folders, assign every mail."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, Field

from taxonomail import prompts
from taxonomail.clustering import NOISE, Clusterer, Clusters, Embedder, nearest, representatives
from taxonomail.config import Settings
from taxonomail.llm import LLM, ask_json
from taxonomail.mail import Mail

EXCERPT_CHARS = 500
NAMING_WORKERS = 8

Log = Callable[[str], None]


class ClusterName(BaseModel):
    """The LLM's name for one cluster."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class FolderDraft(BaseModel):
    """One folder as the LLM proposes it."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str
    clusters: list[int]


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

    def name(cluster: int) -> ClusterName:
        shown = "\n\n".join(
            f"- From: {mails[i].sender_domain}\n  Subject: {mails[i].subject}\n"
            f"  Body: {mails[i].body[:EXCERPT_CHARS]}"
            for i in reps[cluster]
        )
        user = prompts.NAMING_USER.format(
            cluster_id=cluster,
            size=clusters.labels.count(cluster),
            keywords=", ".join(clusters.keywords.get(cluster, [])),
            mails=shown,
        )
        return ask_json(llm, prompts.NAMING_SYSTEM, user, ClusterName)

    with ThreadPoolExecutor(max_workers=NAMING_WORKERS) as pool:
        names = list(pool.map(name, clusters.ids))
    return dict(zip(clusters.ids, names, strict=True))


def _problems(plan: FolderPlan, cluster_ids: Sequence[int]) -> list[str]:
    seen: dict[int, int] = {}
    for folder in plan.folders:
        for cluster in folder.clusters:
            seen[cluster] = seen.get(cluster, 0) + 1
    problems = [f"cluster {c} is in no folder" for c in cluster_ids if c not in seen]
    problems += [f"cluster {c} is in {n} folders" for c, n in seen.items() if n > 1]
    problems += [f"cluster {c} does not exist" for c in seen if c not in set(cluster_ids)]
    ids = [f.id for f in plan.folders]
    problems += [f"folder id {i} is used twice" for i in sorted(set(ids)) if ids.count(i) > 1]
    return problems


def consolidate(
    clusters: Clusters, names: dict[int, ClusterName], llm: LLM, settings: Settings, log: Log
) -> FolderPlan:
    """Ask the LLM to merge the clusters into folders; one retry if the mapping is off."""
    rows = [
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
        for cluster in clusters.ids
    ]
    system = prompts.CONSOLIDATION_SYSTEM.format(
        min_folders=settings.taxonomy.min_folders, max_folders=settings.taxonomy.max_folders
    )
    user = prompts.CONSOLIDATION_USER.format(
        clustered=sum(1 for label in clusters.labels if label != NOISE),
        cluster_count=len(clusters.ids),
        noise=clusters.labels.count(NOISE),
        clusters="\n".join(rows),
        cluster_ids=", ".join(str(c) for c in clusters.ids),
    )
    plan = ask_json(llm, system, user, FolderPlan)
    problems = _problems(plan, clusters.ids)
    if problems:
        log(f"folder plan has {len(problems)} problems, asking again")
        retry = f"{user}\n\nA previous answer had these problems:\n" + "\n".join(problems)
        plan = ask_json(llm, system, retry, FolderPlan)
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
