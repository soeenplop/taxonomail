"""Command line interface: ``taxonomail run``, ``classify`` and ``classify-mail``."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import UTC, datetime, timedelta
from itertools import batched, islice
from pathlib import Path
from typing import IO, Annotated

import typer
from dotenv import find_dotenv, load_dotenv
from pydantic import ValidationError

from taxonomail.classify import (
    Category,
    Classifier,
    ClassifyError,
    FatalClassifyError,
    JevClassifier,
    Verdict,
    folder_paths,
    load_taxonomy,
)
from taxonomail.clustering import (
    BERTopicClusterer,
    Clusterer,
    Embedder,
    OpenAIEmbedder,
)
from taxonomail.config import (
    ClassificationConfig,
    ClusteringConfig,
    EmbeddingConfig,
    LLMConfig,
    RelevanceConfig,
    Settings,
    TaxonomyConfig,
    load_settings,
)
from taxonomail.llm import LLM, OpenAILLM
from taxonomail.mail import Mail, parse_mail
from taxonomail.output import write_result
from taxonomail.pipeline import build_taxonomy
from taxonomail.sources import ImapMailbox, ImapSettings, read_imap, read_maildir

app = typer.Typer(
    help="Discover a folder taxonomy for a mailbox with BERTopic and an LLM.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode=None,
)

# Factories, replaced by fakes in tests.
make_embedder: Callable[[EmbeddingConfig], Embedder] = OpenAIEmbedder
make_clusterer: Callable[[ClusteringConfig], Clusterer] = BERTopicClusterer
make_llm: Callable[[LLMConfig], LLM] = OpenAILLM


def make_classifier(
    config: ClassificationConfig,
    categories: list[Category],
    retries: int,
    relevance: RelevanceConfig | None = None,
) -> Classifier:
    return JevClassifier(config, categories, retries=retries, relevance=relevance)


STALE = "-stale"
"""Key of the --stale-folder among the folder ids (no slug starts with "-")."""


MAX_MAIL_BYTES = 50 * 1024 * 1024
"""classify-mail reads at most this much of a mail from stdin."""
BATCH = 100


@app.callback()
def main() -> None:
    """Discover a folder taxonomy for a mailbox with BERTopic and an LLM."""


def _with_language(settings: Settings, language: str) -> Settings:
    try:
        taxonomy = TaxonomyConfig.model_validate(
            settings.taxonomy.model_dump() | {"language": language}
        )
    except ValidationError:
        raise typer.BadParameter(
            "--language must be a language name such as German", param_hint="--language"
        ) from None
    return settings.model_copy(update={"taxonomy": taxonomy})


def _imap_settings(
    host: str | None, user: str | None, password_env: str, port: int, folder: str
) -> ImapSettings:
    assert host is not None
    if user is None:
        raise typer.BadParameter("--imap-host needs --imap-user")
    if password_env not in os.environ:
        raise typer.BadParameter(f"set the IMAP password in ${password_env}")
    return ImapSettings(host=host, user=user, password_env=password_env, port=port, folder=folder)


def _load(raw_mails: Iterable[bytes], limit: int | None) -> list[Mail]:
    mails: dict[str, Mail] = {}
    for raw in raw_mails:
        mail = parse_mail(raw)
        mails.setdefault(mail.id, mail)
        if limit is not None and len(mails) >= limit:
            break
    return list(mails.values())


@app.command()
def run(
    out: Annotated[Path, typer.Option(help="Directory for the output files.")],
    maildir: Annotated[
        Path | None, typer.Option(help="Read mail from this Maildir.", file_okay=False)
    ] = None,
    imap_host: Annotated[str | None, typer.Option(help="Read mail from this IMAP server.")] = None,
    imap_user: Annotated[str | None, typer.Option(help="IMAP user name.")] = None,
    imap_password_env: Annotated[
        str, typer.Option(help="Environment variable that holds the IMAP password.")
    ] = "IMAP_PASSWORD",
    imap_port: Annotated[int, typer.Option(help="IMAP port.")] = 993,
    imap_folder: Annotated[str, typer.Option(help="IMAP folder to read.")] = "INBOX",
    config: Annotated[
        Path | None, typer.Option(help="TOML settings file (models, clustering).", dir_okay=False)
    ] = None,
    limit: Annotated[int | None, typer.Option(min=1, help="Use at most this many mails.")] = None,
    language: Annotated[
        str | None,
        typer.Option(
            help="Language of the folder and cluster names, e.g. German. "
            "Overrides [taxonomy] language in --config; the default is English."
        ),
    ] = None,
) -> None:
    """Read a mailbox, cluster it with BERTopic, and let an LLM turn the clusters into folders.

    Writes taxonomy.yaml (the folders), assignments.jsonl (the folder of every mail) and
    clusters.json (the named clusters) to --out.
    """
    load_dotenv(find_dotenv(usecwd=True))
    settings = load_settings(config)
    if language is not None:
        settings = _with_language(settings, language)
    if (maildir is None) == (imap_host is None):
        raise typer.BadParameter("give either --maildir or --imap-host")
    if maildir is not None:
        raw = read_maildir(maildir)
    else:
        raw = read_imap(
            _imap_settings(imap_host, imap_user, imap_password_env, imap_port, imap_folder)
        )
    mails = _load(raw, limit)
    typer.echo(f"read {len(mails)} mails")
    if not mails:
        raise typer.Exit(1)
    result = build_taxonomy(
        mails,
        embedder=make_embedder(settings.embedding),
        clusterer=make_clusterer(settings.clustering),
        naming=make_llm(settings.naming),
        consolidation=make_llm(settings.consolidation),
        settings=settings,
        log=typer.echo,
    )
    write_result(result, out)
    methods = Counter(a.method for a in result.assignments)
    typer.echo(
        f"{len(result.folders)} folders; {methods['cluster']} mails by cluster, "
        f"{methods['nearest']} by nearest folder"
    )
    for folder in result.folders:
        typer.echo(f"  {folder.id:<30} {folder.size:>6}  {folder.name}")
    typer.echo(f"written to {out}")


def _load_env(env_file: Path | None) -> None:
    if env_file is None:
        load_dotenv(find_dotenv(usecwd=True))
    elif not load_dotenv(env_file):
        raise typer.BadParameter("no settings in that file", param_hint="--env-file")


def _taxonomy(directory: Path) -> list[Category]:
    try:
        return load_taxonomy(directory)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--taxonomy") from None


EnvFile = Annotated[
    Path | None,
    typer.Option(
        help="Read keys from this .env file instead of searching for .env.", dir_okay=False
    ),
]
TaxonomyDir = Annotated[
    Path,
    typer.Option(help="Output directory of a run (taxonomy.yaml, clusters.json).", file_okay=False),
]
Prefix = Annotated[
    str,
    typer.Option(
        help='Put this before every folder name, separator included, e.g. "INBOX." '
        "for folders under the inbox."
    ),
]
ConfigFile = Annotated[
    Path | None, typer.Option(help="TOML settings file ([classification]).", dir_okay=False)
]


@app.command()
def classify(
    taxonomy: TaxonomyDir,
    maildir: Annotated[
        Path | None, typer.Option(help="Classify the mail in this Maildir.", file_okay=False)
    ] = None,
    imap_host: Annotated[
        str | None, typer.Option(help="Classify mail on this IMAP server.")
    ] = None,
    imap_user: Annotated[str | None, typer.Option(help="IMAP user name.")] = None,
    imap_password_env: Annotated[
        str, typer.Option(help="Environment variable that holds the IMAP password.")
    ] = "IMAP_PASSWORD",
    imap_port: Annotated[int, typer.Option(help="IMAP port.")] = 993,
    imap_folder: Annotated[str, typer.Option(help="IMAP folder to classify.")] = "INBOX",
    move: Annotated[
        bool, typer.Option(help="Move each mail into its folder (IMAP only); creates folders.")
    ] = False,
    prefix: Prefix = "",
    out: Annotated[
        Path | None,
        typer.Option(
            help="Where to write the list; default classified.jsonl in --taxonomy.", dir_okay=False
        ),
    ] = None,
    config: ConfigFile = None,
    env_file: EnvFile = None,
    limit: Annotated[
        int | None, typer.Option(min=1, help="Classify at most this many mails.")
    ] = None,
    stale_folder: Annotated[
        str | None,
        typer.Option(
            help="Also ask whether each mail is still worth keeping, and send old mails that "
            "are not to this folder (see [relevance])."
        ),
    ] = None,
) -> None:
    """Sort existing mail into the folders of a run with Jev.

    Writes the folder of every mail to classified.jsonl. Mails whose best folder is less
    likely than [classification] min_probability stay where they are. With --stale-folder,
    a mail at least [relevance] min_age_days old that Jev is sure (min_probability) is no
    longer worth keeping goes to that folder instead. Nothing moves without --move.
    """
    _load_env(env_file)
    settings = load_settings(config)
    categories = _taxonomy(taxonomy)
    if (maildir is None) == (imap_host is None):
        raise typer.BadParameter("give either --maildir or --imap-host")
    if move and maildir is not None:
        raise typer.BadParameter("--move needs --imap-host", param_hint="--move")
    folders = list(categories)
    if stale_folder is not None:
        if not stale_folder.strip(" -./\\%*"):
            raise typer.BadParameter("give a folder name", param_hint="--stale-folder")
        folders.append(Category(STALE, stale_folder, ""))
    try:
        paths = folder_paths(folders, prefix)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    relevance = settings.relevance if stale_folder is not None else None
    try:
        classifier = make_classifier(settings.classification, categories, 3, relevance)
    except FatalClassifyError as exc:
        raise typer.BadParameter(str(exc)) from None
    now = datetime.now(UTC)
    out = out or taxonomy / "classified.jsonl"
    counts: Counter[str] = Counter()
    done = 0

    def judge(mail: Mail) -> Verdict | ClassifyError:
        try:
            return classifier.classify(mail)
        except FatalClassifyError:
            raise
        except ClassifyError as exc:
            return exc

    with ExitStack() as stack:
        box: ImapMailbox | None = None
        source: Iterator[tuple[str | None, bytes]]
        if maildir is not None:
            source = ((None, raw) for raw in read_maildir(maildir))
        else:
            imap = _imap_settings(imap_host, imap_user, imap_password_env, imap_port, imap_folder)
            box = stack.enter_context(ImapMailbox(imap, readonly=not move))
            source = box.messages(BATCH)
        if limit is not None:
            source = islice(source, limit)
        handle = stack.enter_context(_private_file(out))
        pool = stack.enter_context(ThreadPoolExecutor(settings.classification.workers))
        for chunk in batched(source, BATCH):
            mails = [parse_mail(raw) for _, raw in chunk]
            try:
                verdicts = list(pool.map(judge, mails))
            except FatalClassifyError as exc:
                typer.echo(f"stopped: {exc}", err=True)
                raise typer.Exit(1) from None
            for (uid, _), mail, verdict in zip(chunk, mails, verdicts, strict=True):
                row: dict[str, object] = {
                    "message_id": mail.id,
                    "from": mail.sender,
                    "subject": mail.subject,
                }
                if isinstance(verdict, ClassifyError):
                    row |= {"folder": None, "error": str(verdict)}
                    counts["(not classified: error)"] += 1
                else:
                    target = paths[verdict.folder] if verdict.folder else None
                    if relevance is not None and _is_stale(mail, verdict, relevance, now):
                        target = paths[STALE]
                    row |= {
                        "folder": target,
                        "best": verdict.best,
                        "probability": verdict.probability,
                    }
                    if verdict.stale is not None:
                        row["stale"] = verdict.stale
                    counts[target or "(stays: unsure)"] += 1
                    if box is not None and move and uid is not None and target is not None:
                        row["moved"] = True if box.move(uid, target) else "unconfirmed"
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            done += len(chunk)
            typer.echo(f"classified {done} mails", err=True)
            if all(isinstance(v, ClassifyError) for v in verdicts):
                typer.echo(
                    f"stopped: no mail of a batch could be classified ({verdicts[0]})", err=True
                )
                raise typer.Exit(1)
    for folder, count in counts.most_common():
        typer.echo(f"  {count:>6}  {folder}")
    typer.echo(f"written to {out}")


def _is_stale(mail: Mail, verdict: Verdict, relevance: RelevanceConfig, now: datetime) -> bool:
    """Old enough and surely no longer worth keeping; anything in doubt is kept."""
    return (
        verdict.stale is not None
        and verdict.stale >= relevance.min_probability
        and mail.date is not None
        and now - mail.date >= timedelta(days=relevance.min_age_days)
    )


def _private_file(path: Path) -> IO[str]:
    """A new owner-only file that replaces ``path``: it lists senders and subjects.

    Replacing the directory entry (instead of truncating) keeps the old file's readers on
    the old file and never writes through a symlink.
    """
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        os.replace(temp, path)
    except BaseException:
        os.close(fd)
        os.unlink(temp)
        raise
    return os.fdopen(fd, "w", encoding="utf-8")


def _read_stdin(limit: int) -> bytes | None:
    """All of stdin (a mail filter must read to EOF); ``None`` when it exceeds ``limit``."""
    kept = bytearray()
    too_big = False
    while chunk := sys.stdin.buffer.read(1 << 16):
        if not too_big:
            kept += chunk
            if len(kept) > limit:
                too_big = True
                kept.clear()
    return None if too_big else bytes(kept)


@app.command("classify-mail")
def classify_mail(
    # Plain strings, not paths: Typer's path checks would exit 2 and fail the delivery.
    taxonomy: Annotated[
        str, typer.Option(help="Output directory of a run (taxonomy.yaml, clusters.json).")
    ],
    prefix: Prefix = "",
    config: Annotated[
        str | None, typer.Option(help="TOML settings file ([classification]).")
    ] = None,
    env_file: Annotated[
        str | None,
        typer.Option(help="Read keys from this .env file instead of searching for .env."),
    ] = None,
) -> None:
    """Print the folder for one mail read from stdin, for a mail filter such as Sieve.

    Prints the folder name without a newline, or nothing when Jev is unsure. Never fails the
    delivery: on any error it prints nothing, reports the error on stderr and exits 0.
    It always reads stdin to the end; a mail over 50 MiB is not classified.
    """
    try:
        raw = _read_stdin(MAX_MAIL_BYTES)
        if raw is None:
            typer.echo("taxonomail: mail too large, not classified", err=True)
            return
        _load_env(None if env_file is None else Path(env_file))
        settings = load_settings(None if config is None else Path(config))
        categories = load_taxonomy(Path(taxonomy))
        paths = folder_paths(categories, prefix)
        verdict = make_classifier(settings.classification, categories, 0).classify(parse_mail(raw))
    except ClassifyError as exc:
        typer.echo(f"taxonomail: {exc}", err=True)
        return
    except Exception as exc:  # fail open: the mail is delivered to the inbox
        typer.echo(f"taxonomail: {type(exc).__name__}", err=True)
        return
    if verdict.folder is not None:
        typer.echo(paths[verdict.folder], nl=False)
