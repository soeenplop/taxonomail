"""Command line interface: ``taxonomail run``, ``classify`` and ``classify-mail``."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
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
    folder_path,
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
    config: ClassificationConfig, categories: list[Category], retries: int
) -> Classifier:
    return JevClassifier(config, categories, retries=retries)


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
) -> None:
    """Sort existing mail into the folders of a run with Jev.

    Writes the folder of every mail to classified.jsonl. Mails whose best folder is less
    likely than [classification] min_probability stay where they are. Nothing moves
    without --move.
    """
    _load_env(env_file)
    settings = load_settings(config)
    categories = _taxonomy(taxonomy)
    if (maildir is None) == (imap_host is None):
        raise typer.BadParameter("give either --maildir or --imap-host")
    if move and maildir is not None:
        raise typer.BadParameter("--move needs --imap-host", param_hint="--move")
    try:
        paths = {c.id: folder_path(c, prefix) for c in categories}
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--prefix") from None
    try:
        classifier = make_classifier(settings.classification, categories, 3)
    except FatalClassifyError as exc:
        raise typer.BadParameter(str(exc)) from None
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
                    row |= {
                        "folder": target,
                        "best": verdict.best,
                        "probability": verdict.probability,
                    }
                    counts[target or "(stays: unsure)"] += 1
                    if box is not None and move and uid is not None and target is not None:
                        box.move(uid, target)
                        row["moved"] = True
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            done += len(chunk)
            typer.echo(f"classified {done} mails", err=True)
    for folder, count in counts.most_common():
        typer.echo(f"  {count:>6}  {folder}")
    typer.echo(f"written to {out}")


def _private_file(path: Path) -> IO[str]:
    """Open ``path`` for writing, readable only by the owner: it lists senders and subjects."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    return os.fdopen(fd, "w", encoding="utf-8")


@app.command("classify-mail")
def classify_mail(
    taxonomy: TaxonomyDir,
    prefix: Prefix = "",
    config: ConfigFile = None,
    env_file: EnvFile = None,
) -> None:
    """Print the folder for one mail read from stdin, for a mail filter such as Sieve.

    Prints the folder name without a newline, or nothing when Jev is unsure. Never fails the
    delivery: on any error it prints nothing, reports the error type on stderr and exits 0.
    """
    try:
        _load_env(env_file)
        settings = load_settings(config)
        categories = load_taxonomy(taxonomy)
        paths = {c.id: folder_path(c, prefix) for c in categories}
        raw = sys.stdin.buffer.read(MAX_MAIL_BYTES)
        verdict = make_classifier(settings.classification, categories, 0).classify(parse_mail(raw))
    except ClassifyError as exc:
        typer.echo(f"taxonomail: {exc}", err=True)
        return
    except Exception as exc:  # fail open: the mail is delivered to the inbox
        typer.echo(f"taxonomail: {type(exc).__name__}", err=True)
        return
    if verdict.folder is not None:
        typer.echo(paths[verdict.folder], nl=False)
