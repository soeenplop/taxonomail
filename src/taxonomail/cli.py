"""Command line interface: ``taxonomail run``."""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from taxonomail.clustering import (
    BERTopicClusterer,
    Clusterer,
    Embedder,
    OpenAIEmbedder,
)
from taxonomail.config import ClusteringConfig, EmbeddingConfig, LLMConfig, load_settings
from taxonomail.llm import LLM, build_llm
from taxonomail.mail import Mail, parse_mail
from taxonomail.output import write_result
from taxonomail.pipeline import build_taxonomy
from taxonomail.sources import ImapSettings, read_imap, read_maildir

app = typer.Typer(
    help="Discover a folder taxonomy for a mailbox with BERTopic and an LLM.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode=None,
)

# Factories, replaced by fakes in tests.
make_embedder: Callable[[EmbeddingConfig], Embedder] = OpenAIEmbedder
make_clusterer: Callable[[ClusteringConfig], Clusterer] = BERTopicClusterer
make_llm: Callable[[LLMConfig], LLM] = build_llm


@app.callback()
def main() -> None:
    """Discover a folder taxonomy for a mailbox with BERTopic and an LLM."""


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
    imap_no_ssl: Annotated[bool, typer.Option(help="Connect without TLS.")] = False,
    config: Annotated[
        Path | None, typer.Option(help="TOML settings file (models, clustering).", dir_okay=False)
    ] = None,
    limit: Annotated[int | None, typer.Option(help="Use at most this many mails.")] = None,
) -> None:
    """Read a mailbox, cluster it with BERTopic, and let an LLM turn the clusters into folders.

    Writes taxonomy.yaml (the folders), assignments.jsonl (the folder of every mail) and
    clusters.json (the named clusters) to --out.
    """
    load_dotenv()
    settings = load_settings(config)
    if (maildir is None) == (imap_host is None):
        raise typer.BadParameter("give either --maildir or --imap-host")
    if maildir is not None:
        raw = read_maildir(maildir)
    else:
        assert imap_host is not None
        if imap_user is None:
            raise typer.BadParameter("--imap-host needs --imap-user")
        password = os.environ.get(imap_password_env)
        if password is None:
            raise typer.BadParameter(f"set the IMAP password in ${imap_password_env}")
        raw = read_imap(
            ImapSettings(
                host=imap_host,
                user=imap_user,
                password=password,
                port=imap_port,
                folder=imap_folder,
                ssl=not imap_no_ssl,
            )
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
