"""Read raw messages from a Maildir or an IMAP folder (read-only)."""

from __future__ import annotations

import base64
import imaplib
import logging
import mailbox
import os
import ssl
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


def read_maildir(path: Path) -> Iterator[bytes]:
    """Yield the raw bytes of every message in a Maildir (``cur`` and ``new``)."""
    if not (path / "cur").is_dir() and not (path / "new").is_dir():
        raise FileNotFoundError(f"{path} is not a Maildir (no cur/ or new/)")
    box = mailbox.Maildir(path, factory=None, create=False)
    for key in sorted(box.keys()):
        try:
            with box.get_file(key) as handle:
                raw = handle.read()
        except (OSError, KeyError):
            log.warning("skipping unreadable message %s", key)
            continue
        yield raw


@dataclass(frozen=True)
class ImapSettings:
    """Where to read mail from over IMAP (TLS only). The password is never stored here."""

    host: str
    user: str
    password_env: str = "IMAP_PASSWORD"
    port: int = 993
    folder: str = "INBOX"


def read_imap(settings: ImapSettings, batch: int = 100) -> Iterator[bytes]:
    """Yield the raw bytes of every message in one IMAP folder, without changing any flags."""
    folder = _quote(settings.folder)  # validate before connecting
    if settings.password_env not in os.environ:
        raise RuntimeError(f"environment variable {settings.password_env} is not set")
    context = ssl.create_default_context()
    with imaplib.IMAP4_SSL(settings.host, settings.port, ssl_context=context) as imap:
        imap.debug = 0
        try:
            imap.login(settings.user, os.environ[settings.password_env])
        except imaplib.IMAP4.error:
            raise RuntimeError("IMAP login failed") from None
        status, _ = imap.select(folder, readonly=True)
        if status != "OK":
            raise RuntimeError("can't open IMAP folder")
        status, data = imap.uid("SEARCH", "ALL")
        if status != "OK":
            raise RuntimeError("IMAP search failed")
        uids = data[0].split() if data and data[0] else []
        for start in range(0, len(uids), batch):
            chunk = b",".join(uids[start : start + batch]).decode()
            status, fetched = imap.uid("FETCH", chunk, "(BODY.PEEK[])")
            if status != "OK":
                raise RuntimeError("IMAP fetch failed")
            for item in fetched:
                if isinstance(item, tuple) and len(item) == 2:
                    yield item[1]


def _utf7(name: str) -> str:
    """Encode a folder name as IMAP modified UTF-7 (RFC 3501 section 5.1.3)."""
    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if run:
            b64 = base64.b64encode("".join(run).encode("utf-16-be")).decode()
            out.append("&" + b64.rstrip("=").replace("/", ",") + "-")
            run.clear()

    for char in name:
        if " " <= char <= "~":
            flush()
            out.append("&-" if char == "&" else char)
        else:
            run.append(char)
    flush()
    return "".join(out)


def _quote(folder: str) -> str:
    if any(char in folder for char in "\r\n\0"):
        raise ValueError("invalid IMAP folder name")
    return '"' + _utf7(folder).replace("\\", "\\\\").replace('"', '\\"') + '"'
