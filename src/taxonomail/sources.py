"""Read raw messages from a Maildir or an IMAP folder (read-only)."""

from __future__ import annotations

import imaplib
import mailbox
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


def read_maildir(path: Path) -> Iterator[bytes]:
    """Yield the raw bytes of every message in a Maildir (``cur`` and ``new``)."""
    if not (path / "cur").is_dir() and not (path / "new").is_dir():
        raise FileNotFoundError(f"{path} is not a Maildir (no cur/ or new/)")
    box = mailbox.Maildir(path, factory=None, create=False)
    for key in sorted(box.keys()):
        with box.get_file(key) as handle:
            yield handle.read()


@dataclass(frozen=True)
class ImapSettings:
    """Where to read mail from over IMAP."""

    host: str
    user: str
    password: str
    port: int = 993
    folder: str = "INBOX"
    ssl: bool = True


def read_imap(settings: ImapSettings, batch: int = 100) -> Iterator[bytes]:
    """Yield the raw bytes of every message in one IMAP folder, without changing any flags."""
    connect = imaplib.IMAP4_SSL if settings.ssl else imaplib.IMAP4
    with connect(settings.host, settings.port) as imap:
        imap.login(settings.user, settings.password)
        status, _ = imap.select(_quote(settings.folder), readonly=True)
        if status != "OK":
            raise RuntimeError(f"can't open IMAP folder {settings.folder!r}")
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


def _quote(folder: str) -> str:
    return '"' + folder.replace("\\", "\\\\").replace('"', '\\"') + '"'
