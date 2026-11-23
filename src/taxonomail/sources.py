"""Read raw messages from a Maildir or an IMAP folder; move messages over IMAP."""

from __future__ import annotations

import base64
import imaplib
import logging
import mailbox
import os
import re
import ssl
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

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
    with ImapMailbox(settings) as box:
        for _, raw in box.messages(batch):
            yield raw


class ImapMailbox:
    """One IMAP folder over TLS: read every message and, unless read-only, move messages."""

    def __init__(self, settings: ImapSettings, *, readonly: bool = True) -> None:
        self._folder = _quote(settings.folder)  # validate before connecting
        if settings.password_env not in os.environ:
            raise RuntimeError(f"environment variable {settings.password_env} is not set")
        self._settings = settings
        self._readonly = readonly
        self._created: set[str] = set()
        self._imap: imaplib.IMAP4_SSL | None = None

    def __enter__(self) -> ImapMailbox:
        settings = self._settings
        password = os.environ[settings.password_env]
        if any(char in value for value in (settings.user, password) for char in "\r\n\0"):
            raise ValueError("invalid IMAP credentials")
        context = ssl.create_default_context()
        imap = imaplib.IMAP4_SSL(settings.host, settings.port, ssl_context=context)
        self._imap = imap
        try:
            imap.debug = 0
            try:
                imap.login(settings.user, password)
            except imaplib.IMAP4.error:
                raise RuntimeError("IMAP login failed") from None
            except UnicodeEncodeError:
                raise RuntimeError("IMAP login failed (credentials must be ASCII)") from None
            if not self._readonly and "MOVE" not in _capabilities(imap):
                raise RuntimeError("the IMAP server does not support MOVE")
            status, _ = imap.select(self._folder, readonly=self._readonly)
            if status != "OK":
                raise RuntimeError("can't open IMAP folder")
        except BaseException:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc: object) -> None:
        if self._imap is not None:
            imap, self._imap = self._imap, None
            imap.__exit__(None, None, None)  # logs out; ignores errors on a broken connection

    @property
    def _session(self) -> imaplib.IMAP4_SSL:
        if self._imap is None:
            raise RuntimeError("IMAP mailbox is not open")
        return self._imap

    def messages(self, batch: int = 100) -> Iterator[tuple[str, bytes]]:
        """Yield ``(uid, raw bytes)`` of every message, without changing any flags."""
        imap = self._session
        status, data = imap.uid("SEARCH", "ALL")
        if status != "OK":
            raise RuntimeError("IMAP search failed")
        uids = data[0].split() if data and data[0] else []
        for start in range(0, len(uids), batch):
            chunk = b",".join(uids[start : start + batch]).decode()
            status, fetched = imap.uid("FETCH", chunk, "(UID BODY.PEEK[])")
            if status != "OK":
                raise RuntimeError("IMAP fetch failed")
            yield from _with_uids(fetched)

    def move(self, uid: str, folder: str) -> None:
        """Move one message to ``folder``, creating and subscribing the folder if needed."""
        if self._readonly:
            raise RuntimeError("IMAP mailbox is read-only")
        if not uid.isdigit():
            raise ValueError("invalid IMAP UID")
        imap = self._session
        target = _quote(folder)
        if target not in self._created:
            imap.create(target)  # fails harmlessly when the folder exists; MOVE then checks
            status, _ = imap.subscribe(target)
            if status != "OK":
                raise RuntimeError("IMAP subscribe failed")
            self._created.add(target)
        status, _ = imap.uid("MOVE", uid, target)
        if status != "OK":
            raise RuntimeError("IMAP move failed")


def _capabilities(imap: imaplib.IMAP4_SSL) -> set[str]:
    """Capabilities after login: servers may announce more than before it."""
    status, data = imap.capability()
    if status != "OK" or not data or not isinstance(data[-1], bytes):
        return set()
    return set(data[-1].decode("ascii", "replace").upper().split())


_TOKEN = re.compile(rb'"(?:[^"\\]|\\.)*"|[()\[\]]|[^\s()\[\]"]+')


def _top_level_uid(data: bytes, depth: int) -> str | None:
    """The value of a ``UID`` attribute directly inside the FETCH list (depth 1).

    ``depth`` is the nesting level at the start of ``data``: 0 for the response head, 1 for
    the trailer after a literal. Nested lists, sections and quoted strings are skipped, so a
    flag named ``UID`` in ``FLAGS (UID 42)`` is never taken for the message's UID.
    """
    atoms: list[tuple[int, bytes]] = []  # (depth, atom), with brackets as depth changes
    for match in _TOKEN.finditer(data):
        token = match.group()
        if token in (b"(", b"["):
            depth += 1
            atoms.append((-1, token))
        elif token in (b")", b"]"):
            depth -= 1
            atoms.append((-1, token))
        else:
            atoms.append((depth, token))
    for (level, name), (next_level, value) in pairwise(atoms):
        if level == next_level == 1 and name.upper() == b"UID" and value.isdigit():
            return value.decode()
    return None


def _with_uids(fetched: list[Any]) -> Iterator[tuple[str, bytes]]:
    """Pair each FETCH body with its UID, which may come before or after the body literal.

    imaplib returns a body as ``(head, literal)`` followed by the rest of the same response
    (``b")"`` or ``b" UID 11)"``). The body is complete at that trailer, so a later response
    (an unsolicited FLAGS update, say) can never lend it another message's UID.
    """
    uid: str | None = None
    body: bytes | None = None
    for item in fetched:
        if isinstance(item, tuple) and len(item) == 2:
            if body is not None:
                yield _require(uid), body
            uid, body = _top_level_uid(item[0], 0), item[1]
        elif isinstance(item, bytes) and body is not None:
            if uid is None:
                uid = _top_level_uid(item, 1)
            yield _require(uid), body
            uid, body = None, None
    if body is not None:
        yield _require(uid), body


def _require(uid: str | None) -> str:
    if uid is None:
        raise RuntimeError("IMAP server sent a message without its UID")
    return uid


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
