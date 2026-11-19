import imaplib
from pathlib import Path
from typing import Any, ClassVar

import pytest

from taxonomail.sources import ImapSettings, read_imap, read_maildir
from tests.fakes import raw_mail


def make_maildir(root: Path, count: int) -> Path:
    for sub in ("cur", "new", "tmp"):
        (root / sub).mkdir(parents=True)
    for i in range(count):
        (root / ("new" if i % 2 else "cur") / f"{i}.mail").write_bytes(raw_mail(f"s{i}", n=i))
    return root


def test_maildir_reads_cur_and_new(tmp_path: Path) -> None:
    raws = list(read_maildir(make_maildir(tmp_path / "box", 3)))
    assert len(raws) == 3
    assert all(raw.startswith(b"From:") for raw in raws)


class FakeImap:
    instances: ClassVar[list["FakeImap"]] = []

    def __init__(self, host: str, port: int) -> None:
        self.calls: list[tuple[Any, ...]] = [("connect", host, port)]
        FakeImap.instances.append(self)

    def __enter__(self) -> "FakeImap":
        return self

    def __exit__(self, *exc: object) -> None:
        self.calls.append(("logout",))

    def login(self, user: str, password: str) -> None:
        self.calls.append(("login", user))

    def select(self, folder: str, readonly: bool) -> tuple[str, list[bytes]]:
        self.calls.append(("select", folder, readonly))
        return "OK", [b"3"]

    def uid(self, command: str, *args: str) -> tuple[str, list[Any]]:
        self.calls.append(("uid", command, *args))
        if command == "SEARCH":
            return "OK", [b"1 2 3"]
        uids = args[0].split(",")
        return "OK", [
            x for u in uids for x in ((b"%s (BODY[] {9}" % u.encode(), b"raw-" + u.encode()), b")")
        ]


def test_imap_reads_read_only_in_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(imaplib, "IMAP4_SSL", FakeImap)
    settings = ImapSettings(host="imap.example.org", user="me", password="pw", folder="My Box")
    assert list(read_imap(settings, batch=2)) == [b"raw-1", b"raw-2", b"raw-3"]
    (imap,) = FakeImap.instances
    assert ("select", '"My Box"', True) in imap.calls
    fetches = [c for c in imap.calls if c[:2] == ("uid", "FETCH")]
    assert [c[2] for c in fetches] == ["1,2", "3"]
    assert all(c[3] == "(BODY.PEEK[])" for c in fetches)
