import imaplib
import ssl
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

    def __init__(self, host: str, port: int, ssl_context: ssl.SSLContext) -> None:
        assert ssl_context.check_hostname
        assert ssl_context.verify_mode == ssl.CERT_REQUIRED
        self.calls: list[tuple[Any, ...]] = [("connect", host, port)]
        FakeImap.instances.append(self)

    def __enter__(self) -> "FakeImap":
        return self

    def __exit__(self, *exc: object) -> None:
        self.calls.append(("logout",))

    def login(self, user: str, password: str) -> None:
        assert password == "pw"
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


def settings_for(folder: str, monkeypatch: pytest.MonkeyPatch) -> ImapSettings:
    FakeImap.instances.clear()
    monkeypatch.setattr(imaplib, "IMAP4_SSL", FakeImap)
    monkeypatch.setenv("IMAP_PASSWORD", "pw")
    return ImapSettings(host="imap.example.org", user="me", folder=folder)


def test_imap_reads_read_only_in_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = settings_for("My Box", monkeypatch)
    assert list(read_imap(settings, batch=2)) == [b"raw-1", b"raw-2", b"raw-3"]
    (imap,) = FakeImap.instances
    assert ("select", '"My Box"', True) in imap.calls
    fetches = [c for c in imap.calls if c[:2] == ("uid", "FETCH")]
    assert [c[2] for c in fetches] == ["1,2", "3"]
    assert all(c[3] == "(BODY.PEEK[])" for c in fetches)
    assert "pw" not in repr(settings)


@pytest.mark.parametrize(
    ("folder", "wire"), [("Entwürfe", '"Entw&APw-rfe"'), ("A&B", '"A&-B"'), ("a\\b", '"a\\\\b"')]
)
def test_imap_folder_is_utf7_encoded(
    folder: str, wire: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    list(read_imap(settings_for(folder, monkeypatch)))
    assert ("select", wire, True) in FakeImap.instances[0].calls


@pytest.mark.parametrize("folder", ["a\r\nb", "a\nb", "a\0b"])
def test_imap_rejects_injecting_folder_names(folder: str, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="invalid IMAP folder name"):
        list(read_imap(settings_for(folder, monkeypatch)))
    assert not FakeImap.instances
