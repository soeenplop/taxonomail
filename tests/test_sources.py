import imaplib
import ssl
import traceback
from pathlib import Path
from typing import Any, ClassVar

import pytest

from taxonomail.sources import ImapMailbox, ImapSettings, read_imap, read_maildir
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
    capabilities: tuple[str, ...] = ("IMAP4REV1", "MOVE")
    move_status = "OK"

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
        user.encode("ascii"), password.encode("ascii")  # like imaplib
        self.calls.append(("login", user))

    def select(self, folder: str, readonly: bool) -> tuple[str, list[bytes]]:
        self.calls.append(("select", folder, readonly))
        return "OK", [b"3"]

    def create(self, folder: str) -> tuple[str, list[bytes]]:
        self.calls.append(("create", folder))
        return "NO", [b"[ALREADYEXISTS] exists"]

    def subscribe(self, folder: str) -> tuple[str, list[bytes]]:
        self.calls.append(("subscribe", folder))
        return "OK", [b""]

    def uid(self, command: str, *args: str) -> tuple[str, list[Any]]:
        self.calls.append(("uid", command, *args))
        if command == "SEARCH":
            return "OK", [b"1 2 3"]
        if command == "MOVE":
            return self.move_status, [b""]
        uids = args[0].split(",")
        return "OK", [
            x
            for u in uids
            for x in (
                (b"%s (UID 1%s BODY[] {9}" % (u.encode(), u.encode()), b"raw-" + u.encode()),
                b")",
            )
        ]


def settings_for(
    folder: str, monkeypatch: pytest.MonkeyPatch, user: str = "me", password: str = "pw"
) -> ImapSettings:
    FakeImap.instances.clear()
    monkeypatch.setattr(imaplib, "IMAP4_SSL", FakeImap)
    monkeypatch.setenv("IMAP_PASSWORD", password)
    return ImapSettings(host="imap.example.org", user=user, folder=folder)


def test_imap_reads_read_only_in_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = settings_for("My Box", monkeypatch)
    assert list(read_imap(settings, batch=2)) == [b"raw-1", b"raw-2", b"raw-3"]
    (imap,) = FakeImap.instances
    assert ("select", '"My Box"', True) in imap.calls
    fetches = [c for c in imap.calls if c[:2] == ("uid", "FETCH")]
    assert [c[2] for c in fetches] == ["1,2", "3"]
    assert all(c[3] == "(UID BODY.PEEK[])" for c in fetches)
    assert "pw" not in repr(settings)


@pytest.mark.parametrize(
    ("folder", "wire"), [("Entwürfe", '"Entw&APw-rfe"'), ("A&B", '"A&-B"'), ("a\\b", '"a\\\\b"')]
)
def test_imap_folder_is_utf7_encoded(
    folder: str, wire: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    list(read_imap(settings_for(folder, monkeypatch)))
    assert ("select", wire, True) in FakeImap.instances[0].calls


@pytest.mark.parametrize(
    ("folder", "user", "password", "message", "connects"),
    [
        ("a\r\nb", "me", "pw", "invalid IMAP folder name", False),
        ("a\nb", "me", "pw", "invalid IMAP folder name", False),
        ("a\0b", "me", "pw", "invalid IMAP folder name", False),
        ("INBOX", "me\r\nX LOGOUT", "pw", "invalid IMAP credentials", False),
        ("INBOX", "me", "p\r\nX LOGOUT", "invalid IMAP credentials", False),
        ("INBOX", "me", "pä\u00dfw", "credentials must be ASCII", True),
    ],
)
def test_imap_rejects_bad_input_without_leaking(
    folder: str,
    user: str,
    password: str,
    message: str,
    connects: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = settings_for(folder, monkeypatch, user, password)
    with pytest.raises((ValueError, RuntimeError), match=message) as info:
        list(read_imap(settings))
    shown = str(info.value) + "".join(traceback.format_exception(info.value))
    assert password not in shown and "pä" not in shown
    assert bool(FakeImap.instances) == connects
    assert all(
        call[0] != "login" and call[0] != "select" for i in FakeImap.instances for call in i.calls
    )


def test_mailbox_yields_uids_and_moves_into_created_folders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = settings_for("INBOX", monkeypatch)
    with ImapMailbox(settings, readonly=False) as box:
        messages = list(box.messages())
        assert messages == [("11", b"raw-1"), ("12", b"raw-2"), ("13", b"raw-3")]
        box.move("11", "Mobilität")
        box.move("12", "Mobilität")
    (imap,) = FakeImap.instances
    assert ("select", '"INBOX"', False) in imap.calls
    target = '"Mobilit&AOQ-t"'
    assert [c for c in imap.calls if c[0] in {"create", "subscribe"}] == [
        ("create", target),
        ("subscribe", target),
    ]
    assert [c for c in imap.calls if c[:2] == ("uid", "MOVE")] == [
        ("uid", "MOVE", "11", target),
        ("uid", "MOVE", "12", target),
    ]
    assert imap.calls[-1] == ("logout",)


def test_mailbox_refuses_to_move_without_server_support(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = settings_for("INBOX", monkeypatch)
    monkeypatch.setattr(FakeImap, "capabilities", ("IMAP4REV1",))
    with pytest.raises(RuntimeError, match="MOVE"), ImapMailbox(settings, readonly=False):
        pass
    assert FakeImap.instances[0].calls[-1] == ("logout",)


def test_read_only_mailbox_never_moves(monkeypatch: pytest.MonkeyPatch) -> None:
    with ImapMailbox(settings_for("INBOX", monkeypatch)) as box, pytest.raises(RuntimeError):
        box.move("1", "X")
    assert not [c for c in FakeImap.instances[0].calls if c[:2] == ("uid", "MOVE")]
