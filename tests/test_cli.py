import email.utils
import io
import json
import os
import sys
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import ClassVar, NoReturn

import pytest
import yaml
from typer.testing import CliRunner

from taxonomail import cli
from taxonomail.classify import ClassifyError, FatalClassifyError
from taxonomail.config import RelevanceConfig
from tests.fakes import (
    TOPICS,
    FakeClassifier,
    FakeClusterer,
    FakeEmbedder,
    FakeLLM,
    raw_mail,
    write_taxonomy,
)
from tests.test_sources import make_maildir

runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb", "FORCE_COLOR": None})


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeLLM:
    monkeypatch.chdir(tmp_path)  # no .env from the checkout
    llm = FakeLLM()
    monkeypatch.setattr(cli, "make_embedder", lambda config: FakeEmbedder())
    monkeypatch.setattr(cli, "make_clusterer", lambda config: FakeClusterer())
    monkeypatch.setattr(cli, "make_llm", lambda config: llm)
    return llm


def topic_maildir(tmp_path: Path) -> Path:
    box = make_maildir(tmp_path / "box", 0)
    for i in range(9):
        (box / "cur" / f"{i}").write_bytes(
            f"From: a@example.org\nMessage-ID: <{i}@x>\n"
            f"Subject: {TOPICS[i % 3]} {i}\n\nbody".encode()
        )
    return box


def test_run_on_a_maildir(tmp_path: Path, fakes: FakeLLM) -> None:
    box = topic_maildir(tmp_path)
    out = tmp_path / "out"
    result = runner.invoke(cli.app, ["run", "--maildir", str(box), "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "read 9 mails" in result.output
    assert "3 folders" in result.output
    taxonomy = yaml.safe_load((out / "taxonomy.yaml").read_text())
    assert [f["id"] for f in taxonomy["folders"]] == list(TOPICS)
    rows = [json.loads(line) for line in (out / "assignments.jsonl").read_text().splitlines()]
    assert len(rows) == 9
    assert {r["method"] for r in rows} == {"cluster"}
    assert len(json.loads((out / "clusters.json").read_text())) == 3


def test_language_option_overrides_the_config(tmp_path: Path, fakes: FakeLLM) -> None:
    config = tmp_path / "config.toml"
    config.write_text('[taxonomy]\nlanguage = "French"\nmin_folders = 2\n')
    args = ["run", "--maildir", str(topic_maildir(tmp_path)), "--config", str(config)]
    result = runner.invoke(cli.app, [*args, "--out", str(tmp_path / "a"), "--language", "German"])
    assert result.exit_code == 0, result.output
    assert fakes.calls and all("in German" in system for system, _ in fakes.calls)
    fakes.calls.clear()
    result = runner.invoke(cli.app, [*args, "--out", str(tmp_path / "b")])
    assert result.exit_code == 0, result.output
    assert fakes.calls and all("in French" in system for system, _ in fakes.calls)


def test_bad_language_is_rejected(tmp_path: Path, fakes: FakeLLM) -> None:
    box = topic_maildir(tmp_path)
    args = ["run", "--maildir", str(box), "--out", str(tmp_path / "out"), "--language", "en_US"]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 2
    assert "language name" in result.output
    assert not fakes.calls


@pytest.fixture
def classifier(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeClassifier:
    monkeypatch.chdir(tmp_path)  # no .env from the checkout
    fake = FakeClassifier()
    monkeypatch.setattr(
        cli,
        "make_classifier",
        lambda config, categories, retries, relevance=None, topics=True: fake,
    )
    return fake


def unsure_maildir(tmp_path: Path) -> Path:
    box = topic_maildir(tmp_path)
    (box / "cur" / "x").write_bytes(b"From: a@example.org\nMessage-ID: <x@x>\nSubject: hm\n\nb")
    return box


def test_classify_lists_folders_without_moving(tmp_path: Path, classifier: FakeClassifier) -> None:
    run = write_taxonomy(tmp_path / "run")
    box = unsure_maildir(tmp_path)
    before = sorted(p.name for p in (box / "cur").iterdir())
    result = runner.invoke(cli.app, ["classify", "--taxonomy", str(run), "--maildir", str(box)])
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in (run / "classified.jsonl").read_text().splitlines()]
    assert len(rows) == 10
    assert {r["folder"] for r in rows} == {"Invoices-invoices", "Travel-travel", "Club-club", None}
    assert not any("moved" in r for r in rows)
    assert (run / "classified.jsonl").stat().st_mode & 0o777 == 0o600
    assert sorted(p.name for p in (box / "cur").iterdir()) == before
    assert "(stays: unsure)" in result.output


def test_classify_keeps_going_after_one_failed_mail(
    tmp_path: Path, classifier: FakeClassifier
) -> None:
    classifier.fail_on = "club"
    run = write_taxonomy(tmp_path / "run")
    out = tmp_path / "list.jsonl"
    args = ["classify", "--taxonomy", str(run), "--maildir", str(topic_maildir(tmp_path))]
    result = runner.invoke(cli.app, [*args, "--out", str(out), "--limit", "6"])
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 6
    assert [r["error"] for r in rows if "error" in r] == ["Jev answered HTTP 503"] * 2


def test_classify_moves_over_imap(
    tmp_path: Path, classifier: FakeClassifier, monkeypatch: pytest.MonkeyPatch
) -> None:
    raws = [raw_mail(f"{t} {i}", n=i) for i, t in enumerate([*TOPICS, "hm"])]

    class Box:
        moved: ClassVar[list[tuple[str, str]]] = []

        def __init__(self, settings: object, readonly: bool) -> None:
            assert not readonly

        def __enter__(self) -> "Box":
            return self

        def __exit__(self, *exc: object) -> None:
            pass

        def messages(self, batch: int) -> Iterator[tuple[str, bytes]]:
            yield from ((str(i + 1), raw) for i, raw in enumerate(raws))

        def move(self, uid: str, folder: str) -> bool:
            Box.moved.append((uid, folder))
            return uid != "3"  # another client took mail 3 away

    monkeypatch.setattr(cli, "ImapMailbox", Box)
    monkeypatch.setenv("IMAP_PASSWORD", "pw")
    run = write_taxonomy(tmp_path / "run")
    args = ["classify", "--taxonomy", str(run), "--imap-host", "h", "--imap-user", "u"]
    result = runner.invoke(cli.app, [*args, "--move", "--prefix", "INBOX."])
    assert result.exit_code == 0, result.output
    assert Box.moved == [
        ("1", "INBOX.Invoices-invoices"),
        ("2", "INBOX.Travel-travel"),
        ("3", "INBOX.Club-club"),
    ]
    rows = [json.loads(line) for line in (run / "classified.jsonl").read_text().splitlines()]
    assert [r.get("moved", False) for r in rows] == [True, True, "unconfirmed", False]


def test_classify_refuses_to_move_a_maildir(tmp_path: Path, classifier: FakeClassifier) -> None:
    run = write_taxonomy(tmp_path / "run")
    args = ["classify", "--taxonomy", str(run), "--maildir", str(topic_maildir(tmp_path))]
    result = runner.invoke(cli.app, [*args, "--move"])
    assert result.exit_code == 2
    assert not classifier.seen


def test_classify_stops_on_a_fatal_error(
    tmp_path: Path, classifier: FakeClassifier, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fatal(mail: object) -> NoReturn:
        raise FatalClassifyError("Jev answered HTTP 401")

    monkeypatch.setattr(classifier, "classify", fatal)
    run = write_taxonomy(tmp_path / "run")
    args = ["classify", "--taxonomy", str(run), "--maildir", str(topic_maildir(tmp_path))]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 1
    assert "HTTP 401" in result.output


def test_classify_mail_prints_the_folder(tmp_path: Path, classifier: FakeClassifier) -> None:
    run = write_taxonomy(tmp_path / "run")
    args = ["classify-mail", "--taxonomy", str(run), "--prefix", "INBOX."]
    result = runner.invoke(cli.app, args, input=raw_mail("travel plans"))
    assert result.exit_code == 0
    assert result.stdout == "INBOX.Travel-travel"
    result = runner.invoke(cli.app, args, input=raw_mail("hm"))
    assert (result.exit_code, result.stdout) == (0, "")


@pytest.mark.parametrize("problem", ["error", "taxonomy", "crash"])
def test_classify_mail_never_fails_the_delivery(
    problem: str, tmp_path: Path, classifier: FakeClassifier, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = write_taxonomy(tmp_path / "run")
    secret_subject = "club secret-subject"
    if problem == "error":
        classifier.fail_on = "club"
    elif problem == "taxonomy":
        (run / "taxonomy.yaml").write_text("folders: []")
    else:

        def crash(mail: object) -> NoReturn:
            raise RuntimeError(secret_subject)

        monkeypatch.setattr(classifier, "classify", crash)
    result = runner.invoke(
        cli.app, ["classify-mail", "--taxonomy", str(run)], input=raw_mail(secret_subject)
    )
    assert result.exit_code == 0
    assert result.stdout == ""
    assert "taxonomail:" in result.stderr
    assert "secret-subject" not in result.stderr


def test_classify_mail_reads_an_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("JEV_TEST_KEY", raising=False)
    seen: list[str | None] = []

    def factory(
        config: object,
        categories: object,
        retries: int,
        relevance: object = None,
        topics: bool = True,
    ) -> FakeClassifier:
        seen.append(os.environ.get("JEV_TEST_KEY"))
        assert retries == 0
        return FakeClassifier()

    monkeypatch.setattr(cli, "make_classifier", factory)
    env = tmp_path / "keys.env"
    env.write_text("JEV_TEST_KEY=abc\n")
    run = write_taxonomy(tmp_path / "run")
    args = ["classify-mail", "--taxonomy", str(run), "--env-file", str(env)]
    result = runner.invoke(cli.app, args, input=raw_mail("club"))
    assert result.stdout == "Club-club"
    assert seen == ["abc"]
    monkeypatch.delenv("JEV_TEST_KEY")


def test_classify_stops_when_a_whole_batch_fails(
    tmp_path: Path, classifier: FakeClassifier, monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing(mail: object) -> NoReturn:
        raise ClassifyError("Jev answered HTTP 400")

    monkeypatch.setattr(classifier, "classify", failing)
    run = write_taxonomy(tmp_path / "run")
    args = ["classify", "--taxonomy", str(run), "--maildir", str(topic_maildir(tmp_path))]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 1
    assert "HTTP 400" in result.output
    assert len((run / "classified.jsonl").read_text().splitlines()) == 9


def test_classify_replaces_an_old_list_instead_of_rewriting_it(
    tmp_path: Path, classifier: FakeClassifier
) -> None:
    run = write_taxonomy(tmp_path / "run")
    old = run / "classified.jsonl"
    old.write_text("old\n")
    old.chmod(0o644)
    elsewhere = tmp_path / "elsewhere.txt"
    elsewhere.write_text("keep\n")
    link = tmp_path / "link.jsonl"
    link.symlink_to(elsewhere)
    box = topic_maildir(tmp_path)
    with old.open() as reader:
        result = runner.invoke(cli.app, ["classify", "--taxonomy", str(run), "--maildir", str(box)])
        assert result.exit_code == 0, result.output
        assert reader.read() == "old\n"
    assert old.stat().st_mode & 0o777 == 0o600
    assert len(old.read_text().splitlines()) == 9
    args = ["classify", "--taxonomy", str(run), "--maildir", str(box), "--out", str(link)]
    assert runner.invoke(cli.app, args).exit_code == 0
    assert elsewhere.read_text() == "keep\n"
    assert not link.is_symlink()
    assert not list(run.glob(".*.tmp")) and not list(tmp_path.glob(".*.tmp"))


class Stdin:
    """A stdin whose bytes are counted as they are read."""

    def __init__(self, data: bytes) -> None:
        self.buffer = io.BytesIO(data)

    @property
    def at_eof(self) -> bool:
        return self.buffer.tell() == len(self.buffer.getvalue())


@pytest.mark.parametrize(
    "case", ["too-large", "bad-config", "taxonomy-is-a-file", "mailbox-collision"]
)
def test_classify_mail_reads_all_input_and_fails_open(
    case: str, tmp_path: Path, classifier: FakeClassifier, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = write_taxonomy(tmp_path / "run")
    args = ["--taxonomy", str(run)]
    mail = raw_mail("club", body="x" * 300_000)
    if case == "too-large":
        monkeypatch.setattr(cli, "MAX_MAIL_BYTES", 100_000)
    elif case == "bad-config":
        (tmp_path / "bad.toml").write_text("[nope]\n")
        args += ["--config", str(tmp_path / "bad.toml")]
    elif case == "taxonomy-is-a-file":
        args = ["--taxonomy", str(run / "taxonomy.yaml")]
    else:
        collide(run)
    stdin = Stdin(mail)
    monkeypatch.setattr(sys, "stdin", stdin)
    output = io.StringIO()
    errors = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setattr(sys, "stderr", errors)
    with pytest.raises(SystemExit) as info:
        cli.app(["classify-mail", *args], prog_name="taxonomail")
    assert info.value.code in (0, None)
    assert output.getvalue() == ""
    assert "taxonomail:" in errors.getvalue()
    assert stdin.at_eof
    assert not classifier.seen


def collide(run: Path) -> None:
    taxonomy = yaml.safe_load((run / "taxonomy.yaml").read_text())
    taxonomy["folders"][1]["name"] = taxonomy["folders"][0]["name"].replace("/", ".")
    (run / "taxonomy.yaml").write_text(yaml.safe_dump(taxonomy))


def test_classify_refuses_folders_that_share_a_mailbox(
    tmp_path: Path, classifier: FakeClassifier
) -> None:
    run = write_taxonomy(tmp_path / "run")
    collide(run)
    args = ["classify", "--taxonomy", str(run), "--maildir", str(topic_maildir(tmp_path))]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 2
    assert "same mailbox" in result.output
    assert not classifier.seen


OLD = "Mon, 01 Jan 2024 10:00:00 +0100"


def test_stale_folder_takes_old_mail_jev_is_sure_about(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    asked: list[object] = []
    fake = FakeClassifier(relevance=True)

    def factory(
        config: object,
        categories: object,
        retries: int,
        relevance: object = None,
        topics: bool = True,
    ) -> FakeClassifier:
        asked.append(relevance)
        return fake

    monkeypatch.setattr(cli, "make_classifier", factory)
    now = email.utils.format_datetime(datetime.now(UTC))
    mails = [
        ("club promo", OLD),  # old and surely stale: moves
        ("hm promo", OLD),  # no topic folder, but surely stale: moves
        ("club promo", now),  # stale but new: stays in its topic
        ("club maybe", OLD),  # in doubt: topic
        ("club news", OLD),  # relevant: topic
        ("club promo", None),  # no date: topic
    ]
    box = make_maildir(tmp_path / "box", 0)
    for i, (subject, date) in enumerate(mails):
        (box / "cur" / str(i)).write_bytes(raw_mail(subject, n=i, date=date))
    run = write_taxonomy(tmp_path / "run")
    args = ["classify", "--taxonomy", str(run), "--maildir", str(box)]
    result = runner.invoke(cli.app, [*args, "--stale-folder", "Alt"])
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in (run / "classified.jsonl").read_text().splitlines()]
    assert [r["folder"] for r in rows] == [
        "Alt",
        "Alt",
        "Club-club",
        "Club-club",
        "Club-club",
        "Club-club",
    ]
    assert [r["stale"] for r in rows] == [0.95, 0.95, 0.95, 0.6, 0.05, 0.95]
    assert isinstance(asked[0], RelevanceConfig)
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert asked[1] is None


@pytest.mark.parametrize("name", ["", " / ", "\t", "club/Club"])
def test_bad_stale_folder_is_rejected(
    name: str, tmp_path: Path, classifier: FakeClassifier
) -> None:
    run = write_taxonomy(tmp_path / "run")
    args = ["classify", "--taxonomy", str(run), "--maildir", str(topic_maildir(tmp_path))]
    result = runner.invoke(cli.app, [*args, "--stale-folder", name])
    assert result.exit_code == 2
    assert not classifier.seen


def test_only_stale_moves_nothing_else(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    fake = FakeClassifier(relevance=True)
    asked: list[bool] = []

    def factory(
        config: object,
        categories: object,
        retries: int,
        relevance: object = None,
        topics: bool = True,
    ) -> FakeClassifier:
        asked.append(topics)
        fake.topics = topics
        return fake

    monkeypatch.setattr(cli, "make_classifier", factory)
    raws = [
        raw_mail("club promo", n=1, date=OLD),
        raw_mail("club news", n=2, date=OLD),
        raw_mail("travel maybe", n=3, date=OLD),
    ]

    class Box:
        moved: ClassVar[list[tuple[str, str]]] = []

        def __init__(self, settings: object, readonly: bool) -> None:
            pass

        def __enter__(self) -> "Box":
            return self

        def __exit__(self, *exc: object) -> None:
            pass

        def messages(self, batch: int) -> Iterator[tuple[str, bytes]]:
            yield from ((str(i + 1), raw) for i, raw in enumerate(raws))

        def move(self, uid: str, folder: str) -> bool:
            Box.moved.append((uid, folder))
            return True

    monkeypatch.setattr(cli, "ImapMailbox", Box)
    monkeypatch.setenv("IMAP_PASSWORD", "pw")
    run = write_taxonomy(tmp_path / "run")
    args = ["classify", "--taxonomy", str(run), "--imap-host", "h", "--imap-user", "u", "--move"]
    result = runner.invoke(cli.app, [*args, "--stale-folder", "Alt", "--only-stale"])
    assert result.exit_code == 0, result.output
    assert Box.moved == [("1", "Alt")]
    assert "(stays)" in result.output and "unsure" not in result.output
    rows = [json.loads(line) for line in (run / "classified.jsonl").read_text().splitlines()]
    assert [r["folder"] for r in rows] == ["Alt", None, None]
    assert [r["best"] for r in rows] == [None, None, None]
    assert asked == [False]  # only the relevance question is sent


def test_only_stale_needs_a_stale_folder(tmp_path: Path, classifier: FakeClassifier) -> None:
    run = write_taxonomy(tmp_path / "run")
    args = ["classify", "--taxonomy", str(run), "--maildir", str(topic_maildir(tmp_path))]
    result = runner.invoke(cli.app, [*args, "--only-stale"])
    assert result.exit_code == 2
    assert not classifier.seen


@pytest.mark.parametrize(
    ("days_ago", "option", "config", "moved"),
    [
        (60, None, None, False),  # default 90 days
        (120, None, None, True),
        (60, "30", None, True),  # option overrides the default
        (60, None, 30, True),  # config overrides the default
        (60, "100", 30, False),  # option overrides the config
        (60, "1000000000", None, False),  # huge threshold: no overflow, never stale
    ],
)
def test_min_age_days(
    days_ago: int,
    option: str | None,
    config: int | None,
    moved: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    fake = FakeClassifier(relevance=True)
    monkeypatch.setattr(cli, "make_classifier", lambda *args, **kwargs: fake)
    date = email.utils.format_datetime(datetime.now(UTC) - timedelta(days=days_ago))
    box = make_maildir(tmp_path / "box", 0)
    (box / "cur" / "1").write_bytes(raw_mail("club promo", date=date))
    run = write_taxonomy(tmp_path / "run")
    args = ["classify", "--taxonomy", str(run), "--maildir", str(box), "--stale-folder", "Alt"]
    if option is not None:
        args += ["--min-age-days", option]
    if config is not None:
        (tmp_path / "c.toml").write_text(f"[relevance]\nmin_age_days = {config}\n")
        args += ["--config", str(tmp_path / "c.toml")]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    row = json.loads((run / "classified.jsonl").read_text())
    assert (row["folder"] == "Alt") is moved


def test_min_age_days_needs_a_stale_folder(tmp_path: Path, classifier: FakeClassifier) -> None:
    run = write_taxonomy(tmp_path / "run")
    args = ["classify", "--taxonomy", str(run), "--maildir", str(topic_maildir(tmp_path))]
    result = runner.invoke(cli.app, [*args, "--min-age-days", "10"])
    assert result.exit_code == 2
    assert not classifier.seen
