import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from taxonomail import cli
from tests.fakes import TOPICS, FakeClusterer, FakeEmbedder, FakeLLM
from tests.test_sources import make_maildir

runner = CliRunner(env={"NO_COLOR": "1", "TERM": "dumb", "FORCE_COLOR": None})


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    llm = FakeLLM()
    monkeypatch.setattr(cli, "make_embedder", lambda config: FakeEmbedder())
    monkeypatch.setattr(cli, "make_clusterer", lambda config: FakeClusterer())
    monkeypatch.setattr(cli, "make_llm", lambda config: llm)
    return llm


def test_run_on_a_maildir(tmp_path: Path, fakes: FakeLLM) -> None:
    box = make_maildir(tmp_path / "box", 0)
    for i in range(9):
        (box / "cur" / f"{i}").write_bytes(
            f"From: a@example.org\nMessage-ID: <{i}@x>\n"
            f"Subject: {TOPICS[i % 3]} {i}\n\nbody".encode()
        )
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
