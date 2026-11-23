import json
from pathlib import Path

import httpx2
import pytest

from taxonomail.classify import (
    Category,
    ClassifyError,
    FatalClassifyError,
    JevClassifier,
    folder_path,
    load_taxonomy,
)
from taxonomail.config import ClassificationConfig
from taxonomail.mail import Mail
from tests.fakes import TOPICS, write_taxonomy

MAIL = Mail(id="<1@x>", sender="a@example.org", sender_domain="example.org", subject="Hi", body="b")
KEY = "test-key-123"


def answer(choice: str, probabilities: dict[str, float]) -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "model": "jev-1",
            "answers": {
                "folder": {
                    "type": "choice",
                    "choice": choice,
                    "confidence": 0.5,
                    "probabilities": probabilities,
                }
            },
        },
    )


@pytest.fixture
def run(tmp_path: Path) -> Path:
    return write_taxonomy(tmp_path / "run")


def make(
    run: Path, handler: httpx2.MockTransport, retries: int = 0, **config: object
) -> tuple[JevClassifier, list[float]]:
    sleeps: list[float] = []
    classifier = JevClassifier(
        ClassificationConfig.model_validate(config),
        load_taxonomy(run),
        retries=retries,
        env={"TYPESAFE_API_KEY": KEY},
        transport=handler,
        sleep=sleeps.append,
    )
    return classifier, sleeps


def test_request_and_answer(run: Path) -> None:
    requests: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return answer("travel", {"invoices": 0.1, "travel": 0.8, "club": 0.1})

    classifier, _ = make(run, httpx2.MockTransport(handler))
    verdict = classifier.classify(MAIL)
    assert (verdict.folder, verdict.best, verdict.probability) == ("travel", "travel", 0.8)
    (request,) = requests
    assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    body = json.loads(request.content)
    assert body["model"] == "jev-latest"
    assert body["state"] == MAIL.text()
    question = body["questions"]["folder"]
    assert question["type"] == "choice"
    assert list(question["criteria"]) == list(TOPICS)
    assert question["criteria"]["club"] == {
        "folder": "Club/club",
        "what": "All club.",
        "includes": ["club mails"],
    }


def test_unsure_answer_keeps_the_mail_in_place(run: Path) -> None:
    classifier, _ = make(
        run,
        httpx2.MockTransport(lambda r: answer("club", {"club": 0.45, "travel": 0.55 - 0.1})),
        min_probability=0.6,
    )
    verdict = classifier.classify(MAIL)
    assert verdict.folder is None
    assert verdict.best == "club"


@pytest.mark.parametrize(
    "response",
    [
        httpx2.Response(200, text="not json"),
        httpx2.Response(200, json={"answers": {}}),
        answer("elsewhere", {"elsewhere": 1.0}),
    ],
)
def test_bad_answers_fail_without_echoing_them(response: httpx2.Response, run: Path) -> None:
    classifier, _ = make(run, httpx2.MockTransport(lambda r: response))
    with pytest.raises(ClassifyError) as info:
        classifier.classify(MAIL)
    assert "elsewhere" not in str(info.value) and "not json" not in str(info.value)


def test_retries_rate_limits_and_server_errors(run: Path) -> None:
    statuses = iter([429, 503])

    def handler(request: httpx2.Request) -> httpx2.Response:
        status = next(statuses, 200)
        if status != 200:
            return httpx2.Response(status, text="Hi from the mail")
        return answer("club", {"club": 1.0})

    classifier, sleeps = make(run, httpx2.MockTransport(handler), retries=3)
    assert classifier.classify(MAIL).folder == "club"
    assert sleeps == [1, 2]


def test_gives_up_after_retries(run: Path) -> None:
    classifier, sleeps = make(run, httpx2.MockTransport(lambda r: httpx2.Response(503)), retries=2)
    with pytest.raises(ClassifyError, match="HTTP 503") as info:
        classifier.classify(MAIL)
    assert not isinstance(info.value, FatalClassifyError)
    assert sleeps == [1, 2]


def test_transport_errors_are_retried_then_reported(run: Path) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("boom", request=request)

    classifier, sleeps = make(run, httpx2.MockTransport(handler), retries=1)
    with pytest.raises(ClassifyError, match="ConnectError"):
        classifier.classify(MAIL)
    assert sleeps == [1]


def test_rejected_key_is_fatal_and_not_retried(run: Path) -> None:
    classifier, sleeps = make(
        run, httpx2.MockTransport(lambda r: httpx2.Response(401, text=KEY)), retries=3
    )
    with pytest.raises(FatalClassifyError, match="HTTP 401") as info:
        classifier.classify(MAIL)
    assert KEY not in str(info.value)
    assert sleeps == []


def test_missing_key_never_names_a_custom_variable(run: Path) -> None:
    config = ClassificationConfig(api_key_env="MY_JEV_KEY")
    with pytest.raises(FatalClassifyError) as info:
        JevClassifier(config, load_taxonomy(run), env={})
    assert "MY_JEV_KEY" not in str(info.value)


def test_taxonomy_without_clusters(run: Path) -> None:
    (run / "clusters.json").unlink()
    categories = load_taxonomy(run)
    assert [c.id for c in categories] == list(TOPICS)
    assert all(c.includes == () for c in categories)


@pytest.mark.parametrize(
    "content",
    [
        "folders: []",
        "folders:\n- {id: A B, name: x}",
        "folders:\n- {id: a, name: x}\n- {id: a, name: y}",
        "[: not yaml",
    ],
)
def test_invalid_taxonomy_is_rejected(content: str, run: Path) -> None:
    (run / "taxonomy.yaml").write_text(content)
    with pytest.raises(ValueError, match="taxonomy"):
        load_taxonomy(run)


@pytest.mark.parametrize(
    ("name", "prefix", "path"),
    [
        ("Mobilität", "", "Mobilität"),
        ("Bank/Finanzen", "INBOX.", "INBOX.Bank-Finanzen"),
        ("Vereine & Co.", "", "Vereine & Co"),
        ("...", "", "fallback"),
    ],
)
def test_folder_path_drops_hierarchy_separators(
    name: str, prefix: str, path: str, run: Path
) -> None:
    assert folder_path(Category("fallback", name, ""), prefix) == path


def test_folder_path_rejects_line_breaks(run: Path) -> None:
    with pytest.raises(ValueError):
        folder_path(Category("a", "A", ""), "INBOX\r\n")
