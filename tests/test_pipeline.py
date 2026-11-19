from taxonomail.config import Settings
from taxonomail.mail import Mail, parse_mail
from taxonomail.pipeline import Result, build_taxonomy
from tests.fakes import TOPICS, FakeClusterer, FakeEmbedder, FakeLLM, Plan, raw_mail


def mails() -> list[Mail]:
    subjects = [f"{TOPICS[i % 3]} number {i}" for i in range(12)] + ["random thing"]
    return [parse_mail(raw_mail(s, n=i)) for i, s in enumerate(subjects)]


def run(llm: FakeLLM) -> Result:
    return build_taxonomy(
        mails(),
        embedder=FakeEmbedder(),
        clusterer=FakeClusterer(),
        naming=llm,
        consolidation=llm,
        settings=Settings(),
        log=lambda _: None,
    )


def test_builds_folders_and_assigns_every_mail() -> None:
    llm = FakeLLM()
    result = run(llm)
    assert [(f.id, f.size) for f in result.folders] == [("invoices", 5), ("travel", 4), ("club", 4)]
    assert len(result.assignments) == 13
    noise = [a for a in result.assignments if a.method == "nearest"]
    assert [a.mail.subject for a in noise] == ["random thing"]
    assert result.cluster_names[1].name == "travel mails"
    assert result.cluster_folders == {0: "invoices", 1: "travel", 2: "club"}
    assert sum("Representative mails" in m[0]["content"] for _, m in llm.calls) == 3


def test_bad_plan_is_retried_then_repaired() -> None:
    def merged_without_club(ids: list[int]) -> dict[str, object]:
        folders = [
            {"id": "money-travel", "name": "M", "description": "d", "clusters": [0, 1, 0, 9]}
        ]
        return {"folders": folders}

    plans: list[Plan] = [merged_without_club, merged_without_club]
    llm = FakeLLM(plans)
    result = run(llm)
    consolidations = [m for _, m in llm.calls if "exactly one folder" in m[0]["content"]]
    assert len(consolidations) == 2
    assert "cluster 2 is in no folder" in consolidations[1][0]["content"]
    assert "cluster 9 does not exist" in consolidations[1][0]["content"]
    # The cluster left out goes to the only folder there is.
    assert result.cluster_folders == {0: "money-travel", 1: "money-travel", 2: "money-travel"}
    assert [(f.id, f.size) for f in result.folders] == [("money-travel", 13)]
