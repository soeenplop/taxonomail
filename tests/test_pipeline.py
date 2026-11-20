import pytest

from taxonomail.config import Settings, TaxonomyConfig
from taxonomail.mail import Mail, parse_mail
from taxonomail.pipeline import Result, build_taxonomy
from tests.fakes import (
    TOPICS,
    FakeClusterer,
    FakeEmbedder,
    FakeLLM,
    Plan,
    one_folder_per_cluster,
    raw_mail,
)


def mails() -> list[Mail]:
    subjects = [f"{TOPICS[i % 3]} number {i}" for i in range(12)] + ["random thing"]
    return [parse_mail(raw_mail(s, n=i)) for i, s in enumerate(subjects)]


def run(llm: FakeLLM, settings: Settings | None = None) -> Result:
    return build_taxonomy(
        mails(),
        embedder=FakeEmbedder(),
        clusterer=FakeClusterer(),
        naming=llm,
        consolidation=llm,
        settings=settings or Settings(),
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
    result = run(llm, Settings(taxonomy=TaxonomyConfig(min_folders=1)))
    consolidations = [m for _, m in llm.calls if "exactly one folder" in m[0]["content"]]
    assert len(consolidations) == 2
    assert "cluster 2 is in no folder" in consolidations[1][0]["content"]
    assert "cluster 9 does not exist" in consolidations[1][0]["content"]
    # The cluster left out goes to the only folder there is.
    assert result.cluster_folders == {0: "money-travel", 1: "money-travel", 2: "money-travel"}
    assert [(f.id, f.size) for f in result.folders] == [("money-travel", 13)]


def test_folder_count_outside_bounds_is_retried_then_rejected() -> None:
    llm = FakeLLM()
    with pytest.raises(ValueError, match="3 folders") as info:
        run(llm, Settings(taxonomy=TaxonomyConfig(min_folders=1, max_folders=2)))
    consolidations = [m for _, m in llm.calls if "exactly one folder" in m[0]["content"]]
    assert len(consolidations) == 2
    assert "must have 1 to 2" in consolidations[1][0]["content"]
    assert "invoices" not in str(info.value)


def test_folder_emptied_by_repair_does_not_count() -> None:
    def overlapping(ids: list[int]) -> dict[str, object]:
        folders = [
            {"id": "a", "name": "A", "description": "d", "clusters": [0, 1, 2]},
            {"id": "b", "name": "B", "description": "d", "clusters": [1]},
        ]
        return {"folders": folders}

    llm = FakeLLM([overlapping, overlapping])
    with pytest.raises(ValueError, match="1 folders"):
        run(llm, Settings(taxonomy=TaxonomyConfig(min_folders=2)))


def test_language_goes_into_both_prompts() -> None:
    llm = FakeLLM()
    run(llm)
    assert all("words in English" in system for system, _ in llm.calls)
    assert all("Write English as people speak it every day" in system for system, _ in llm.calls)
    assert not any("in German" in system for system, _ in llm.calls)
    llm = FakeLLM()
    run(llm, Settings(taxonomy=TaxonomyConfig(language="German")))
    assert len(llm.calls) == 4
    assert all("in German" in system and "English" not in system for system, _ in llm.calls)
    assert all("Write German as people speak it every day" in system for system, _ in llm.calls)
    consolidation = [system for system, m in llm.calls if "exactly one folder" in m[0]["content"]]
    assert "never one specific\n   company, brand" in consolidation[0]
    assert "even when one sender dominates" in consolidation[0]
    loanwords = 'German\nspeakers say "Shopping" rather than "Einkauf", but "Rechnungen"'
    assert all(loanwords in system for system, _ in llm.calls)


def test_folder_ids_become_ascii_slugs() -> None:
    def french(ids: list[int]) -> dict[str, object]:
        folders = [
            {"id": "Électricité & Gaz", "name": "Électricité", "description": "d", "clusters": [0]},
            {"id": "voyages", "name": "Voyages", "description": "d", "clusters": [1, 2]},
        ]
        return {"folders": folders}

    result = run(FakeLLM([french]), Settings(taxonomy=TaxonomyConfig(min_folders=2)))
    assert [f.id for f in result.folders] == ["electricite-gaz", "voyages"]
    assert result.folders[0].name == "Électricité"
    assert {a.folder for a in result.assignments} == {"electricite-gaz", "voyages"}


def test_folder_id_without_ascii_letters_is_asked_again() -> None:
    def japanese(ids: list[int]) -> dict[str, object]:
        return {"folders": [{"id": "旅行", "name": "旅行", "description": "d", "clusters": ids}]}

    llm = FakeLLM([japanese, one_folder_per_cluster])
    result = run(llm)
    assert "folders: value_error" in llm.calls[-1][1][-1]["content"]
    assert [f.id for f in result.folders] == ["invoices", "travel", "club"]


def test_ids_that_collide_after_the_retry_get_a_suffix() -> None:
    def colliding(ids: list[int]) -> dict[str, object]:
        folders = [
            {"id": "café", "name": "Café", "description": "d", "clusters": [0]},
            {"id": "cafe", "name": "Cafe", "description": "d", "clusters": [1]},
            {"id": "Café!", "name": "Café!", "description": "d", "clusters": [2]},
        ]
        return {"folders": folders}

    llm = FakeLLM([colliding, colliding])
    result = run(llm, Settings(taxonomy=TaxonomyConfig(min_folders=3)))
    consolidations = [m for _, m in llm.calls if "exactly one folder" in m[0]["content"]]
    assert "folders 1 and 2 use the same id" in consolidations[1][0]["content"]
    assert [(f.id, f.name, f.size) for f in result.folders] == [
        ("cafe", "Café", 5),
        ("cafe-2", "Cafe", 4),
        ("cafe-3", "Café!", 4),
    ]
    assert result.cluster_folders == {0: "cafe", 1: "cafe-2", 2: "cafe-3"}
