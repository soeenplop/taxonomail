from pathlib import Path

import pytest
from pydantic import ValidationError

from taxonomail.config import Settings, load_settings


@pytest.mark.parametrize(
    ("toml", "section", "field", "expected"),
    [
        ("[naming]\nmax_tokens = 5\n", "naming", "model", "gpt-6.1-terra"),
        ('[naming]\nmodel = "m"\n', "naming", "max_tokens", 1000),
        ('[consolidation]\nmodel = "m"\n', "consolidation", "max_tokens", 16000),
    ],
)
def test_overriding_one_field_keeps_the_step_defaults(
    tmp_path: Path, toml: str, section: str, field: str, expected: object
) -> None:
    path = tmp_path / "config.toml"
    path.write_text(toml)
    assert getattr(getattr(load_settings(path), section), field) == expected


def test_invalid_values_are_rejected_without_echoing_the_input(tmp_path: Path) -> None:
    secret = "sk-synthetic-secret"
    path = tmp_path / "config.toml"
    path.write_text(f'[embedding]\napi_key = "{secret}"\n')
    with pytest.raises(ValidationError) as error:
        load_settings(path)
    assert secret not in str(error.value)
    with pytest.raises(ValidationError):
        Settings.model_validate({"taxonomy": {"min_folders": 30, "max_folders": 5}})
