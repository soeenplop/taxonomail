import traceback

import pytest
from pydantic import BaseModel

from taxonomail.config import LLMConfig
from taxonomail.llm import AnthropicLLM, LLMError, OpenAILLM, ask_json


class Answer(BaseModel):
    name: str


class Scripted:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.seen: list[list[dict[str, str]]] = []

    def chat(self, system: str, messages: list[dict[str, str]]) -> str:
        self.seen.append(messages)
        return self.replies.pop(0)


def test_invalid_answer_is_retried_with_the_error() -> None:
    llm = Scripted("not json", '```json\n{"name": "ok"}\n```')
    assert ask_json(llm, "s", "u", Answer).name == "ok"
    retry = llm.seen[1]
    assert [m["role"] for m in retry] == ["user", "assistant", "user"]
    assert "not valid" in retry[2]["content"]


def test_failure_and_traceback_do_not_leak_the_response() -> None:
    llm = Scripted('{"name": 1, "x": "SECRET-MARKER"} SECRET-MARKER', "SECRET-MARKER")
    with pytest.raises(LLMError) as info:
        ask_json(llm, "s", "u", Answer)
    assert len(llm.seen) == 2
    assert "SECRET-MARKER" not in llm.seen[1][2]["content"]
    final = "".join(traceback.format_exception(info.value))
    assert "SECRET-MARKER" not in final


def test_credential_selection() -> None:
    env = {"OPENAI_API_KEY": "sk-openai", "ANTHROPIC_API_KEY": "sk-ant", "MINE": "sk-mine"}

    def keys(cls: type, **cfg: str) -> str:
        made: dict[str, str] = {}
        cls(LLMConfig(model="m", **cfg), lambda **kw: made.update(kw), env)
        return made["api_key"]

    assert keys(OpenAILLM) == "sk-openai"
    assert keys(OpenAILLM, base_url="http://x/v1") == "unused"
    assert keys(OpenAILLM, base_url="http://x/v1", api_key_env="MINE") == "sk-mine"
    assert keys(AnthropicLLM) == "sk-ant"
    with pytest.raises(LLMError, match="NOPE") as info:
        keys(AnthropicLLM, api_key_env="NOPE")
    assert "sk-" not in str(info.value)
