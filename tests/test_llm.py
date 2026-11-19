from pydantic import BaseModel

from taxonomail.llm import ask_json


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
