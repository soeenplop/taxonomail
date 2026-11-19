"""LLM backends: Anthropic (Claude) and any OpenAI-compatible server.

Both answer with one JSON object, validated against a pydantic model. An invalid answer is
sent back once with the error, then the call fails.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from taxonomail.config import LLMConfig

ATTEMPTS = 2


class LLMError(Exception):
    """The model gave no valid answer."""


class LLM(Protocol):
    """Sends one chat turn and returns the text of the answer."""

    def chat(self, system: str, messages: list[dict[str, str]]) -> str:
        """Return the model's reply to ``messages``."""
        ...


def ask_json[T: BaseModel](llm: LLM, system: str, user: str, model: type[T]) -> T:
    """Ask for a JSON answer matching ``model``; retry once with the validation error."""
    schema = json.dumps(model.model_json_schema())
    system = (
        f"{system}\n\nAnswer with a single JSON object and nothing else. "
        f"It must match this JSON schema:\n{schema}"
    )
    messages = [{"role": "user", "content": user}]
    problem = ""
    for _ in range(ATTEMPTS):
        reply = llm.chat(system, messages)
        try:
            return model.model_validate_json(_strip_fences(reply))
        except ValidationError as exc:
            problem = str(exc)
        messages = [
            *messages,
            {"role": "assistant", "content": reply},
            {
                "role": "user",
                "content": f"That answer was not valid:\n{problem}\n\nSend the corrected JSON.",
            },
        ]
    raise LLMError(f"no valid answer after {ATTEMPTS} attempts: {problem}")


def _strip_fences(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    return (match.group(1) if match else text).strip()


class AnthropicLLM:
    """Claude through the Anthropic SDK."""

    def __init__(self, config: LLMConfig, client: Any = None) -> None:
        import anthropic

        self._config = config
        self._client: Any = client or anthropic.Anthropic(
            api_key=os.environ.get(config.api_key_env or "ANTHROPIC_API_KEY"),
            base_url=config.base_url,
        )

    def chat(self, system: str, messages: list[dict[str, str]]) -> str:
        """Send one request and return the text blocks of the answer."""
        response = self._client.messages.create(
            model=self._config.model,
            max_tokens=self._config.max_tokens,
            system=system,
            messages=messages,
        )
        return "".join(block.text for block in response.content if block.type == "text")


class OpenAILLM:
    """OpenAI or any OpenAI-compatible server (Ollama, vLLM, ...)."""

    def __init__(self, config: LLMConfig, client: Any = None) -> None:
        import openai

        self._config = config
        self._client: Any = client or openai.OpenAI(
            api_key=os.environ.get(config.api_key_env or "OPENAI_API_KEY") or "unused",
            base_url=config.base_url,
        )

    def chat(self, system: str, messages: list[dict[str, str]]) -> str:
        """Send one request in JSON mode and return the answer's text."""
        # OpenAI itself wants max_completion_tokens; compatible servers still take max_tokens.
        limit = "max_tokens" if self._config.base_url else "max_completion_tokens"
        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=[{"role": "system", "content": system}, *messages],
            response_format={"type": "json_object"},
            **{limit: self._config.max_tokens},
        )
        return response.choices[0].message.content or ""


def build_llm(config: LLMConfig) -> LLM:
    """The backend for one step."""
    if config.provider == "anthropic":
        return AnthropicLLM(config)
    return OpenAILLM(config)
