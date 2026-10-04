"""Provider-agnostic LLM interface."""

import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class LLMError(RuntimeError):
    pass


class LLMNotConfigured(LLMError):
    pass


@dataclass(frozen=True)
class LLMResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int


class LLMProvider(ABC):
    name: str
    model: str
    fast_model: str

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 512,
        json_mode: bool = False,
        fast: bool = False,
    ) -> LLMResult:
        """Single non-streaming completion. ``fast`` selects the cheaper model."""

    def stream(
        self,
        messages: list[dict[str, str]],
        on_token: Callable[[str], None],
        *,
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 1024,
    ) -> LLMResult:
        """Streaming completion: ``on_token`` gets each text delta; returns the full result.

        Default implementation emits the whole completion as one delta (providers override).
        """
        result = self.complete(messages, temperature=temperature, seed=seed, max_tokens=max_tokens)
        if result.text:
            on_token(result.text)
        return result

    def complete_json(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        """Completion parsed as a JSON object; raises ``LLMError`` on malformed output."""
        result = self.complete(messages, json_mode=True, **kwargs)
        text = result.text.strip()
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"LLM returned invalid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise LLMError("LLM returned JSON that is not an object")
        return parsed
