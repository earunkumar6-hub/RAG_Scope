"""OpenAI chat-completions provider."""

from collections.abc import Callable

import httpx

from app.llm.base import LLMError, LLMProvider, LLMResult


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str,
        fast_model: str,
        base_url: str | None = None,
        http_client: httpx.Client | None = None,
        timeout: float = 60.0,
    ):
        from openai import OpenAI

        self.model = model
        self.fast_model = fast_model
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
            timeout=timeout,
            max_retries=2,
        )

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
        from openai import OpenAIError

        model = self.fast_model if fast else self.model
        kwargs: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
        }
        if seed is not None:
            kwargs["seed"] = seed
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except OpenAIError as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc
        usage = resp.usage
        return LLMResult(
            text=resp.choices[0].message.content or "",
            model=resp.model or model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )

    def stream(
        self,
        messages: list[dict[str, str]],
        on_token: Callable[[str], None],
        *,
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 1024,
    ) -> LLMResult:
        from openai import OpenAIError

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if seed is not None:
            kwargs["seed"] = seed
        parts: list[str] = []
        model, prompt_tokens, completion_tokens = self.model, 0, 0
        try:
            for chunk in self._client.chat.completions.create(**kwargs):
                model = chunk.model or model
                if chunk.usage:  # final chunk (include_usage) has no choices
                    prompt_tokens = chunk.usage.prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens
                if chunk.choices and chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    parts.append(delta)
                    on_token(delta)
        except OpenAIError as exc:
            raise LLMError(f"OpenAI request failed: {exc}") from exc
        return LLMResult("".join(parts), model, prompt_tokens, completion_tokens)
