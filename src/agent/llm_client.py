from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class LLMSettings:
    """Configuration loaded from environment variables or passed directly."""

    api_key: str
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    temperature: float = 0.2
    timeout_sec: float = 60.0
    max_retries: int = 3

    @classmethod
    def from_env(cls, model_env: str = "AGENT_MODEL", key_env: str = "OPENAI_API_KEY") -> LLMSettings:
        api_key = os.getenv(key_env, "").strip()
        if not api_key:
            raise ValueError(f"Missing required API key in environment variable '{key_env}'.")
        base_url = os.getenv("AGENT_LLM_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
        model = os.getenv(model_env, "gpt-4o-mini").strip()
        temp_raw = os.getenv("AGENT_LLM_TEMPERATURE", "0.2").strip()
        timeout_raw = os.getenv("AGENT_LLM_TIMEOUT_SEC", "60.0").strip()
        retries_raw = os.getenv("AGENT_LLM_MAX_RETRIES", "3").strip()

        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=float(temp_raw or "0.2"),
            timeout_sec=float(timeout_raw or "60.0"),
            max_retries=int(retries_raw or "3"),
        )


class OpenAICompatibleAsyncClient:
    """Async JSON-focused LLM client with exponential backoff on transient errors."""

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings

    async def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{self.settings.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        last_error: Exception | None = None
        async with httpx.AsyncClient(timeout=self.settings.timeout_sec) as client:
            for attempt in range(1, self.settings.max_retries + 2):
                try:
                    resp = await client.post(url, headers=headers, json=payload)
                    if resp.status_code in {429, 500, 502, 503, 504}:
                        raise httpx.HTTPStatusError(
                            f"Transient HTTP {resp.status_code}: {resp.text[:300]}",
                            request=resp.request,
                            response=resp,
                        )
                    resp.raise_for_status()
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    if not isinstance(content, str) or not content.strip():
                        raise ValueError("Model returned empty message content.")
                    # Validate JSON round-trip early so BrainEngine/CriticEngine get clean JSON
                    json.loads(content)
                    return content
                except (httpx.RequestError, httpx.HTTPStatusError, ValueError, KeyError, IndexError) as exc:
                    last_error = exc
                    if attempt > self.settings.max_retries:
                        break
                    backoff = min(12.0, (2 ** (attempt - 1)) * 0.75)
                    await asyncio.sleep(backoff)

        raise RuntimeError(f"LLM request failed after retries: {last_error}") from last_error