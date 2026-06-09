"""
Model abstraction layer for WorkerShield.

Swap router and synthesis models at runtime by passing a model_id to
get_router_llm() / get_synthesis_llm(), or let them fall back to the
defaults block in config/model_config.yaml.

Supported providers: ollama, openai
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "model_config.yaml"


def get_model_config() -> dict[str, Any]:
    """Return the parsed model_config.yaml as a dict."""
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


@dataclass
class LLMClient:
    """
    Unified chat interface for Ollama (local) and OpenAI API backends.

    provider: "ollama" | "openai"
    model:    ollama model tag or OpenAI model name
    base_url: Ollama host (ignored for openai)
    thinking_overhead: extra num_predict tokens for Ollama thinking models
    max_tokens: token budget for the visible response
    """

    provider: str
    model: str
    base_url: str | None = None
    thinking_overhead: int = 0
    max_tokens: int = 1500

    def chat(self, system_prompt: str, user_message: str) -> str:
        """Send a message and return the raw text response."""
        if self.provider == "ollama":
            return self._chat_ollama(system_prompt, user_message)
        if self.provider == "openai":
            return self._chat_openai(system_prompt, user_message)
        raise ValueError(f"Unknown provider: {self.provider!r}")

    def _chat_ollama(self, system_prompt: str, user_message: str) -> str:
        import ollama  # lazy import

        num_predict = self.max_tokens + self.thinking_overhead
        client = ollama.Client(host=self.base_url)
        response = client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            options={"num_predict": num_predict},
        )
        return response.message.content

    def _chat_openai(self, system_prompt: str, user_message: str) -> str:
        from openai import OpenAI  # lazy import

        client = OpenAI()  # reads OPENAI_API_KEY from env
        response = client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
        )
        return response.choices[0].message.content


class ModelFactory:
    """
    Build LLMClient instances by looking up a model_id in the registry.

    Registry is loaded from config/model_config.yaml.
    If no model_id is supplied the defaults block is used.
    """

    def __init__(self) -> None:
        cfg = get_model_config()
        self._ollama_base_url: str = cfg["ollama"]["base_url"]
        self._defaults: dict[str, str] = cfg["defaults"]

        # Build flat registries keyed by model_id
        self._router_registry: dict[str, dict] = {
            e["model_id"]: e for e in cfg.get("router_models", [])
        }
        self._synthesis_registry: dict[str, dict] = {
            e["model_id"]: e for e in cfg.get("synthesis_models", [])
        }

    # ── public API ────────────────────────────────────────────────────────────

    def get_router_llm(self, model_id: str | None = None) -> LLMClient:
        """Return an LLMClient for the router role."""
        mid = model_id or self._defaults["router"]
        return self._build(mid, self._router_registry)

    def get_synthesis_llm(self, model_id: str | None = None) -> LLMClient:
        """Return an LLMClient for the synthesis role."""
        mid = model_id or self._defaults["synthesis"]
        return self._build(mid, self._synthesis_registry)

    def router_model_ids(self) -> list[str]:
        """Ordered list of available router model_ids (for UI dropdowns)."""
        return list(self._router_registry)

    def synthesis_model_ids(self) -> list[str]:
        """Ordered list of available synthesis model_ids (for UI dropdowns)."""
        return list(self._synthesis_registry)

    def default_router_model(self) -> str:
        return self._defaults["router"]

    def default_synthesis_model(self) -> str:
        return self._defaults["synthesis"]

    # ── internal ──────────────────────────────────────────────────────────────

    def _build(self, model_id: str, registry: dict[str, dict]) -> LLMClient:
        entry = registry.get(model_id)
        if entry is None:
            available = list(registry)
            raise ValueError(
                f"model_id {model_id!r} not found in registry. "
                f"Available: {available}"
            )
        provider = entry["provider"]
        return LLMClient(
            provider=provider,
            model=entry["model_name"],
            base_url=self._ollama_base_url if provider == "ollama" else None,
            thinking_overhead=entry.get("thinking_overhead", 0),
            max_tokens=entry.get("max_tokens", 1500),
        )
