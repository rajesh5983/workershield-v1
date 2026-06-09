"""
Model abstraction layer for WorkerShield.

Swap between local Ollama models and Anthropic API by changing
model_provider in config/model_config.yaml — no code changes required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "model_config.yaml"


def get_model_config() -> dict[str, Any]:
    """Return the parsed model_config.yaml as a dict."""
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


@dataclass
class LLMClient:
    """
    Unified chat interface for Ollama (local) and Anthropic API backends.

    Usage:
        llm = LLMClient(provider="local", model="gemma4:26b", base_url="http://...", thinking_overhead=400)
        text = llm.chat(system_prompt="You are...", user_message="...", max_tokens=256)
    """

    provider: str
    model: str
    base_url: str | None = None
    thinking_overhead: int = 0  # extra num_predict budget for models that use internal thinking tokens

    def chat(self, system_prompt: str, user_message: str, max_tokens: int = 1500) -> str:
        """Send a message and return the raw text response."""
        if self.provider == "local":
            return self._chat_ollama(system_prompt, user_message, max_tokens)
        if self.provider == "anthropic":
            return self._chat_anthropic(system_prompt, user_message, max_tokens)
        raise ValueError(f"Unknown provider: {self.provider!r}")

    def _chat_ollama(self, system_prompt: str, user_message: str, max_tokens: int) -> str:
        import ollama  # lazy import — not required when provider is anthropic

        num_predict = max_tokens + self.thinking_overhead
        client = ollama.Client(host=self.base_url)
        response = client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            options={"num_predict": num_predict},
        )
        return response.message.content

    def _chat_anthropic(self, system_prompt: str, user_message: str, max_tokens: int) -> str:
        import anthropic  # lazy import — not required when provider is local

        client = anthropic.Anthropic()
        message = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return message.content[0].text


class ModelFactory:
    """
    Build LLMClient instances for each role based on config/model_config.yaml.

    The active provider is set by the top-level ``model_provider`` key.
    """

    def __init__(self) -> None:
        self._config = get_model_config()
        self._provider = self._config["model_provider"]
        logger.debug("ModelFactory initialised: provider=%s", self._provider)

    def _local_client(self, model: str) -> LLMClient:
        cfg = self._config["local"]
        return LLMClient(
            provider="local",
            model=model,
            base_url=cfg["base_url"],
            thinking_overhead=cfg.get("thinking_overhead", 0),
        )

    def get_router_llm(self) -> LLMClient:
        """Return the LLM client configured for the router/classifier role."""
        if self._provider == "local":
            return self._local_client(self._config["local"]["router"])
        return LLMClient(provider="anthropic", model=self._config["anthropic"]["router"])

    def get_synthesis_llm(self) -> LLMClient:
        """Return the LLM client configured for the synthesis role."""
        if self._provider == "local":
            return self._local_client(self._config["local"]["synthesis"])
        return LLMClient(provider="anthropic", model=self._config["anthropic"]["synthesis"])

    def get_comparison_llm(self, model_name: str) -> LLMClient:
        """
        Return an LLMClient for any model listed in comparison_models.

        Used by the Gradio comparison panel to run side-by-side evaluations.
        """
        for entry in self._config.get("comparison_models", []):
            if entry["name"] == model_name:
                if entry["provider"] == "local":
                    return self._local_client(model_name)
                return LLMClient(provider="anthropic", model=model_name)
        raise ValueError(f"Model {model_name!r} not found in comparison_models")

    @property
    def active_provider(self) -> str:
        return self._provider

    @property
    def router_model(self) -> str:
        cfg_key = "local" if self._provider == "local" else "anthropic"
        return self._config[cfg_key]["router"]

    @property
    def synthesis_model(self) -> str:
        cfg_key = "local" if self._provider == "local" else "anthropic"
        return self._config[cfg_key]["synthesis"]
