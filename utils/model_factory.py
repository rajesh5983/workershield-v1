"""
Model abstraction layer for WorkerShield.

Active provider (anthropic | openai | local) is read from model_provider
in config/model_config.yaml.  Switch stacks by calling set_model_provider()
or editing the file; ModelFactory re-reads config on every instantiation.

Supported providers: anthropic, openai, local (Ollama)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "model_config.yaml"

# Ollama models that emit a thinking preamble before the actual response.
# The extra tokens are added to num_predict so the answer isn't truncated.
_THINKING_OVERHEAD: dict[str, int] = {
    "gemma4:26b": 400,
}


# ---------------------------------------------------------------------------
# Config I/O
# ---------------------------------------------------------------------------

def get_model_config() -> dict[str, Any]:
    """Return the parsed model_config.yaml as a dict."""
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def set_model_provider(provider: str) -> None:
    """Overwrite the model_provider line in config/model_config.yaml in-place."""
    text = _CONFIG_PATH.read_text()
    text = re.sub(
        r'^model_provider:\s*\S+.*$',
        f'model_provider: {provider}',
        text,
        flags=re.MULTILINE,
    )
    _CONFIG_PATH.write_text(text)


# ---------------------------------------------------------------------------
# Shared JSON parser
# ---------------------------------------------------------------------------

def parse_llm_json(text: str) -> dict | None:
    """Parse JSON from LLM output, tolerating markdown fences and minor formatting errors.

    Strategy: try a clean parse first (preserves apostrophes in string values),
    then fall back to single-quote replacement for local models that emit
    Python-dict-style output.
    """
    if not text:
        return None

    # Strip markdown fences
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = re.sub(r'```\s*$', '', text).strip()

    # Find outermost { }
    start = text.find('{')
    end   = text.rfind('}')
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = text[start:end + 1]

    # Pass 1 — clean parse (handles Anthropic/OpenAI proper JSON)
    clean = re.sub(r',\s*}', '}', candidate)
    clean = re.sub(r',\s*]', ']', clean)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # Pass 2 — single-quote + Python literal fixes (for local models)
    clean2 = candidate.replace("'", '"')
    clean2 = re.sub(r'\bTrue\b',  'true',  clean2)
    clean2 = re.sub(r'\bFalse\b', 'false', clean2)
    clean2 = re.sub(r'\bNone\b',  'null',  clean2)
    clean2 = re.sub(r',\s*}', '}', clean2)
    clean2 = re.sub(r',\s*]', ']', clean2)
    try:
        return json.loads(clean2)
    except json.JSONDecodeError:
        pass

    return None


# ---------------------------------------------------------------------------
# Unified LLM client
# ---------------------------------------------------------------------------

@dataclass
class LLMClient:
    """
    Unified chat interface for Ollama, OpenAI, and Anthropic backends.

    provider: "ollama" | "openai" | "anthropic"
    model:    model tag / name / ID for the chosen provider
    base_url: Ollama host URL (ignored for openai / anthropic)
    max_tokens: visible response token budget
    thinking_overhead: extra num_predict tokens for Ollama thinking models
    """

    provider:          str
    model:             str
    base_url:          str | None = None
    max_tokens:        int = 1500
    thinking_overhead: int = 0

    def chat(self, system_prompt: str, user_message: str) -> str:
        """Dispatch to the correct backend and return the raw text response."""
        if self.provider == "ollama":
            return self._chat_ollama(system_prompt, user_message)
        if self.provider == "openai":
            return self._chat_openai(system_prompt, user_message)
        if self.provider == "anthropic":
            return self._chat_anthropic(system_prompt, user_message)
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
            options={
                "num_predict": num_predict,
                "temperature": 0.1,
                # Prevent thinking-token preamble from consuming the visible budget
                "stop": ["</think>", "<|end|>"],
            },
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

    def _chat_anthropic(self, system_prompt: str, user_message: str) -> str:
        import anthropic  # lazy import

        client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        message = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return message.content[0].text


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class ModelFactory:
    """
    Build LLMClient instances from config/model_config.yaml.

    The active provider is the model_provider key in the config.
    Call set_model_provider() to switch stacks; ModelFactory re-reads the
    config on each instantiation so the change takes effect immediately.
    """

    def __init__(self) -> None:
        cfg = get_model_config()
        self._provider: str = cfg.get("model_provider", "anthropic")
        self._cfg = cfg

    # ── public API ────────────────────────────────────────────────────────────

    def get_router_llm(self) -> LLMClient:
        """Return an LLMClient configured for the router role."""
        return self._build("router")

    def get_synthesis_llm(self) -> LLMClient:
        """Return an LLMClient configured for the synthesis role."""
        return self._build("synthesis")

    def active_provider(self) -> str:
        return self._provider

    def router_model_name(self) -> str:
        return self._cfg[self._provider]["router"]

    def synthesis_model_name(self) -> str:
        return self._cfg[self._provider]["synthesis"]

    # ── internal ──────────────────────────────────────────────────────────────

    def _build(self, role: str) -> LLMClient:
        provider     = self._provider
        provider_cfg = self._cfg.get(provider, {})
        model        = provider_cfg.get(role)
        if not model:
            raise ValueError(
                f"No {role!r} model configured for provider {provider!r}"
            )

        # "local" maps to Ollama internally
        llm_provider = "ollama" if provider == "local" else provider
        base_url     = provider_cfg.get("base_url") if provider == "local" else None
        max_tokens   = 256 if role == "router" else 1500
        overhead     = _THINKING_OVERHEAD.get(model, 0) if llm_provider == "ollama" else 0

        return LLMClient(
            provider=llm_provider,
            model=model,
            base_url=base_url,
            max_tokens=max_tokens,
            thinking_overhead=overhead,
        )
