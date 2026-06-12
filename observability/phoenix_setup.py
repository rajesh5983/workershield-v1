"""
Phoenix tracing setup for WorkerShield.

Launches the local Phoenix UI server, registers an OTEL tracer provider
pointing at it, and instruments the Anthropic SDK so every LLM call is
recorded automatically.

Safe to import multiple times — initialisation is guarded by a module-level
flag so the server and instrumentor are only set up once per process.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_initialized = False

PHOENIX_UI_URL = "http://192.168.100.10:6006"


def setup_phoenix():
    """
    Start the Phoenix server (if not already running) and configure OTEL tracing.

    Returns the Phoenix Session object, or None if setup fails.
    """
    global _initialized
    if _initialized:
        return None

    try:
        import os

        import phoenix as px
        from openinference.instrumentation.anthropic import AnthropicInstrumentor
        from phoenix.otel import register

        # Use env vars — the host/port params on launch_app are deprecated in Phoenix 17
        os.environ.setdefault("PHOENIX_HOST", "0.0.0.0")
        os.environ.setdefault("PHOENIX_PORT", "6006")

        session = px.launch_app()

        tracer_provider = register(
            project_name="workershield",
            endpoint="http://localhost:6006/v1/traces",
            verbose=False,
        )

        AnthropicInstrumentor().instrument(tracer_provider=tracer_provider)

        _initialized = True
        print(f"Phoenix tracing active — {PHOENIX_UI_URL}")
        logger.info("Phoenix tracing active — %s", PHOENIX_UI_URL)
        return session

    except Exception as exc:
        logger.warning("Phoenix setup failed (tracing disabled): %s", exc)
        return None
