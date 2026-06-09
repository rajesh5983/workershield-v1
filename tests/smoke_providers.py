"""
Smoke test — run the killer demo query against all three model stacks and
print a comparison table.

Usage:
    cd /projects/workershield-v1
    python tests/smoke_providers.py

OpenAI is skipped automatically if OPENAI_API_KEY is not set or is a placeholder.
"""

from __future__ import annotations

import os
import sys
import time

# Ensure project root is on the path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv("/projects/workershield-v1/.env")

from utils.model_factory import set_model_provider, get_model_config
from agents.graph import graph  # compiles the graph at import time

SMOKE_QUERY = "What are psychosocial hazards under WHS law?"

PROVIDERS = ["anthropic", "local", "openai"]


def _openai_available() -> bool:
    key = os.environ.get("OPENAI_API_KEY", "")
    return bool(key) and key.startswith("sk-") and key != "sk-your-openai-key-here"


def _run_provider(provider: str) -> dict:
    """Run the smoke query for a single provider. Returns a result dict."""
    set_model_provider(provider)
    cfg = get_model_config()
    router_model    = cfg[provider]["router"]
    synthesis_model = cfg[provider]["synthesis"]

    start = time.time()
    try:
        result  = graph.invoke({"query": SMOKE_QUERY})
        elapsed = round(time.time() - start, 1)

        import json
        raw_final = result.get("final_answer", "")
        try:
            parsed    = json.loads(raw_final)
            answer    = parsed.get("answer", raw_final)
        except (json.JSONDecodeError, TypeError):
            answer = raw_final

        domains    = result.get("detected_domains", [])
        confidence = result.get("confidence", "—")
        preview    = (answer or "").replace("\n", " ")[:50]

        return {
            "provider":   provider,
            "router":     router_model,
            "synthesis":  synthesis_model,
            "domains":    ", ".join(domains) or "—",
            "confidence": confidence or "—",
            "preview":    preview,
            "latency":    f"{elapsed}s",
            "status":     "pass",
        }
    except Exception as exc:
        elapsed = round(time.time() - start, 1)
        return {
            "provider":   provider,
            "router":     router_model,
            "synthesis":  synthesis_model,
            "domains":    "ERROR",
            "confidence": "—",
            "preview":    str(exc)[:50],
            "latency":    f"{elapsed}s",
            "status":     "fail",
        }


def main() -> None:
    print(f"\n{'='*72}")
    print(f"WorkerShield — Provider Smoke Test")
    print(f"Query: {SMOKE_QUERY!r}")
    print(f"{'='*72}\n")

    results = []

    for provider in PROVIDERS:
        if provider == "openai" and not _openai_available():
            print(f"[{provider.upper():12}]  SKIP — OPENAI_API_KEY not set")
            results.append({
                "provider":   provider,
                "router":     "gpt-4o-mini",
                "synthesis":  "gpt-4o",
                "domains":    "SKIP",
                "confidence": "—",
                "preview":    "OPENAI_API_KEY not configured",
                "latency":    "—",
                "status":     "skip",
            })
            continue

        print(f"[{provider.upper():12}]  Running… ", end="", flush=True)
        r = _run_provider(provider)
        print(f"{r['status'].upper()}  ({r['latency']})")
        print(f"               domains    : {r['domains']}")
        print(f"               confidence : {r['confidence']}")
        print(f"               answer     : {r['preview']!r}")
        results.append(r)

    # Restore default
    set_model_provider("anthropic")

    # Comparison table
    print(f"\n{'='*72}")
    print(f"{'Provider':<12} {'Router Result':<25} {'Confidence':<12} {'Answer Quality':<16} {'Latency'}")
    print(f"{'─'*72}")
    for r in results:
        quality = "pass" if r["status"] == "pass" and r["domains"] not in ("—", "ERROR", "SKIP") else r["status"]
        print(
            f"{r['provider']:<12} "
            f"{r['domains']:<25} "
            f"{r['confidence']:<12} "
            f"{quality:<16} "
            f"{r['latency']}"
        )
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
