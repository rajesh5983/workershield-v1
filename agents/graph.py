"""
WorkerShield LangGraph state machine.

Flow:
  START → router_node → (conditional) → domain node(s) → synthesis_node → output_node → END

Conditional edge after router_node:
  cross_domain=True  → all detected domain nodes run in parallel (via Send)
  cross_domain=False → only the single detected domain node runs
"""

from __future__ import annotations

import logging
import operator
import os
from typing import Annotated, Any

import requests
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from typing_extensions import TypedDict

from agents.router import router_node as _router_node
from agents.synthesis import synthesis_node as _synthesis_node

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://192.168.100.1:11434")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "http://localhost:6333")
COLLECTION  = "workershield"
EMBED_MODEL = "nomic-embed-text"
MAX_CHARS   = 6_000
TOP_K       = 3

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class WorkerShieldState(TypedDict):
    query:             str
    detected_domains:  list[str]
    cross_domain:      bool
    # Annotated with operator.add so parallel nodes can each append without
    # overwriting each other's results
    safeshift_chunks:  Annotated[list[dict], operator.add]
    fairdesk_chunks:   Annotated[list[dict], operator.add]
    healthnav_chunks:  Annotated[list[dict], operator.add]
    synthesis_input:   str
    final_answer:      str
    citations:         list[dict]  # {doc_id, doc_title, section, domain, excerpt}
    confidence:        str         # "high", "medium", or "low"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _embed(text: str) -> list[float]:
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]
    r = requests.post(
        f"{OLLAMA_HOST}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["embedding"]


def _retrieve(query: str, domain: str, client: QdrantClient) -> list[dict]:
    """Embed query, query Qdrant filtered to domain, return top-K payloads."""
    vec  = _embed(query)
    hits = client.query_points(
        collection_name=COLLECTION,
        query=vec,
        query_filter=Filter(
            must=[FieldCondition(key="domain", match=MatchValue(value=domain))]
        ),
        limit=TOP_K,
        with_payload=True,
    ).points

    chunks = []
    for h in hits:
        p = h.payload
        chunks.append({
            "doc_id":   p.get("doc_id", ""),
            "domain":   p.get("domain", ""),
            "title":    p.get("title", ""),
            "source":   p.get("source", ""),
            "section":  p.get("section", ""),
            "page":     p.get("page_estimate", 1),
            "score":    round(h.score, 4),
            "text":     p.get("text", ""),
        })

    doc_ids = [c["doc_id"] for c in chunks]
    sections = [c["section"][:40] for c in chunks]
    logger.info("[%s] retrieved %d chunks: %s", domain, len(chunks),
                list(zip(doc_ids, sections)))
    return chunks


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def router_node(state: WorkerShieldState) -> dict[str, Any]:
    logger.info("[router] query=%r", state["query"])
    result = _router_node(state)
    logger.info("[router] → domains=%s cross_domain=%s",
                result["detected_domains"], result["cross_domain"])
    return result


def safeshift_node(state: WorkerShieldState) -> dict[str, Any]:
    client = QdrantClient(url=QDRANT_HOST)
    chunks = _retrieve(state["query"], "safeshift", client)
    logger.info("[safeshift] %d chunks retrieved", len(chunks))
    return {"safeshift_chunks": chunks}


def fairdesk_node(state: WorkerShieldState) -> dict[str, Any]:
    client = QdrantClient(url=QDRANT_HOST)
    chunks = _retrieve(state["query"], "fairdesk", client)
    logger.info("[fairdesk] %d chunks retrieved", len(chunks))
    return {"fairdesk_chunks": chunks}


def healthnav_node(state: WorkerShieldState) -> dict[str, Any]:
    client = QdrantClient(url=QDRANT_HOST)
    chunks = _retrieve(state["query"], "healthnav", client)
    logger.info("[healthnav] %d chunks retrieved", len(chunks))
    return {"healthnav_chunks": chunks}


_DOMAIN_NODES = {
    "safeshift": "safeshift_node",
    "fairdesk":  "fairdesk_node",
    "healthnav": "healthnav_node",
}


def synthesis_node(state: WorkerShieldState) -> dict[str, Any]:
    """Delegate to agents.synthesis which owns the full synthesis logic."""
    return _synthesis_node(state)


def output_node(state: WorkerShieldState) -> dict[str, Any]:
    """Format the final response — no-op in state terms; side-effect is the log."""
    logger.info(
        "[output] answer_len=%d citations=%d confidence=%s",
        len(state.get("final_answer", "")),
        len(state.get("citations", [])),
        state.get("confidence", ""),
    )
    return {}


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------

def _route_domains(state: WorkerShieldState):
    """
    Return a list of Send objects — one per detected domain node.
    LangGraph fans these out in parallel when there are multiple.
    """
    return [
        Send(_DOMAIN_NODES[d], state)
        for d in state["detected_domains"]
        if d in _DOMAIN_NODES
    ]


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    g = StateGraph(WorkerShieldState)

    g.add_node("router_node",    router_node)
    g.add_node("safeshift_node", safeshift_node)
    g.add_node("fairdesk_node",  fairdesk_node)
    g.add_node("healthnav_node", healthnav_node)
    g.add_node("synthesis_node", synthesis_node)
    g.add_node("output_node",    output_node)

    g.add_edge(START, "router_node")

    # Conditional fan-out from router → domain node(s)
    g.add_conditional_edges("router_node", _route_domains)

    # All domain nodes converge on synthesis
    for node in _DOMAIN_NODES.values():
        g.add_edge(node, "synthesis_node")

    g.add_edge("synthesis_node", "output_node")
    g.add_edge("output_node", END)

    return g


# Compiled graph — importable by the UI and other modules
graph = build_graph().compile()


# ---------------------------------------------------------------------------
# End-to-end test harness
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s  %(name)s  %(message)s",
    )

    TEST_QUERY = (
        "What are my obligations if a FIFO worker has a mental health condition "
        "and wants to reduce hours?"
    )

    print("\n" + "=" * 72)
    print(f"TEST QUERY: {TEST_QUERY}")
    print("=" * 72)

    initial_state: WorkerShieldState = {
        "query":            TEST_QUERY,
        "detected_domains": [],
        "cross_domain":     False,
        "safeshift_chunks": [],
        "fairdesk_chunks":  [],
        "healthnav_chunks": [],
        "synthesis_input":  "",
        "final_answer":     "",
        "citations":        [],
        "confidence":       "",
    }

    # Stream node-by-node so we can print state after each step
    seen_nodes: set[str] = set()
    # Accumulate state across all node updates
    accumulated: dict = {k: v for k, v in initial_state.items()}

    for step in graph.stream(initial_state, stream_mode="updates"):
        for node_name, node_output in step.items():
            if node_output is None:
                continue
            # Merge into accumulated state (lists use operator.add semantics)
            for k, v in node_output.items():
                if isinstance(v, list) and isinstance(accumulated.get(k), list):
                    accumulated[k] = accumulated[k] + v
                else:
                    accumulated[k] = v

            if node_name in seen_nodes:
                continue
            seen_nodes.add(node_name)

            print(f"\n{'─'*72}")
            print(f"NODE: {node_name}")
            print(f"{'─'*72}")

            if node_name == "router_node":
                print(f"  detected_domains : {node_output.get('detected_domains')}")
                print(f"  cross_domain     : {node_output.get('cross_domain')}")

            elif node_name in ("safeshift_node", "fairdesk_node", "healthnav_node"):
                key    = node_name.replace("_node", "_chunks")
                chunks = node_output.get(key, [])
                print(f"  chunks retrieved : {len(chunks)}")
                for c in chunks:
                    print(f"    [{c['doc_id']}] score={c['score']}  "
                          f"section={repr(c['section'][:40])}  "
                          f"text={repr(c['text'][:80])}")

            elif node_name == "synthesis_node":
                answer = node_output.get("final_answer", "")
                cits   = node_output.get("citations", [])
                print(f"  answer length    : {len(answer)} chars")
                print(f"  citations        : {len(cits)}")
                print(f"\n--- FINAL ANSWER ---\n")
                print(answer)
                print(f"\n--- CITATIONS ---")
                for cit in cits:
                    sec = f" § {cit['section'][:40]}" if cit["section"] else ""
                    print(f"  [{cit['doc_id']}] {cit['doc_title']}{sec}")

            elif node_name == "output_node":
                print("  (formatting complete — state ready for UI)")

    print(f"\n{'='*72}")
    print(f"DOMAINS HIT     : {accumulated.get('detected_domains', [])}")
    print(f"CROSS DOMAIN    : {accumulated.get('cross_domain', False)}")
    print(f"SAFESHIFT chunks: {len(accumulated.get('safeshift_chunks', []))}")
    print(f"FAIRDESK  chunks: {len(accumulated.get('fairdesk_chunks', []))}")
    print(f"HEALTHNAV chunks: {len(accumulated.get('healthnav_chunks', []))}")
    print("=" * 72 + "\n")
