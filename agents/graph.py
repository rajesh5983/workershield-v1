"""
WorkerShield LangGraph state machine.

Flow:
  START → router_node → (conditional) → domain node(s) → synthesis_node → output_node → END

Conditional edge after router_node:
  cross_domain=True  → all detected domain nodes run in parallel (via Send)
  cross_domain=False → only the single detected domain node runs
"""

from __future__ import annotations

import asyncio
import json
import logging
import operator
import os
import time
from pathlib import Path
from typing import Annotated, Any

import requests
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from opentelemetry import trace as _otel_trace
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition, Filter, Fusion, MatchValue, Prefetch, SparseVector,
)
from typing_extensions import TypedDict

from agents.router import router_node as _router_node
from agents.synthesis import synthesis_node as _synthesis_node
from observability.phoenix_setup import setup_phoenix
from utils.model_factory import get_model_config
from utils.reranker import get_reranker

load_dotenv()

# Initialise Phoenix tracing before any LLM calls are made
setup_phoenix()

# Tracer for custom Qdrant retrieval spans (no-op when Phoenix is unavailable)
_retrieval_tracer = _otel_trace.get_tracer("workershield.retrieval")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://192.168.100.1:11434")
_qdrant_host_raw = os.environ.get("QDRANT_HOST", "http://localhost:6333")
QDRANT_HOST = _qdrant_host_raw if "://" in _qdrant_host_raw else f"http://{_qdrant_host_raw}:6333"
COLLECTION  = "workershield"
EMBED_MODEL = "nomic-embed-text"
MAX_CHARS   = 6_000
TOP_K       = 3

# Prefetch multiplier for hybrid RRF — wider candidate pool before fusion
_HYBRID_PREFETCH_K = TOP_K * 5

# Read retrieval mode from config; re-read on each module load so a config
# change takes effect on the next Python invocation without code edits.
_RETRIEVAL_MODE: str = get_model_config().get("retrieval_mode", "dense_only")
logger.info("[graph] retrieval_mode=%s", _RETRIEVAL_MODE)

# ---------------------------------------------------------------------------
# Sparse encoder — lazy singleton (fastembed Qdrant/bm25)
# ---------------------------------------------------------------------------

_sparse_encoder: SparseTextEmbedding | None = None


def _get_sparse_encoder() -> SparseTextEmbedding:
    global _sparse_encoder
    if _sparse_encoder is None:
        logger.info("[graph] initialising BM25 sparse encoder")
        _sparse_encoder = SparseTextEmbedding(model_name="Qdrant/bm25")
    return _sparse_encoder


def _sparse_embed(text: str) -> SparseVector:
    """Return a Qdrant SparseVector for text using fastembed BM25."""
    enc    = _get_sparse_encoder()
    result = next(enc.embed([text[:MAX_CHARS]]))
    return SparseVector(
        indices=result.indices.tolist(),
        values=result.values.tolist(),
    )


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
    incident_data:     list[dict]  # populated by incident_check_node when query matches
    synthesis_input:   str
    final_answer:      str
    citations:         list[dict]  # {doc_id, doc_title, section, domain, excerpt}
    confidence:        str         # "high", "medium", "low", or "insufficient"


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


def _hits_to_chunks(hits: list) -> list[dict]:
    """Convert raw Qdrant ScoredPoint objects to chunk dicts."""
    chunks = []
    for h in hits:
        p = h.payload
        chunks.append({
            "doc_id":  p.get("doc_id", ""),
            "domain":  p.get("domain", ""),
            "title":   p.get("title", ""),
            "source":  p.get("source", ""),
            "section": p.get("section", ""),
            "page":    p.get("page_estimate", 1),
            "score":   round(h.score, 4),
            "text":    p.get("text", ""),
        })
    return chunks


def _query_qdrant_dense(vec: list[float], domain: str, client: QdrantClient) -> list[dict]:
    """Dense-only retrieval via qdrant-client (named vector 'text-dense')."""
    domain_filter = Filter(
        must=[FieldCondition(key="domain", match=MatchValue(value=domain))]
    )
    hits = client.query_points(
        collection_name=COLLECTION,
        query=vec,
        using="text-dense",
        query_filter=domain_filter,
        limit=TOP_K,
        with_payload=True,
    ).points
    return _hits_to_chunks(hits)


def _query_qdrant_hybrid_full(
    dense_vec: list[float],
    sparse_vec: SparseVector,
    domain: str,
) -> list[dict]:
    """Hybrid RRF retrieval via direct REST call.

    qdrant-client 1.18.0 has a serialisation bug with query=Fusion.RRF in
    query_points; the raw REST endpoint accepts the exact same payload correctly.
    """
    domain_filter = {"must": [{"key": "domain", "match": {"value": domain}}]}
    payload = {
        "prefetch": [
            {
                "query":  dense_vec,
                "using":  "text-dense",
                "limit":  _HYBRID_PREFETCH_K,
                "filter": domain_filter,
            },
            {
                "query":  {"indices": sparse_vec.indices, "values": sparse_vec.values},
                "using":  "text-sparse",
                "limit":  _HYBRID_PREFETCH_K,
                "filter": domain_filter,
            },
        ],
        "query":        {"fusion": "rrf"},
        "limit":        TOP_K,
        "with_payload": True,
    }
    resp = requests.post(
        f"{QDRANT_HOST}/collections/{COLLECTION}/points/query",
        json=payload,
        timeout=30,
    )
    resp.raise_for_status()
    pts = resp.json()["result"]["points"]
    return [
        {
            "doc_id":  p["payload"].get("doc_id", ""),
            "domain":  p["payload"].get("domain", ""),
            "title":   p["payload"].get("title", ""),
            "source":  p["payload"].get("source", ""),
            "section": p["payload"].get("section", ""),
            "page":    p["payload"].get("page_estimate", 1),
            "score":   round(p["score"], 4),
            "text":    p["payload"].get("text", ""),
        }
        for p in pts
    ]


def _query_qdrant(vec: list[float], domain: str, client: QdrantClient) -> list[dict]:
    """Route to dense-only retrieval (named vector path)."""
    return _query_qdrant_dense(vec, domain, client)


def _retrieve(query: str, domain: str, client: QdrantClient) -> list[dict]:
    """Embed query (dense + optional sparse), query Qdrant, return top-K chunks."""
    dense_vec = _embed(query)

    if _RETRIEVAL_MODE == "hybrid":
        try:
            sparse_vec = _sparse_embed(query)
            chunks     = _query_qdrant_hybrid_full(dense_vec, sparse_vec, domain)
        except Exception as exc:
            logger.warning("[%s] hybrid retrieval failed (%s) — falling back to dense", domain, exc)
            chunks = _query_qdrant_dense(dense_vec, domain, client)
    else:
        chunks = _query_qdrant_dense(dense_vec, domain, client)

    logger.info("[%s] retrieved %d chunks  mode=%s: %s",
                domain, len(chunks), _RETRIEVAL_MODE,
                list(zip([c["doc_id"] for c in chunks],
                         [c["section"][:40] for c in chunks])))
    return chunks


def _retrieve_traced(query: str, domain: str, client: QdrantClient) -> list[dict]:
    """Like _retrieve but wrapped in a custom OTEL span with timing attributes."""
    with _retrieval_tracer.start_as_current_span("qdrant.retrieve") as span:
        span.set_attribute("workershield.domain", domain)
        span.set_attribute("workershield.retrieval_mode", _RETRIEVAL_MODE)

        t0        = time.monotonic()
        dense_vec = _embed(query)
        embed_ms  = round((time.monotonic() - t0) * 1000, 1)
        span.set_attribute("workershield.embedding_time_ms", embed_ms)

        if _RETRIEVAL_MODE == "hybrid":
            try:
                t1         = time.monotonic()
                sparse_vec = _sparse_embed(query)
                sparse_ms  = round((time.monotonic() - t1) * 1000, 1)
                span.set_attribute("workershield.sparse_embed_ms", sparse_ms)
                chunks     = _query_qdrant_hybrid_full(dense_vec, sparse_vec, domain)
                span.set_attribute("workershield.fusion", "RRF")
            except Exception as exc:
                logger.warning("[%s] hybrid retrieval failed (%s) — falling back to dense", domain, exc)
                chunks = _query_qdrant_dense(dense_vec, domain, client)
        else:
            chunks = _query_qdrant_dense(dense_vec, domain, client)

        span.set_attribute("workershield.chunks_retrieved", len(chunks))
        if chunks:
            span.set_attribute("workershield.top_chunk_score", chunks[0]["score"])

        logger.info("[%s] retrieved %d chunks  mode=%s  embed_ms=%.0f: %s",
                    domain, len(chunks), _RETRIEVAL_MODE, embed_ms,
                    list(zip([c["doc_id"] for c in chunks],
                             [c["section"][:40] for c in chunks])))
        return chunks


# ---------------------------------------------------------------------------
# Incident check — MCP client helpers
# ---------------------------------------------------------------------------

_MCP_SERVER_PATH = str(Path(__file__).parent.parent / "mcp_server" / "incident_server.py")

# Keywords that signal the user wants incident statistics or history
_INCIDENT_KEYWORDS = frozenset([
    "how many", "incident", "incidents", "cases", "history",
    "trend", "statistics", "stats", "reported",
])


def _is_incident_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in _INCIDENT_KEYWORDS)


async def _mcp_call_tool(tool_name: str, arguments: dict) -> str:
    """Call an MCP tool on the incident server via stdio transport."""
    params = StdioServerParameters(command="python3", args=[_MCP_SERVER_PATH])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            return result.content[0].text if result.content else "{}"


def _build_mcp_args(query: str) -> tuple[str, dict]:
    """Derive the best MCP tool name and arguments from the natural-language query."""
    q = query.lower()

    # Detect domain filter
    domain: str | None = None
    if "safeshift" in q or "whs" in q or "safety" in q:
        domain = "safeshift"
    elif "fairdesk" in q or "fair work" in q or "dismissal" in q or "casual" in q:
        domain = "fairdesk"
    elif "healthnav" in q or "return to work" in q or "rtw" in q or "workers comp" in q:
        domain = "healthnav"

    # Detect category keyword
    category: str | None = None
    _cat_keywords = [
        ("fatigue", ["fatigue"]),
        ("return_to_work", ["return to work", "rtw", "return-to-work"]),
        ("mental_health", ["mental health", "psychological", "burnout"]),
        ("bullying", ["bullying", "harassment"]),
        ("underpayment", ["underpayment", "underpaid"]),
        ("unfair_dismissal", ["unfair dismissal", "dismissal"]),
        ("casual_conversion", ["casual conversion"]),
        ("musculoskeletal", ["musculoskeletal", "back injury", "shoulder"]),
        ("workers_compensation", ["workers compensation", "workercomp", "worksafe"]),
        ("slip_trip_fall", ["slip", "trip", "fall"]),
        ("manual_handling", ["manual handling", "lifting"]),
    ]
    for cat, triggers in _cat_keywords:
        if any(t in q for t in triggers):
            category = cat
            break

    # Detect status filter
    status: str | None = None
    if "open" in q and ("case" in q or "incident" in q):
        status = "open"
    elif "closed" in q and ("case" in q or "incident" in q):
        status = "closed"

    # Summary tool is best for "how many" / trend / count questions
    wants_summary = any(kw in q for kw in ["how many", "count", "summary", "trend", "statistic"])
    if wants_summary:
        return "get_incident_summary", {}

    # Filtered query for specific domain / category / status requests
    args = {k: v for k, v in [("domain", domain), ("category", category), ("status", status)] if v}
    return "query_incidents", args


def _fetch_incident_data(query: str) -> list[dict]:
    """
    Call the MCP incident server to fetch data relevant to the query.
    Falls back to direct DB queries if the MCP subprocess call fails.
    """
    tool_name, args = _build_mcp_args(query)
    logger.info("[incident_check] calling MCP tool=%s args=%s", tool_name, args)

    try:
        raw = asyncio.run(_mcp_call_tool(tool_name, args))
        data = json.loads(raw)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return [data]
        return []
    except Exception as exc:
        logger.warning("[incident_check] MCP call failed (%s) — falling back to direct DB", exc)
        # Direct fallback so the node never silently returns empty
        from data.incidents_db import (  # noqa: PLC0415
            get_incident_summary as _direct_summary,
            query_incidents as _direct_query,
        )
        if tool_name == "get_incident_summary":
            return [_direct_summary()]
        return _direct_query(**{k: v for k, v in args.items() if k in ("domain", "status", "category")})


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
    chunks = _retrieve_traced(state["query"], "safeshift", client)
    return {"safeshift_chunks": chunks}


def fairdesk_node(state: WorkerShieldState) -> dict[str, Any]:
    client = QdrantClient(url=QDRANT_HOST)
    chunks = _retrieve_traced(state["query"], "fairdesk", client)
    return {"fairdesk_chunks": chunks}


def healthnav_node(state: WorkerShieldState) -> dict[str, Any]:
    client = QdrantClient(url=QDRANT_HOST)
    chunks = _retrieve_traced(state["query"], "healthnav", client)
    return {"healthnav_chunks": chunks}


def incident_check_node(state: WorkerShieldState) -> dict[str, Any]:
    """Fetch incident statistics from the MCP server and add to state."""
    data = _fetch_incident_data(state["query"])
    logger.info("[incident_check] retrieved %d incident data items", len(data))
    return {"incident_data": data}


_DOMAIN_NODES = {
    "safeshift": "safeshift_node",
    "fairdesk":  "fairdesk_node",
    "healthnav": "healthnav_node",
}


def synthesis_node(state: WorkerShieldState) -> dict[str, Any]:
    """Apply cross-encoder reranking (if enabled), then delegate to agents.synthesis."""
    reranker_cfg = get_model_config().get("reranker", {})
    rerank_enabled = reranker_cfg.get("enabled", False)
    top_n = reranker_cfg.get("top_n_per_domain", 3)

    with _retrieval_tracer.start_as_current_span("workershield.rerank") as span:
        if rerank_enabled:
            domain_chunks: dict[str, list[dict]] = {}
            if state.get("safeshift_chunks"):
                domain_chunks["safeshift"] = state["safeshift_chunks"]
            if state.get("fairdesk_chunks"):
                domain_chunks["fairdesk"] = state["fairdesk_chunks"]
            if state.get("healthnav_chunks"):
                domain_chunks["healthnav"] = state["healthnav_chunks"]

            if domain_chunks:
                reranker = get_reranker()
                reranked = reranker.rerank_by_domain(state["query"], domain_chunks, top_n=top_n)

                # Shallow-copy state so we don't mutate the LangGraph-owned dict
                state = dict(state)
                state["safeshift_chunks"] = reranked.get("safeshift", [])
                state["fairdesk_chunks"]  = reranked.get("fairdesk", [])
                state["healthnav_chunks"] = reranked.get("healthnav", [])

                all_reranked = [c for chunks in reranked.values() for c in chunks]
                top_score = max((c.get("rerank_score", 0.0) for c in all_reranked), default=0.0)
                logger.info(
                    "[rerank] domains=%s top_score=%.4f chunks_per_domain=%s",
                    list(reranked.keys()),
                    top_score,
                    {d: len(cs) for d, cs in reranked.items()},
                )
                span.set_attribute("workershield.rerank_applied", True)
                span.set_attribute("workershield.top_rerank_score", top_score)
            else:
                span.set_attribute("workershield.rerank_applied", False)
                span.set_attribute("workershield.top_rerank_score", 0.0)
        else:
            span.set_attribute("workershield.rerank_applied", False)
            span.set_attribute("workershield.top_rerank_score", 0.0)

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
    Return a list of Send objects — one per detected domain node, plus
    optionally incident_check_node when the query mentions incident statistics.
    LangGraph fans all of these out in parallel.
    """
    sends = [
        Send(_DOMAIN_NODES[d], state)
        for d in state["detected_domains"]
        if d in _DOMAIN_NODES
    ]
    if _is_incident_query(state["query"]):
        sends.append(Send("incident_check_node", state))
    return sends


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    g = StateGraph(WorkerShieldState)

    g.add_node("router_node",        router_node)
    g.add_node("safeshift_node",     safeshift_node)
    g.add_node("fairdesk_node",      fairdesk_node)
    g.add_node("healthnav_node",     healthnav_node)
    g.add_node("incident_check_node", incident_check_node)
    g.add_node("synthesis_node",     synthesis_node)
    g.add_node("output_node",        output_node)

    g.add_edge(START, "router_node")

    # Conditional fan-out from router → domain node(s) + optional incident check
    g.add_conditional_edges("router_node", _route_domains)

    # All domain nodes converge on synthesis
    for node in _DOMAIN_NODES.values():
        g.add_edge(node, "synthesis_node")

    # incident_check_node also feeds into synthesis
    g.add_edge("incident_check_node", "synthesis_node")

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
        "incident_data":    [],
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
