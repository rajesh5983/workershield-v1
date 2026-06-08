"""
WorkerShield synthesis node.

Receives a populated WorkerShieldState (all domain chunk lists filled),
calls Claude Sonnet 4.6 with a structured context block, and returns
a JSON-shaped answer with inline citations, a confidence score, and
(for cross-domain queries) an explicit cross-domain connection paragraph.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import anthropic
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """You are WorkerShield, an Australian workplace compliance assistant.
You answer questions about WHS obligations, Fair Work entitlements, and occupational health using only the provided source documents.
Always cite the specific document and section for every claim you make.
Never answer from general knowledge — only from provided context.
For cross-domain queries, include a paragraph explicitly connecting the obligations across domains.

Respond with ONLY a valid JSON object — no markdown fences, no prose outside the JSON — in exactly this shape:

{
  "answer": "<full markdown answer with inline citations like [HN01] or [FD03]>",
  "citations": [
    {
      "doc_id": "<e.g. HN01>",
      "doc_title": "<full document title>",
      "domain": "<safeshift|fairdesk|healthnav>",
      "section": "<section heading or empty string>",
      "excerpt": "<first 150 chars of the source chunk used>"
    }
  ],
  "cross_domain_connection": "<one paragraph connecting obligations across domains — omit key if not cross-domain>",
  "confidence": "<high|medium|low>"
}

Confidence guide:
- high:   multiple chunks with score > 0.75 directly address the query
- medium: relevant chunks but scores mixed or query only partially covered
- low:    few chunks match, answer may be incomplete"""


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _build_context(state: dict[str, Any]) -> str:
    """Group chunks by domain and format as a structured context block."""
    domain_order = [
        ("safeshift", state.get("safeshift_chunks", [])),
        ("fairdesk",  state.get("fairdesk_chunks",  [])),
        ("healthnav", state.get("healthnav_chunks",  [])),
    ]

    sections: list[str] = [f"QUERY: {state['query']}\n"]

    for domain, chunks in domain_order:
        if not chunks:
            continue
        domain_label = domain.upper()
        sections.append(f"── {domain_label} CHUNKS ──────────────────────────────")
        for i, c in enumerate(chunks, 1):
            sec = f" § {c['section']}" if c.get("section") else ""
            sections.append(
                f"[{c['doc_id']}] {c.get('title', '')}{sec}\n"
                f"source: {c.get('source', '')}  |  score: {c.get('score', 0):.4f}\n"
                f"{c.get('text', '')}"
            )
            if i < len(chunks):
                sections.append("---")

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Confidence scorer
# ---------------------------------------------------------------------------

def _score_confidence(all_chunks: list[dict]) -> str:
    if not all_chunks:
        return "low"
    scores = [c.get("score", 0.0) for c in all_chunks]
    high_count = sum(1 for s in scores if s >= 0.75)
    avg = sum(scores) / len(scores)
    if high_count >= 2 and avg >= 0.72:
        return "high"
    if avg >= 0.65:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Citation extractor
# ---------------------------------------------------------------------------

def _extract_citations(answer_text: str, all_chunks: list[dict]) -> list[dict]:
    """Return deduplicated citations for doc_ids that appear in the answer."""
    seen: set[str] = set()
    citations: list[dict] = []
    for c in all_chunks:
        doc_id = c.get("doc_id", "")
        if not doc_id or doc_id in seen:
            continue
        if f"[{doc_id}]" in answer_text:
            seen.add(doc_id)
            citations.append({
                "doc_id":    doc_id,
                "doc_title": c.get("title", ""),
                "domain":    c.get("domain", ""),
                "section":   c.get("section", ""),
                "excerpt":   c.get("text", "")[:150],
            })
    return citations


# ---------------------------------------------------------------------------
# Public node
# ---------------------------------------------------------------------------

def synthesis_node(state: dict[str, Any]) -> dict[str, Any]:
    """
    LangGraph node — synthesise a cited answer from retrieved chunks.

    Returns partial state update:
      synthesis_input, final_answer, citations
    """
    all_chunks: list[dict] = (
        state.get("safeshift_chunks", [])
        + state.get("fairdesk_chunks",  [])
        + state.get("healthnav_chunks", [])
    )

    synthesis_input = _build_context(state)
    cross_domain    = state.get("cross_domain", False)

    logger.info(
        "[synthesis] chunks=%d  cross_domain=%s  context_chars=%d",
        len(all_chunks), cross_domain, len(synthesis_input),
    )

    # ── Sonnet call ─────────────────────────────────────────────────────────
    try:
        api_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        message    = api_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": synthesis_input}],
        )
        raw = message.content[0].text.strip()

        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.rsplit("```", 1)[0]

        parsed: dict = json.loads(raw)

    except json.JSONDecodeError as exc:
        logger.error("[synthesis] JSON parse failed: %s\nraw=%r", exc, raw[:300])
        parsed = {
            "answer":     raw,
            "citations":  [],
            "confidence": "low",
        }
    except Exception as exc:
        logger.error("[synthesis] API call failed: %s", exc)
        parsed = {
            "answer": (
                f"**WorkerShield synthesis unavailable** (API error: {exc})\n\n"
                + "\n\n".join(
                    f"**[{c['doc_id']}]** {c.get('title', '')}\n> {c.get('text', '')[:300]}"
                    for c in all_chunks
                )
            ),
            "citations":  [],
            "confidence": "low",
        }

    # ── Post-process ─────────────────────────────────────────────────────────
    answer_text = parsed.get("answer", "")

    # Re-derive citations from chunks if Sonnet omitted or under-cited
    model_citations = parsed.get("citations", [])
    derived         = _extract_citations(answer_text, all_chunks)
    # Merge: model citations take precedence for doc_ids it covered;
    # derived fills in any doc_ids referenced in the answer but missing from model output
    model_ids       = {c["doc_id"] for c in model_citations}
    merged_citations = model_citations + [c for c in derived if c["doc_id"] not in model_ids]

    # Inject confidence if model didn't provide a valid value
    confidence = parsed.get("confidence", "")
    if confidence not in ("high", "medium", "low"):
        confidence = _score_confidence(all_chunks)

    # Build the final structured output stored in state
    final_structured: dict = {
        "answer":     answer_text,
        "citations":  merged_citations,
        "confidence": confidence,
    }
    if cross_domain and parsed.get("cross_domain_connection"):
        final_structured["cross_domain_connection"] = parsed["cross_domain_connection"]

    logger.info(
        "[synthesis] confidence=%s  citations=%d  answer_chars=%d",
        confidence, len(merged_citations), len(answer_text),
    )

    return {
        "synthesis_input": synthesis_input,
        "final_answer":    json.dumps(final_structured, ensure_ascii=False, indent=2),
        "citations":       merged_citations,
    }


# ---------------------------------------------------------------------------
# Standalone test harness
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import requests as _requests

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    OLLAMA  = os.environ.get("OLLAMA_HOST", "http://192.168.100.1:11434")
    QDRANT  = os.environ.get("QDRANT_HOST", "http://localhost:6333")

    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    DEMO_QUERY  = (
        "My FIFO worker has a mental health condition and wants to reduce hours "
        "— what are my obligations under safety law and fair work?"
    )
    TOP_K       = 3
    EMBED_MODEL = "nomic-embed-text"
    MAX_CHARS   = 6_000

    def _embed(text: str) -> list[float]:
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS]
        r = _requests.post(
            f"{OLLAMA}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["embedding"]

    def _retrieve(domain: str) -> list[dict]:
        client = QdrantClient(url=QDRANT)
        vec    = _embed(DEMO_QUERY)
        hits   = client.query_points(
            collection_name="workershield",
            query=vec,
            query_filter=Filter(
                must=[FieldCondition(key="domain", match=MatchValue(value=domain))]
            ),
            limit=TOP_K,
            with_payload=True,
        ).points
        return [
            {
                "doc_id":  h.payload.get("doc_id", ""),
                "domain":  h.payload.get("domain", ""),
                "title":   h.payload.get("title", ""),
                "source":  h.payload.get("source", ""),
                "section": h.payload.get("section", ""),
                "page":    h.payload.get("page_estimate", 1),
                "score":   round(h.score, 4),
                "text":    h.payload.get("text", ""),
            }
            for h in hits
        ]

    # Build a mock state with all three domains populated
    mock_state = {
        "query":            DEMO_QUERY,
        "detected_domains": ["safeshift", "fairdesk", "healthnav"],
        "cross_domain":     True,
        "safeshift_chunks": _retrieve("safeshift"),
        "fairdesk_chunks":  _retrieve("fairdesk"),
        "healthnav_chunks": _retrieve("healthnav"),
        "synthesis_input":  "",
        "final_answer":     "",
        "citations":        [],
    }

    print(f"\nQuery  : {DEMO_QUERY}")
    print(f"Chunks : safeshift={len(mock_state['safeshift_chunks'])}  "
          f"fairdesk={len(mock_state['fairdesk_chunks'])}  "
          f"healthnav={len(mock_state['healthnav_chunks'])}\n")

    result = synthesis_node(mock_state)

    print("=" * 72)
    print("SYNTHESIS OUTPUT (JSON)")
    print("=" * 72)
    print(result["final_answer"])
    print("=" * 72)
