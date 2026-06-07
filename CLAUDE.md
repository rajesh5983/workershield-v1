# WorkerShield v1 — Claude Code Project Context

## Project

WorkerShield is an agentic RAG platform providing cited, practical guidance across three
Australian workplace compliance domains: **SafeShift** (WHS law), **FairDesk** (Fair Work),
and **HealthNav** (Occupational Health). Built by Raj Prasannakumar under BrickByData /
ModernAnalyticsLab as a portfolio project demonstrating production-grade agentic RAG
architecture for technical hiring managers at Microsoft-aligned data and AI consulting
firms evaluating Fabric Practice Lead candidates.

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent framework | LangGraph (`StateGraph`) |
| Vector store | Qdrant — Docker, `localhost:6333`, collection: `workershield` |
| Embedding model | Ollama `nomic-embed-text` — local, 768 dimensions |
| Router LLM | Claude Haiku via Anthropic API — JSON classification |
| Synthesis LLM | Claude Sonnet 4 via Anthropic API — cited answers |
| Demo UI | Gradio — `ui/app.py`, `localhost:7860` |
| Observability | JSONL run logs — `logs/run_log.jsonl`, `utils/logger.py` |
| Dev environment | PortfolioLab Hyper-V, Ubuntu 24.04, VS Code Remote SSH |

---

## Three Domains

| Domain key | Scope |
|---|---|
| `safeshift` | WHS law, QLD codes of practice, WHS Act duties, PPE, manual handling, fatigue (safety lens) |
| `fairdesk` | Fair Work Act, NES, casual conversion, flexible working, termination, modern awards |
| `healthnav` | Occupational health, workers compensation, mental health at work, WorkCover QLD obligations |

---

## Corpus

9 documents — 3 per domain — all Australian open government sources.
Full document list with source URLs and chunk configuration: `corpus/corpus_registry.yaml`.
Chunk strategy: sliding window, 400 tokens, 50 token overlap, applied uniformly across all documents.

---

## LangGraph State

`WorkerShieldState` TypedDict — all inter-node data flows through this object:

```python
query: str                     # raw user input
detected_domains: list[str]    # router output: ['safeshift', 'fairdesk', 'healthnav']
cross_domain: bool             # True when query spans multiple domains
safeshift_chunks: list[dict]   # top-5 chunks from SafeShift partition
fairdesk_chunks: list[dict]    # top-5 chunks from FairDesk partition
healthnav_chunks: list[dict]   # top-5 chunks from HealthNav partition
synthesis_input: str           # assembled context string passed to Sonnet
final_answer: str              # Sonnet's cited answer
citations: list[dict]          # [{doc_title, section, domain}, ...]
```

---

## Graph Flow

```
START → router_node → retrieval_node → synthesis_node → output_node → END
```

Conditional edge after `router_node`:
- `cross_domain = True` → retrieval queries **all three** domain partitions
- `cross_domain = False` → retrieval queries **only** `detected_domains`

---

## Key Files

| File | Purpose |
|---|---|
| `corpus/corpus_registry.yaml` | Document registry — drives the full ingestion pipeline |
| `prompts/PROMPTS.md` | Master prompt reference — router, synthesis, citation format |
| `agents/router.py` | Domain classifier — Haiku + keyword fallback |
| `agents/retrieval.py` | Three domain retrievers + cross-domain merge |
| `agents/graph.py` | LangGraph state machine definition |
| `ingest/load_qdrant.py` | PDF extraction, chunking, embedding, Qdrant upsert |
| `ui/app.py` | Gradio demo interface |
| `utils/logger.py` | JSONL run logging |
| `utils/log_reader.py` | Log summary viewer |
| `docs/ARCHITECTURE.md` | Full system architecture reference |

---

## Coding Conventions

- **Python:** `snake_case`, type hints on all functions, Google-style docstrings
- All agent functions return typed dicts matching `WorkerShieldState` fields
- Wrap all Anthropic and Qdrant API calls in `try/except`; log errors to run log
- **Commit prefixes:** `feat:` / `fix:` / `docs:` / `refactor:` / `test:`
- Australian English in all documentation and inline comments

---

## The Killer Demo Query

```
"My FIFO worker has a mental health condition and wants to reduce hours —
what are my obligations under safety law and fair work?"
```

Expected behaviour: `cross_domain = True`, all three retrievers fire, citations drawn
from SafeShift (WHS duty of care), FairDesk (flexible working NES entitlements), and
HealthNav (mental health employer obligations) documents in a single synthesised answer.
