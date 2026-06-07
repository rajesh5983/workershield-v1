# WorkerShield v1 — System Architecture

---

## 1. System Overview

WorkerShield is a three-domain agentic Retrieval-Augmented Generation (RAG) platform designed to answer complex Australian workplace compliance questions that span safety law, fair work entitlements, and occupational health obligations. Rather than routing queries to a single knowledge base, the system operates across three discrete retrieval domains — SafeShift, FairDesk, and HealthNav — and employs a LangGraph-orchestrated agent pipeline to classify each query, retrieve contextually relevant regulatory content, and synthesise a single, coherent, citation-grounded response. The architecture is purpose-built to handle the class of question that currently consumes disproportionate legal expenditure in Australian organisations: multi-domain compliance queries that touch more than one regulatory framework simultaneously.

The primary audience for WorkerShield is HR managers, WHS officers, and operations leads within Australian SMBs and mid-market organisations who currently depend on costly external legal advice for queries that are, in substance, routine. By grounding responses directly in authoritative Australian open-data sources — WorkSafe Queensland, Fair Work Commission, and Safe Work Australia publications — WorkerShield delivers legally traceable answers with inline citations, enabling practitioners to act with confidence on first-line queries and escalate to counsel only when genuinely necessary.

---

## 2. High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT LAYER                                                    │
│                                                                 │
│                    User Query (Gradio UI)                       │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  AGENT LAYER  ·  LangGraph StateGraph                           │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Router Agent  ·  Claude Haiku                          │   │
│   │  Classifies query → detected_domains + cross_domain     │   │
│   └────────────────────────┬────────────────────────────────┘   │
│                            │                                    │
│            conditional edges on detected_domains                │
│                            │                                    │
│          ┌─────────────────┼──────────────────┐                 │
│          ▼                 ▼                  ▼                 │
│   ┌────────────┐   ┌──────────────┐   ┌──────────────┐         │
│   │ SafeShift  │   │  FairDesk    │   │  HealthNav   │         │
│   │ Retrieval  │   │  Retrieval   │   │  Retrieval   │         │
│   │ (Qdrant)   │   │  (Qdrant)    │   │  (Qdrant)    │         │
│   └─────┬──────┘   └──────┬───────┘   └──────┬───────┘         │
│         └─────────────────┼──────────────────┘                 │
│                           │ chunks with metadata                │
│                           ▼                                     │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Synthesis Agent  ·  Claude Sonnet                      │   │
│   │  Assembles context → generates cited answer             │   │
│   └────────────────────────┬────────────────────────────────┘   │
│                            │                                    │
└────────────────────────────┼────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  OUTPUT LAYER                                                   │
│                                                                 │
│   Gradio UI  ·  answer + citations + domain indicators          │
│   JSONL Run Log  ·  observability                               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  DATA LAYER  (serves all retrieval nodes)                       │
│                                                                 │
│   Qdrant Collection: workershield                               │
│   3 domain partitions via payload metadata filtering            │
│   768-dim vectors  ·  nomic-embed-text  ·  cosine distance      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. LangGraph State Machine

All inter-node data flows through a single immutable `WorkerShieldState` TypedDict. Nodes read from and write to this shared state object; no direct inter-node coupling exists outside of it.

| Field | Type | Purpose |
|---|---|---|
| `query` | `str` | The user's raw input question, unmodified from the Gradio interface |
| `detected_domains` | `list[str]` | Domains identified by the Router Agent; values are `safeshift`, `fairdesk`, `healthnav` |
| `cross_domain` | `bool` | `True` when the Router determines the query spans multiple regulatory domains; triggers full three-domain retrieval |
| `safeshift_chunks` | `list[dict]` | Top-5 chunks retrieved from the SafeShift domain partition, each carrying payload metadata (doc_id, title, section) |
| `fairdesk_chunks` | `list[dict]` | Top-5 chunks retrieved from the FairDesk domain partition, each carrying payload metadata |
| `healthnav_chunks` | `list[dict]` | Top-5 chunks retrieved from the HealthNav domain partition, each carrying payload metadata |
| `synthesis_input` | `str` | The assembled context string — all retrieved chunks concatenated with domain labels — passed to the Synthesis Agent |
| `final_answer` | `str` | Sonnet's synthesised, citation-grounded answer ready for display |
| `citations` | `list[dict]` | Structured citation list; each entry carries `doc_title`, `section`, and `domain` for rendering in the UI |

---

## 4. Node Responsibilities

| Node | LLM / Tool | Input Fields | Output Fields | Key Logic |
|---|---|---|---|---|
| `router_node` | Claude Haiku | `query` | `detected_domains`, `cross_domain` | Sends the query to Haiku with a classification prompt; expects a structured JSON response `{domains: [...], cross_domain: bool, reasoning: str}`. Falls back to keyword matching (e.g. "WHS", "manual task" → `safeshift`) if the model returns malformed output. Fast and cheap by design — Haiku keeps routing latency negligible. |
| `retrieval_node` | Qdrant | `detected_domains`, `cross_domain` | `safeshift_chunks`, `fairdesk_chunks`, `healthnav_chunks` | Embeds the query using the local Ollama `nomic-embed-text` model (768 dimensions). Queries Qdrant with a payload filter `{domain: <target>}`. Retrieves top-5 chunks per active domain. When `cross_domain` is `True`, all three domain collections are queried regardless of `detected_domains`. |
| `synthesis_node` | Claude Sonnet | `safeshift_chunks`, `fairdesk_chunks`, `healthnav_chunks`, `query` | `final_answer`, `citations` | Assembles a structured context string from all available chunks, labelled by domain. Passes this to Sonnet with an instruction to answer the user's query using only the provided context and to format citations inline as `[Source: <title> — <section>]`. Extracts structured citations from the response for separate rendering. |
| `output_node` | None | `final_answer`, `citations` | Formatted Gradio output | No model call. Formats the answer and citation list into the Gradio component schema and appends a structured JSONL record to the run log for observability. |

---

## 5. Conditional Graph Edges

The critical branching decision occurs immediately after `router_node` completes. The conditional edge function inspects the `cross_domain` flag on the state object:

- **`cross_domain = True`** — The retrieval node queries all three domain partitions (SafeShift, FairDesk, HealthNav) unconditionally, regardless of which specific domains the router identified. This ensures that a multi-domain query never silently drops relevant context due to classification uncertainty.

- **`cross_domain = False`** — The retrieval node queries only the domain or domains listed in `detected_domains`. Single-domain queries are handled efficiently without incurring the latency or token cost of unnecessary retrievals.

This design is the principal architectural decision that enables WorkerShield's signature capability: a single query can draw authoritative, cited content from all three regulatory frameworks and present it as a unified response. It also means the cross-domain retrieval path is always deterministic — the router's confidence threshold for setting `cross_domain = True` is deliberately conservative, preferring over-retrieval to missed context.

---

## 6. The Killer Demo Query Path

**Query submitted:**
> *"My FIFO worker has a mental health condition and wants to reduce hours — what are my obligations under safety law and fair work?"*

**Step 1 — Router Node (Claude Haiku)**
Haiku classifies the query against all three domain schemas. It detects signals across all domains: "FIFO worker" and "mental health condition" activate `safeshift` (duty of care, fatigue) and `healthnav` (mental health employer obligations); "wants to reduce hours" activates `fairdesk` (flexible working entitlements). Because the query clearly spans multiple frameworks, Haiku sets `cross_domain = True`.

**Step 2 — Conditional Edge**
Because `cross_domain = True`, the graph routes to the retrieval node with an instruction to query all three domain partitions — no filtering applied.

**Step 3 — Retrieval Node (Qdrant × 3)**
The query is embedded via `nomic-embed-text`. Three separate Qdrant queries execute in sequence, each filtered by domain:

- **SafeShift chunks** — returns content on PCBU duty of care obligations, managing psychosocial hazards, and mental health as a workplace risk under the WHS Act 2011.
- **FairDesk chunks** — returns content on flexible working arrangement entitlements under the National Employment Standards, including who may request reduced hours and grounds for refusal.
- **HealthNav chunks** — returns content on employer obligations when a worker discloses a mental health condition, including reasonable adjustments and return-to-work planning.

**Step 4 — Synthesis Node (Claude Sonnet)**
Sonnet receives a structured context block containing all 15 retrieved chunks, organised by domain. It synthesises a single coherent answer that:

- Explains the PCBU's duty to manage psychosocial hazards under the WHS Act (SafeShift)
- Confirms the worker's right to request flexible working under the NES and the employer's limited grounds for refusal (FairDesk)
- Outlines the employer's obligations upon disclosure of a mental health condition, including reasonable adjustments (HealthNav)

Each claim is attributed inline, e.g. `[Source: WHS Act Key Duties — PCBU Obligations]`.

**Step 5 — Output Node**
The formatted answer and structured citation list are returned to the Gradio UI. Domain indicators flag that all three knowledge domains contributed to the response. The full run is appended to the JSONL log.

---

## 7. Technology Stack

| Component | Technology | Purpose | Notes |
|---|---|---|---|
| Agent Orchestration | LangGraph (LangChain) | Defines the `StateGraph`, nodes, edges, and conditional routing logic | Chosen for explicit state management and deterministic edge control |
| Router LLM | Claude Haiku (Anthropic) | Fast, low-cost query classification; returns structured JSON | Haiku keeps router latency under ~500 ms |
| Synthesis LLM | Claude Sonnet (Anthropic) | High-quality multi-document synthesis with citation formatting | Sonnet balances quality and cost for production-grade answers |
| Embedding Model | `nomic-embed-text` via Ollama | Converts query and document chunks to 768-dim vectors | Runs locally; no external embedding API dependency |
| Vector Database | Qdrant | Stores and retrieves document chunks with payload metadata filtering | Single collection with domain partitioning via metadata filters |
| Document Ingestion | PyPDF + LangChain Text Splitters | PDF parsing and sliding-window chunking (400 tokens, 50 overlap) | Configured via `corpus_registry.yaml` |
| UI | Gradio | Web interface for query input, answer display, and citation rendering | Minimal dependency footprint; suitable for portfolio demonstration |
| API Layer | Python (FastAPI-compatible) | Optional programmatic access to the agent pipeline | Decoupled from UI via `api/` module |
| Observability | JSONL run log | Appends structured records per query (input, domains, chunks, answer) | Written by `output_node`; stored in `logs/` |
| Configuration | PyYAML | Drives corpus ingestion from `corpus/corpus_registry.yaml` | Single source of truth for all document metadata |
| LLM SDK | Anthropic Python SDK | Manages Claude API calls for router and synthesis nodes | |

---

## 8. File Structure

```
workershield-v1/
│
├── agents/                         # LangGraph agent definitions
│   └── __init__.py                 # Package init; will export graph, nodes, state
│
├── api/                            # Optional API layer for programmatic access
│   └── __init__.py                 # Package init; will expose query endpoint
│
├── corpus/                         # All document data
│   ├── corpus_registry.yaml        # Master registry: 9 docs, metadata, chunk config
│   ├── raw/                        # Downloaded source PDFs (9 files)
│   │   └── .gitkeep
│   └── chunked/                    # Post-ingestion chunked JSON output
│       └── .gitkeep
│
├── docs/                           # Project documentation
│   └── ARCHITECTURE.md             # This document
│
├── ingest/                         # Ingestion pipeline
│   └── __init__.py                 # Package init; will expose ingest runner
│
├── logs/                           # JSONL run logs (observability)
│   └── .gitkeep
│
├── prompts/                        # Prompt templates for router and synthesis nodes
│
├── tests/                          # Test suite
│   └── __init__.py                 # Package init
│
├── ui/                             # Gradio UI
│   └── __init__.py                 # Package init; will expose Gradio app
│
├── utils/                          # Shared utilities (embedding, Qdrant client, etc.)
│   └── __init__.py                 # Package init
│
├── .gitignore                      # Standard Python ignores + corpus/raw PDFs
├── requirements.txt                # Python dependencies
└── README.md                       # Project overview (to be created)
```
