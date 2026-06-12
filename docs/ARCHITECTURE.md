# WorkerShield v1 — System Architecture

---

## 1. System Overview

WorkerShield is a three-domain agentic RAG platform for Australian workplace compliance. A LangGraph `StateGraph` orchestrates six node types — router, domain retrievers, incident check, synthesis, output — connected by conditional fan-out edges that activate one, two, or all three Qdrant partitions plus an optional MCP-backed incident statistics node. The synthesis node runs a pre-LLM refusal threshold check; queries whose retrieved chunks fall below calibrated confidence thresholds are declined gracefully rather than synthesised from weak evidence. Every query is traced end-to-end via Arize Phoenix.

---

## 2. System Architecture

```mermaid
flowchart LR
    User(["User Query"])
    UI["Gradio UI\nlocalhost:7860"]
    Router["router_node\nClaude Haiku\nJSON classification"]
    SS["safeshift_node\nQdrant retriever"]
    FD["fairdesk_node\nQdrant retriever"]
    HN["healthnav_node\nQdrant retriever"]
    IC["incident_check_node\nMCP client"]
    Synth["synthesis_node\nClaude Sonnet"]
    Threshold{{"Refusal\nThreshold"}}
    Out["output_node\nJSONL log"]
    Qdrant[("Qdrant :6333\ncollection: workershield\n1,268 vectors · 768-dim cosine")]
    Ollama["Ollama :11434\nnomic-embed-text\n768d local embeddings"]
    Phoenix["Arize Phoenix :6006\nOTEL tracing"]
    MCP["MCP Server\nmcp_server/incident_server.py\nFastMCP stdio"]
    SQLite[("SQLite\ndata/incidents.db\n50 records · 3 domains")]

    User --> UI
    UI --> Router
    Router -->|"safeshift"| SS
    Router -->|"fairdesk"| FD
    Router -->|"healthnav"| HN
    Router -->|"incident keywords"| IC
    SS & FD & HN & IC --> Synth
    Synth --> Threshold
    Threshold -->|"avg≥0.65 or max≥0.70\nnormal synthesis"| Out
    Threshold -->|"avg<0.65 AND max<0.70\nrefusal response"| Out
    Out -->|"answer + citations\nconfidence badge"| UI
    UI --> User

    Qdrant -.->|domain-filtered search| SS
    Qdrant -.->|domain-filtered search| FD
    Qdrant -.->|domain-filtered search| HN
    Ollama -.->|embed query| SS
    Ollama -.->|embed query| FD
    Ollama -.->|embed query| HN
    MCP -.->|stdio transport| IC
    SQLite -.->|SQL queries| MCP
    Phoenix -.->|OTEL spans| Router
    Phoenix -.->|OTEL spans| Synth
```

---

## 3. LangGraph State Machine

All inter-node data flows through a single `WorkerShieldState` TypedDict. No direct node-to-node coupling exists outside this object. The `safeshift_chunks`, `fairdesk_chunks`, and `healthnav_chunks` fields are annotated with `operator.add` so parallel retrieval nodes can append without overwriting each other.

### State fields

| Field | Type | Set by | Purpose |
|---|---|---|---|
| `query` | `str` | caller | Raw user input, unmodified |
| `detected_domains` | `list[str]` | `router_node` | Domains identified: `safeshift`, `fairdesk`, `healthnav` |
| `cross_domain` | `bool` | `router_node` | `True` → fan out to all three domain nodes |
| `safeshift_chunks` | `Annotated[list[dict], operator.add]` | `safeshift_node` | Top-K chunks from SafeShift partition (score, text, metadata) |
| `fairdesk_chunks` | `Annotated[list[dict], operator.add]` | `fairdesk_node` | Top-K chunks from FairDesk partition |
| `healthnav_chunks` | `Annotated[list[dict], operator.add]` | `healthnav_node` | Top-K chunks from HealthNav partition |
| `incident_data` | `list[dict]` | `incident_check_node` | Incident records or summary dict from MCP server; empty list when node not triggered |
| `synthesis_input` | `str` | `synthesis_node` | Assembled context string passed to Sonnet (includes incident data section when present) |
| `final_answer` | `str` | `synthesis_node` | JSON-serialised answer object (answer, citations, confidence) |
| `citations` | `list[dict]` | `synthesis_node` | Structured citation list for UI rendering |
| `confidence` | `str` | `synthesis_node` | `"high"` / `"medium"` / `"low"` / `"insufficient"` |

### State machine graph

```mermaid
stateDiagram-v2
    [*] --> router_node
    router_node --> safeshift_node      : safeshift in domains\nor cross_domain=True
    router_node --> fairdesk_node       : fairdesk in domains\nor cross_domain=True
    router_node --> healthnav_node      : healthnav in domains\nor cross_domain=True
    router_node --> incident_check_node : incident keywords\nin query
    safeshift_node      --> synthesis_node
    fairdesk_node       --> synthesis_node
    healthnav_node      --> synthesis_node
    incident_check_node --> synthesis_node

    state incident_check_node {
        [*] --> detect_tool
        detect_tool --> call_summary  : "how many" / trend
        detect_tool --> call_filtered : specific domain/category
        call_summary  --> [*]
        call_filtered --> [*]
    }

    state synthesis_node {
        [*] --> refusal_check
        refusal_check --> llm_call    : avg≥0.65 OR max≥0.70
        refusal_check --> refusal_out : avg<0.65 AND max<0.70
        llm_call --> [*]
        refusal_out --> [*]
    }

    synthesis_node --> output_node
    output_node --> [*]
```

### Conditional routing

After `router_node`, LangGraph's `Send` primitive fans out to domain nodes (and optionally to `incident_check_node`):

- **`cross_domain = True`** → all three domain nodes run (in parallel), regardless of which domains the router detected. This ensures multi-domain queries never silently drop context.
- **`cross_domain = False`** → only the node(s) in `detected_domains` run.
- **Incident keywords detected** → `incident_check_node` is added to the `Send` list and runs in parallel with domain retrievers. Triggered when the query contains any of: `"how many"`, `"incident"`, `"incidents"`, `"cases"`, `"history"`, `"trend"`, `"statistics"`, `"stats"`, `"reported"`.

The router sets `cross_domain = True` conservatively — preferring over-retrieval to missed context.

---

## 4. Node Responsibilities

| Node | LLM / Tool | Key Logic |
|---|---|---|
| `router_node` | Claude Haiku | Sends the query with a JSON classification prompt; expects `{domains: [...], cross_domain: bool, reasoning: str}`. Falls back to keyword matching (`"WHS"` → `safeshift`, `"casual"` → `fairdesk`, etc.) when the model returns malformed output. |
| `safeshift_node` | Qdrant + Ollama | Embeds the query via `nomic-embed-text` (768d). Queries Qdrant filtered by `domain=safeshift`. Returns top-K chunk dicts with score, text, doc_id, title, section. Wrapped in a custom OTEL span recording embed time and top-chunk score. |
| `fairdesk_node` | Qdrant + Ollama | Same as above, filtered by `domain=fairdesk`. |
| `healthnav_node` | Qdrant + Ollama | Same as above, filtered by `domain=healthnav`. |
| `incident_check_node` | MCP client → SQLite | Detects whether the query wants summary counts or a filtered list. Calls the MCP server via `mcp.client.stdio` (spawns `incident_server.py` as a subprocess). Returns `incident_data` list — either a summary dict or filtered records. Falls back to direct SQLite queries via `data/incidents_db.py` if MCP call fails. |
| `synthesis_node` | Claude Sonnet | (1) Checks refusal threshold across all chunks. (2) If proceeding, assembles domain-labelled context + optional `── INCIDENT DATABASE ──` section. (3) Calls Sonnet with a strict JSON-output system prompt. (4) Post-processes: unwraps double-encoded responses, heals unescaped inner quotes, derives citations, computes confidence. |
| `output_node` | None | Logs the run to `logs/run_log.jsonl`. No state mutation. |

### Refusal threshold

Before the synthesis LLM call, `synthesis_node` checks:

```
avg_score = mean([chunk["score"] for chunk in all_chunks])
max_score = max([chunk["score"] for chunk in all_chunks])

if avg_score < 0.65 AND max_score < 0.70:
    → skip LLM, return structured refusal with confidence="insufficient"
```

Thresholds calibrated from observed score distributions:

| Query type | avg_score | max_score |
|---|---|---|
| Out-of-scope (e.g. "capital of France") | ~0.58 | ~0.62 |
| In-scope (e.g. "psychosocial hazards under WHS") | ~0.74 | ~0.77 |

The gap between 0.62 and 0.74 gives clean separation with the chosen thresholds.

---

## 5. MCP Incident Database

### Overview

A FastMCP server (`mcp_server/incident_server.py`) exposes a SQLite incident database as Model Context Protocol tools. This enables both external MCP clients (Claude Code) and the internal `incident_check_node` to query workplace incident records alongside the document corpus.

### Architecture

```mermaid
flowchart LR
    subgraph external["External MCP clients"]
        CC["Claude Code\nclaude mcp add workershield-incidents"]
    end

    subgraph agent["WorkerShield agent"]
        IC["incident_check_node\nasyncio.run → mcp.client.stdio"]
    end

    subgraph server["MCP Server"]
        MCP["incident_server.py\nFastMCP stdio"]
    end

    subgraph storage["Shared storage"]
        DB["data/incidents_db.py\nSQLite query helpers"]
        SQLite[("data/incidents.db\n50 records · 3 domains\nJune 2025 – June 2026")]
    end

    CC -->|"stdio transport"| MCP
    IC -->|"stdio transport\n(subprocess)"| MCP
    MCP --> DB
    DB --> SQLite
    IC -->|"direct fallback\nif MCP fails"| DB
```

### Tools exposed

| Tool | Arguments | Returns |
|---|---|---|
| `get_incident_summary()` | — | Counts grouped by domain × status; category breakdown; avg resolution time |
| `query_incidents(domain, status, category, date_from, date_to, limit)` | All optional | Filtered list of incident records (max 50) |
| `get_incident_detail(incident_id)` | `incident_id: str` | Full record for one incident, or error if not found |

### Tool selection logic

`_build_mcp_args()` in `agents/graph.py` selects the tool from query signals:

- "how many" / "count" / "summary" / "trend" → `get_incident_summary`
- Specific domain/category/status keywords detected → `query_incidents` with extracted filters
- Exact ID mentioned → `get_incident_detail`

### Registration

```bash
claude mcp add workershield-incidents \
  -- python3 /projects/workershield-v1/mcp_server/incident_server.py
```

Self-test (no MCP client required): `python3 mcp_server/incident_server.py --test`

---

## 6. Corpus and Chunking

10 logical doc_ids across 9 registered source documents. SS03 is split at ingestion into two doc_ids (legislative clauses vs. duties guide) to enable per-section retrieval precision.

```mermaid
flowchart TD
    subgraph safeshift["SafeShield — WHS Law"]
        SS01["SS01\nManaging Work Environment CoP\nsection_header · 42 pages"]
        SS02["SS02\nHazardous Manual Tasks CoP\nsection_header · 71 pages"]
        SS03a["SS03a\nQLD WHS Act 2011\nclause_boundary · 308 pages"]
        SS03b["SS03b\nGuide to Model WHS Act\nclause_boundary · 42 pages"]
    end
    subgraph fairdesk["FairDesk — Fair Work"]
        FD01["FD01\nIntroduction to NES\nrecursive · 2 pages"]
        FD02["FD02\nCasual Employment Statement\nrecursive · 3 pages"]
        FD03["FD03\nFlexible Working Guide\nrecursive"]
    end
    subgraph healthnav["HealthNav — Occupational Health"]
        HN01["HN01\nWork-Related Mental Health\nsection_header · 43 pages"]
        HN02["HN02\nFatigue Fact Sheet\nrecursive · 3 pages"]
        HN03["HN03\nWorkers Compensation Guide\nsection_header · 12 pages"]
    end

    Qdrant[("Qdrant\ncollection: workershield\n1,268 vectors · 768-dim cosine")]

    SS01 & SS02 & SS03a & SS03b --> Qdrant
    FD01 & FD02 & FD03 --> Qdrant
    HN01 & HN02 & HN03 --> Qdrant
```

**Chunking parameters (uniform):** 400-token window, 50-token overlap, sliding stride.

**Strategy selection per document:**

| Strategy | When used | Documents |
|---|---|---|
| `section_header` | Long documents with clear numbered sections | SS01, SS02, HN01, HN03 |
| `clause_boundary` | Legislation with strict numbered clause hierarchy | SS03a, SS03b |
| `recursive` | Short fact sheets and prose-heavy guides | FD01, FD02, FD03, HN02 |

See [`docs/CHUNKING_DECISIONS.md`](CHUNKING_DECISIONS.md) for full per-document rationale.

---

## 7. Ingest Pipeline

Runs once to populate Qdrant. Driven entirely by `corpus/corpus_registry.yaml` — adding a new document requires only a registry entry and the PDF.

```mermaid
flowchart LR
    Registry["corpus_registry.yaml\n9 document entries\nchunk strategy per doc"]
    PDFs["PDF files\ncorpus/raw/"]
    pypdf["pypdf\npage-by-page text extraction"]
    chunker["ingest/load_qdrant.py\nsliding window\n400 tok · 50 overlap"]
    embed["Ollama :11434\nnomic-embed-text\n768-dim vectors"]
    Qdrant[("Qdrant :6333\ncollection: workershield\nupsert PointStruct")]

    Registry -->|"strategy + metadata\nper doc_id"| chunker
    PDFs --> pypdf
    pypdf -->|"full page text"| chunker
    chunker -->|"chunk dict\n+ payload metadata"| embed
    embed -->|"vector + payload\ndoc_id, domain, title\nsection, page, text"| Qdrant
```

**Payload fields stored per vector:** `doc_id`, `domain`, `title`, `source` (URL), `section`, `page_estimate`, `text`.

---

## 8. Observability Stack

```mermaid
flowchart LR
    subgraph prod["Production — per query"]
        Router2["router_node\nHaiku call"]
        Synth2["synthesis_node\nSonnet call"]
        Log["output_node\nlogs/run_log.jsonl"]
    end

    subgraph phoenix["Arize Phoenix — live tracing"]
        PhoenixUI["Phoenix UI\nlocalhost:6006\nOTEL trace viewer"]
        Spans["Per-query spans:\n• router LLM call\n• qdrant.retrieve × domain\n• synthesis LLM call\n• embedding_time_ms\n• top_chunk_score\n• token usage"]
    end

    subgraph ragas["RAGAS — offline evaluation"]
        GoldenSet["8-query golden dataset\ntests/ragas_eval.py"]
        Judge["OpenAI GPT-4o-mini\njudge LLM"]
        EmbedJudge["text-embedding-3-small\njudge embeddings"]
        Scores["4 metrics:\nFaithfulness · Context Precision\nContext Recall · Answer Relevancy"]
        Results["tests/ragas_results.json\ntests/RAGAS_RESULTS.md"]
    end

    Router2 -.->|OTEL auto-instrumentation\nAnthropicInstrumentor| PhoenixUI
    Synth2 -.->|OTEL auto-instrumentation| PhoenixUI
    Synth2 -.->|custom spans\n_retrieval_tracer| Spans
    Spans --> PhoenixUI

    GoldenSet --> Judge
    GoldenSet --> EmbedJudge
    Judge & EmbedJudge --> Scores
    Scores --> Results
```

**RAGAS results summary (run date: 2026-06-12):**

| Metric | Score | Target | Status |
|---|---|---|---|
| Faithfulness | 0.894 | ≥ 0.85 | ✅ |
| Context Precision | 0.750 | ≥ 0.70 | ✅ |
| Context Recall | 0.750 | ≥ 0.70 | ✅ |
| Answer Relevancy | 0.639 | ≥ 0.80 | ⚠️ |

Answer Relevancy misses target primarily on Q2 (casual overtime on public holidays — corpus coverage gap) and Q5 (Code of Practice definition — definitional chunks not surfacing). Both are retrieval corpus issues, not synthesis failures. See [`tests/RAGAS_RESULTS.md`](../tests/RAGAS_RESULTS.md) for the full per-query breakdown.

---

## 9. Demo Queries

### Primary demo — cross-domain

**Query:** *"My FIFO worker has a mental health condition and wants to reduce hours — what are my obligations?"*

**Expected behaviour:** `cross_domain = True`, all three retrievers fire, citations from SafeShift (WHS duty of care), FairDesk (NES flexible working entitlements), and HealthNav (mental health employer obligations) in a single synthesised answer.

**Step-by-step:**

1. **Router (Haiku):** Detects `["healthnav", "fairdesk"]` from explicit signals ("mental health condition" → HealthNav, "reduce hours" → FairDesk). FIFO + mental health + safety duties triggers `cross_domain = True`, pulling SafeShift regardless.
2. **Conditional edge:** `Send` fans out to all three domain nodes in parallel.
3. **Retrievers:** Three Qdrant queries execute — each returns top-K chunks filtered by domain. SafeShift returns PCBU psychosocial hazard duties; FairDesk returns NES flexible working entitlements; HealthNav returns mental health reasonable adjustment obligations.
4. **Refusal check:** Scores are well above threshold — synthesis proceeds normally.
5. **Synthesis (Sonnet):** Receives all chunks as a domain-labelled context block. Returns a JSON answer with inline `[doc_id]` citations and an explicit `cross_domain_connection` paragraph.
6. **Output:** Domain badges for all three light up; confidence badge shows `HIGH` or `MEDIUM`; citations table lists sources by domain.

### MCP incident demo

**Query:** *"How many fatigue-related incidents have we had this year, and what are our obligations to manage fatigue risk?"*

**Expected behaviour:** `incident_check_node` fires alongside `safeshift_node` + `healthnav_node`. Answer combines internal incident statistics ("6 fatigue-related incidents recorded in SafeShift records") with document-based obligations (WHS fatigue management duties).

**Step-by-step:**

1. **Router (Haiku):** Detects `["safeshift", "healthnav"]`, `cross_domain = True`. Keyword scan also detects `"how many"` + `"incidents"`.
2. **Conditional edge:** `Send` to `safeshift_node`, `healthnav_node`, and `incident_check_node` — all three run in parallel.
3. **incident_check_node:** Calls MCP server with `get_incident_summary`. Returns summary dict with 6 fatigue incidents in SafeShift category.
4. **Synthesis context:** Domain chunks + `── INCIDENT DATABASE ──` section injected. Sonnet is instructed to reference incident stats as "internal incident records".
5. **Output:** Answer references both the count and the WHS obligations; citations table shows document sources only (incident stats cited in prose).

---

## 10. File Structure

```
workershield-v1/
├── agents/
│   ├── graph.py            # LangGraph StateGraph — WorkerShieldState, all nodes, MCP routing
│   ├── router.py           # Domain classifier — Haiku + keyword fallback
│   ├── retrieval.py        # Domain retriever helpers
│   └── synthesis.py        # Synthesis node — refusal threshold, context builder, Sonnet call
├── corpus/
│   ├── corpus_registry.yaml    # Master registry — 9 documents, metadata, chunk config
│   └── raw/                    # Source PDFs (gitignored)
├── data/
│   ├── incidents.db            # SQLite — 50 synthetic incident records (3 domains)
│   ├── incidents_db.py         # Shared SQLite query helpers
│   ├── incidents_schema.md     # Schema design document
│   └── generate_incidents.py   # Synthetic data generator (seed=42, reproducible)
├── docs/
│   ├── ARCHITECTURE.md         # This document
│   └── CHUNKING_DECISIONS.md   # Per-document strategy rationale
├── ingest/
│   └── load_qdrant.py          # PDF → chunk → embed → Qdrant upsert
├── logs/
│   └── run_log.jsonl           # Per-query JSONL observability log
├── mcp_server/
│   ├── incident_server.py      # FastMCP stdio server — 3 incident query tools
│   └── README.md               # Registration instructions for Claude Code and app
├── observability/
│   └── phoenix_setup.py        # Arize Phoenix OTEL setup
├── prompts/
│   └── PROMPTS.md              # Router and synthesis prompt reference
├── tests/
│   ├── ragas_eval.py           # RAGAS evaluation runner
│   ├── ragas_results.json      # Raw scores (machine-readable)
│   └── RAGAS_RESULTS.md        # Human-readable evaluation results
├── ui/
│   └── app.py                  # Gradio demo interface — 5 example queries
└── utils/
    ├── model_factory.py        # LLMClient + parse_llm_json (JSON healing)
    ├── logger.py               # JSONL run logger
    └── log_reader.py           # Log summary viewer
```

---

## 11. What Is Deliberately Out of Scope for v1

**No ReAct reflection loop.** Retrieval and synthesis are single-pass. After retrieval, the system cannot decide "these chunks aren't good enough — I need to reformulate the query and try again." The refusal threshold is the only post-retrieval escape valve.

**No conversation memory.** Each query is fully stateless. Follow-up questions cannot reference prior answers. This is intentional — in compliance contexts, each answer should be independently reproducible and auditable without relying on session context.

**No input-side guardrails.** Beyond the retrieval confidence threshold, there is no input classification to detect off-topic or harmful queries before retrieval runs. The refusal is a post-retrieval signal, not a pre-retrieval gate.

**Local model stack not exposed in v1 UI.** The codebase supports `model_provider: local` (Ollama Mistral) in `config/model_config.yaml`, and the `ModelFactory` respects this. The v1 Gradio UI runs the Anthropic stack only — the provider switcher was removed to keep the demo focused.

**Single collection, domain partitioning via metadata.** Qdrant stores all documents in one collection (`workershield`) with domain filtered at query time via payload conditions. Per-domain collections would give cleaner separation but add ingest complexity for a 10-document corpus.
