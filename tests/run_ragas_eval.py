"""
RAGAS evaluation pipeline for WorkerShield v1.

Runs all 8 golden-dataset queries through the live WorkerShield graph
(Anthropic stack: Haiku router + Sonnet synthesis), then evaluates
retrieved context and generated answers using four RAGAS metrics judged
by OpenAI gpt-4o-mini + text-embedding-3-small.

Provider split rationale:
  App stack  : Anthropic — production inference, citation quality
  RAGAS judge: OpenAI    — fast, reliable structured evaluation output

Metrics:
  - Faithfulness        — is the answer grounded in the retrieved context?
  - Context Precision   — is retrieved context relevant to the query?
  - Context Recall      — does retrieved context cover the ground truth?
  - Answer Relevancy    — is the answer on-topic for the query?

Usage:
  cd /projects/workershield-v1
  python tests/run_ragas_eval.py
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning)

from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# Guard: OPENAI_API_KEY must be present
# ---------------------------------------------------------------------------

_OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
if not _OPENAI_KEY or _OPENAI_KEY.startswith("sk-placeholder"):
    print(
        "ERROR: OPENAI_API_KEY is missing or is a placeholder in .env.\n"
        "Add your real OpenAI key before running the evaluation.\n"
        "  echo 'OPENAI_API_KEY=sk-...' >> /projects/workershield-v1/.env"
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# RAGAS judge — OpenAI gpt-4o-mini + text-embedding-3-small
# ---------------------------------------------------------------------------

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

judge_llm = LangchainLLMWrapper(
    ChatOpenAI(model="gpt-4o-mini", api_key=_OPENAI_KEY)
)
judge_embeddings = LangchainEmbeddingsWrapper(
    OpenAIEmbeddings(model="text-embedding-3-small", api_key=_OPENAI_KEY)
)

# ---------------------------------------------------------------------------
# RAGAS metrics — pre-initialised singleton instances
# (ragas 0.4.x evaluate() requires isinstance(m, Metric))
# ---------------------------------------------------------------------------

from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
    answer_relevancy,
)
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.run_config import RunConfig

METRICS = [faithfulness, context_precision, context_recall, answer_relevancy]
METRIC_KEYS   = ["faithfulness", "context_precision", "context_recall", "answer_relevancy"]
METRIC_LABELS = {
    "faithfulness":      "Faithfulness",
    "context_precision": "Context Precision",
    "context_recall":    "Context Recall",
    "answer_relevancy":  "Answer Relevancy",
}

# ---------------------------------------------------------------------------
# WorkerShield graph
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from agents.graph import build_graph, WorkerShieldState
from utils.model_factory import get_model_config

# ---------------------------------------------------------------------------
# Golden dataset
# ---------------------------------------------------------------------------

from tests.golden_dataset import GOLDEN_DATASET

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_query(graph, query: str) -> dict:
    """Invoke the WorkerShield graph and return the full result state."""
    initial: WorkerShieldState = {
        "query":            query,
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
    return graph.invoke(initial)


def _extract_answer_text(final_answer_field: str) -> str:
    """
    synthesis_node stores final_answer as a JSON string.
    Extract the plain-text 'answer' field for RAGAS evaluation.
    """
    try:
        parsed = json.loads(final_answer_field)
        return parsed.get("answer", final_answer_field)
    except (json.JSONDecodeError, TypeError):
        return final_answer_field


def _collect_contexts(result: dict) -> list[str]:
    """Flatten all retrieved chunk texts across the three domain lists."""
    all_chunks = (
        result.get("safeshift_chunks", [])
        + result.get("fairdesk_chunks",  [])
        + result.get("healthnav_chunks", [])
    )
    return [c.get("text", "") for c in all_chunks if c.get("text")]


def _score_or_none(result_row, key: str) -> float | None:
    """Safely extract a metric score; return None if missing or NaN."""
    val = result_row.get(key)
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) else round(f, 4)
    except (TypeError, ValueError):
        return None


def _fmt(v: float | None) -> str:
    return f"{v:.4f}" if v is not None else "  N/A "


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation() -> None:
    t_start = time.time()

    cfg            = get_model_config()
    model_provider = cfg.get("model_provider", "unknown")
    provider_cfg   = cfg.get(model_provider, {})
    router_model   = provider_cfg.get("router", "unknown")
    synth_model    = provider_cfg.get("synthesis", "unknown")

    print(f"\n{'='*72}")
    print(f"WorkerShield RAGAS Evaluation")
    print(f"  App provider    : {model_provider}")
    print(f"  Router model    : {router_model}")
    print(f"  Synthesis model : {synth_model}")
    print(f"  Judge LLM       : gpt-4o-mini  (OpenAI)")
    print(f"  Judge embeddings: text-embedding-3-small  (OpenAI)")
    print(f"  Queries         : {len(GOLDEN_DATASET)}")
    print(f"{'='*72}\n")

    graph = build_graph().compile()

    samples:               list[SingleTurnSample] = []
    raw_answers:           list[str]              = []
    raw_contexts:          list[list[str]]        = []
    detected_domains_list: list[list[str]]        = []

    print("Running queries through WorkerShield graph...")
    for i, item in enumerate(GOLDEN_DATASET, 1):
        query = item["query"]
        print(f"  [{i}/{len(GOLDEN_DATASET)}] {query[:70]}{'...' if len(query) > 70 else ''}")

        result   = _run_query(graph, query)
        answer   = _extract_answer_text(result.get("final_answer", ""))
        contexts = _collect_contexts(result)
        domains  = result.get("detected_domains", [])

        raw_answers.append(answer)
        raw_contexts.append(contexts)
        detected_domains_list.append(domains)

        sample = SingleTurnSample(
            user_input         = query,
            response           = answer,
            retrieved_contexts = contexts if contexts else ["(no context retrieved)"],
            reference          = item["ground_truth"],
        )
        samples.append(sample)

    t_graph_done = time.time()
    print(f"  Graph queries complete in {t_graph_done - t_start:.1f}s\n")

    print(f"Running RAGAS evaluation ({len(METRICS)} metrics × {len(samples)} queries)...")

    dataset = EvaluationDataset(samples=samples)
    # Reduced concurrency + extended timeout + retries to prevent GPT-4o-mini
    # rate-limit stalls that produce 0.0 outlier scores and TimeoutError exceptions.
    run_config = RunConfig(timeout=180, max_retries=3, max_workers=2)

    eval_result = evaluate(
        dataset          = dataset,
        metrics          = METRICS,
        llm              = judge_llm,
        embeddings       = judge_embeddings,
        run_config       = run_config,
        show_progress    = True,
        raise_exceptions = False,
    )

    t_eval_done = time.time()

    # ── Extract per-query scores ──────────────────────────────────────────────
    result_df = eval_result.to_pandas()
    per_query: list[dict] = []

    for idx, row in result_df.iterrows():
        scores = {k: _score_or_none(row, k) for k in METRIC_KEYS}
        per_query.append({
            "query_num":        int(idx) + 1,
            "query":            GOLDEN_DATASET[idx]["query"],
            "expected_domains": GOLDEN_DATASET[idx]["expected_domains"],
            "detected_domains": detected_domains_list[idx],
            "expected_doc_ids": GOLDEN_DATASET[idx]["expected_doc_ids"],
            "answer_excerpt":   raw_answers[idx][:200],
            "context_count":    len(raw_contexts[idx]),
            "scores":           scores,
        })

    # ── Aggregate averages ────────────────────────────────────────────────────
    aggregates: dict[str, float | None] = {}
    for key in METRIC_KEYS:
        vals = [q["scores"][key] for q in per_query if q["scores"][key] is not None]
        aggregates[key] = round(sum(vals) / len(vals), 4) if vals else None

    total_time = t_eval_done - t_start

    # ── Print results table ───────────────────────────────────────────────────
    col_w = 18
    header = (
        f"{'Query #':<8}"
        f"{'Faithfulness':>{col_w}}"
        f"{'Ctx Precision':>{col_w}}"
        f"{'Ctx Recall':>{col_w}}"
        f"{'Ans Relevancy':>{col_w}}"
    )
    separator = "─" * len(header)

    print(f"\n{'='*72}")
    print("RAGAS SCORES PER QUERY")
    print(f"{'='*72}")
    print(header)
    print(separator)

    for q in per_query:
        s = q["scores"]
        print(
            f"  Q{q['query_num']:<5}"
            f"{_fmt(s['faithfulness']):>{col_w}}"
            f"{_fmt(s['context_precision']):>{col_w}}"
            f"{_fmt(s['context_recall']):>{col_w}}"
            f"{_fmt(s['answer_relevancy']):>{col_w}}"
        )

    print(separator)
    print(
        f"{'AVERAGE':<8}"
        f"{_fmt(aggregates['faithfulness']):>{col_w}}"
        f"{_fmt(aggregates['context_precision']):>{col_w}}"
        f"{_fmt(aggregates['context_recall']):>{col_w}}"
        f"{_fmt(aggregates['answer_relevancy']):>{col_w}}"
    )
    print(f"{'='*72}\n")

    print("Aggregate scores:")
    for key in METRIC_KEYS:
        label = METRIC_LABELS[key]
        val   = aggregates[key]
        print(f"  {label:<22}: {_fmt(val)}")

    print(f"\nTotal run time: {total_time:.1f}s  "
          f"({(t_graph_done - t_start):.1f}s graph + "
          f"{(t_eval_done - t_graph_done):.1f}s RAGAS eval)")

    retrieval_mode = cfg.get("retrieval_mode", "dense_only")
    reranker_enabled = cfg.get("reranker", {}).get("enabled", False)
    retrieval_label = f"{retrieval_mode}_reranked" if reranker_enabled else retrieval_mode

    # ── Save JSON results ─────────────────────────────────────────────────────
    ts_now = datetime.now(timezone.utc)
    output = {
        "timestamp":          ts_now.isoformat(),
        "model_provider":     model_provider,
        "router_model":       router_model,
        "synthesis_model":    synth_model,
        "retrieval_mode":     retrieval_label,
        "judge_llm":          "gpt-4o-mini (OpenAI)",
        "judge_embeddings":   "text-embedding-3-small (OpenAI)",
        "metrics":            METRIC_KEYS,
        "total_time_seconds": round(total_time, 1),
        "per_query":          per_query,
        "aggregates":         aggregates,
    }

    # Latest (always overwritten)
    results_path = Path(__file__).parent / "ragas_results.json"
    results_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nFull results saved to: {results_path}")

    # Archived history — same date+mode re-runs overwrite (clean re-run semantics)
    date_str      = ts_now.strftime("%Y-%m-%d")
    history_dir   = Path(__file__).parent / "ragas_history"
    history_dir.mkdir(exist_ok=True)
    history_label = f"{retrieval_label}_{date_str}"
    history_path  = history_dir / f"{history_label}.json"
    history_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Archived to        : {history_path}")

    # Append row to COMPARISON.md
    _append_comparison_row(history_dir, date_str, retrieval_label, aggregates)

    # ── Generate RAGAS_RESULTS.md ─────────────────────────────────────────────
    _write_markdown(output, per_query, aggregates, total_time)

    print(f"\nActive app provider : {model_provider}  "
          f"(router={router_model}, synthesis={synth_model})")
    print(f"Retrieval mode     : {retrieval_mode}")
    print(f"RAGAS judge        : gpt-4o-mini + text-embedding-3-small")
    print("Evaluation complete.\n")


# ---------------------------------------------------------------------------
# COMPARISON.md updater
# ---------------------------------------------------------------------------

def _append_comparison_row(
    history_dir: "Path",
    date_str: str,
    retrieval_mode: str,
    aggregates: dict,
) -> None:
    """Upsert one result row in ragas_history/COMPARISON.md.

    If a row for the same date and retrieval_mode already exists it is replaced
    in-place (clean re-run semantics). Otherwise the row is appended.
    """
    comp_path = history_dir / "COMPARISON.md"

    def _f2(v):
        return f"{v:.4f}" if v is not None else "N/A"

    row = (
        f"| {date_str} | {retrieval_mode} "
        f"| {_f2(aggregates.get('faithfulness'))} "
        f"| {_f2(aggregates.get('context_precision'))} "
        f"| {_f2(aggregates.get('context_recall'))} "
        f"| {_f2(aggregates.get('answer_relevancy'))} |"
    )

    match_prefix = f"| {date_str} | {retrieval_mode} "

    if comp_path.exists():
        lines   = comp_path.read_text().rstrip().split("\n")
        updated = False
        new_lines = []
        for line in lines:
            if line.startswith(match_prefix):
                new_lines.append(row)
                updated = True
            else:
                new_lines.append(line)
        if not updated:
            new_lines.append(row)
        comp_path.write_text("\n".join(new_lines) + "\n")
        action = "updated" if updated else "appended"
    else:
        header = (
            "# WorkerShield RAGAS Retrieval Comparison\n\n"
            "Results across retrieval configurations. Each row is one eval run.\n\n"
            "| Date | Config | Faithfulness | Context Precision | Context Recall | Answer Relevancy |\n"
            "|---|---|---|---|---|---|\n"
        )
        comp_path.write_text(header + row + "\n")
        action = "created"

    print(f"COMPARISON.md      : row {action} ({retrieval_mode}  {date_str})")


# ---------------------------------------------------------------------------
# Markdown report writer
# ---------------------------------------------------------------------------

def _write_markdown(output: dict, per_query: list, aggregates: dict, total_time: float) -> None:
    def _fmtmd(v: float | None) -> str:
        return f"{v:.4f}" if v is not None else "N/A"

    def _flag(v: float | None) -> str:
        if v is None:
            return " ⚠"
        return " ← low" if v < 0.5 else ""

    ts      = output["timestamp"]
    prov    = output["model_provider"]
    rmod    = output["router_model"]
    smod    = output["synthesis_model"]
    ret_mode = output.get("retrieval_mode", "dense_only")

    rows = []
    for q in per_query:
        s = q["scores"]
        rows.append(
            f"| Q{q['query_num']} | "
            f"{_fmtmd(s['faithfulness'])}{_flag(s['faithfulness'])} | "
            f"{_fmtmd(s['context_precision'])}{_flag(s['context_precision'])} | "
            f"{_fmtmd(s['context_recall'])}{_flag(s['context_recall'])} | "
            f"{_fmtmd(s['answer_relevancy'])}{_flag(s['answer_relevancy'])} |"
        )

    agg_row = (
        f"| **AVG** | "
        f"**{_fmtmd(aggregates['faithfulness'])}** | "
        f"**{_fmtmd(aggregates['context_precision'])}** | "
        f"**{_fmtmd(aggregates['context_recall'])}** | "
        f"**{_fmtmd(aggregates['answer_relevancy'])}** |"
    )

    query_list = "\n".join(
        f"{i+1}. {q['query']}" for i, q in enumerate(per_query)
    )

    md = f"""# WorkerShield RAGAS Evaluation Results

**Run date:** {ts}
**Total run time:** {total_time:.1f}s

## Provider Configuration

| Role | Provider | Model |
|---|---|---|
| App router | `{prov}` | `{rmod}` |
| App synthesis | `{prov}` | `{smod}` |
| Retrieval mode | — | `{ret_mode}` |
| RAGAS judge LLM | OpenAI | `gpt-4o-mini` |
| RAGAS judge embeddings | OpenAI | `text-embedding-3-small` |

**Rationale for provider split:** The application stack uses Anthropic for high-quality cited
synthesis. The RAGAS evaluation judge uses OpenAI GPT-4o-mini, which provides fast, reliable
structured output — essential for metrics that require the judge to follow strict JSON schemas.
This separation also keeps evaluation inference independent of the production stack,
allowing either to be changed without affecting the other.

---

## Methodology

### Golden Dataset

8 hand-crafted queries spanning all three WorkerShield domains:

{query_list}

Each query has:
- A **ground truth** answer representing the ideal response
- **Expected document IDs** that should appear in retrieved context
- **Expected domain(s)** the router should detect

### Metrics

| Metric | What it measures |
|---|---|
| **Faithfulness** | Is every claim in the answer supported by the retrieved context? Values near 1.0 mean no hallucinations relative to context. |
| **Context Precision** | Are the retrieved chunks relevant to the query? High precision means the retriever is not pulling in noise. |
| **Context Recall** | Does the retrieved context cover the ground truth answer? Low recall means relevant documents were not retrieved. |
| **Answer Relevancy** | Is the generated answer on-topic for the query? Measures whether the answer addresses what was asked. |

Scores range from 0.0 (worst) to 1.0 (best).

---

## Results

| Query | Faithfulness | Context Precision | Context Recall | Answer Relevancy |
|---|---|---|---|---|
{chr(10).join(rows)}
| **---** | **---** | **---** | **---** | **---** |
{agg_row}

---

## Interpretation Notes

- **Faithfulness ≥ 0.80** is the target for a RAG system used in compliance contexts — low faithfulness means the synthesiser is drifting from retrieved evidence.
- **Context Recall < 0.60** suggests the Qdrant retriever is missing relevant document chunks for that query; consider index tuning or increasing `TOP_K`.
- **Context Precision < 0.60** means too many irrelevant chunks are being retrieved; domain filtering may need tightening.
- **Answer Relevancy < 0.70** often indicates the synthesis prompt is over-generalising beyond the question scope.
- Scores marked **← low** are below 0.5 and warrant investigation.
- **⚠** scores could not be computed for that sample.

---

*Generated by `tests/run_ragas_eval.py` · WorkerShield v1*
"""

    md_path = Path(__file__).parent / "RAGAS_RESULTS.md"
    md_path.write_text(md)
    print(f"Markdown summary saved to: {md_path}")


if __name__ == "__main__":
    run_evaluation()
