"""
WorkerShield Gradio demo interface.

Compiles the LangGraph graph once at startup, then invokes it on each query.
Displays the synthesised markdown answer, a citations table, domain badges,
a confidence indicator, and the active model stack summary.
"""

from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv("/projects/workershield-v1/.env")

import gradio as gr  # noqa: E402 — must come after load_dotenv

from agents.graph import build_graph  # noqa: E402
from utils.model_factory import ModelFactory  # noqa: E402

# ---------------------------------------------------------------------------
# Compile graph once at startup
# ---------------------------------------------------------------------------

graph = build_graph().compile()

# ---------------------------------------------------------------------------
# Domain badge config
# ---------------------------------------------------------------------------

_DOMAIN_COLOURS = {
    "safeshift": ("#1B3A6B", "#DBEAFE", "SafeShift"),
    "fairdesk":  ("#0F766E", "#CCFBF1", "FairDesk"),
    "healthnav": ("#B45309", "#FEF3C7", "HealthNav"),
}

_CONFIDENCE_STYLES = {
    "high":         ("HIGH",        "#166534", "#DCFCE7"),
    "medium":       ("MEDIUM",      "#92400E", "#FEF3C7"),
    "low":          ("LOW",         "#991B1B", "#FEE2E2"),
    "insufficient": ("OUT OF SCOPE","#374151", "#F3F4F6"),
}

# ---------------------------------------------------------------------------
# RAGAS evaluation data — loaded once at startup
# ---------------------------------------------------------------------------

_RAGAS_PATH = Path(__file__).parent.parent / "tests" / "ragas_results.json"

_RAGAS_TARGETS = {
    "faithfulness":      ("Faithfulness",      0.85),
    "context_precision": ("Context Precision", 0.70),
    "context_recall":    ("Context Recall",    0.70),
    "answer_relevancy":  ("Answer Relevancy",  0.80),
}

def _load_ragas_data() -> dict:
    try:
        return json.loads(_RAGAS_PATH.read_text())
    except Exception:
        return {}

_RAGAS_DATA = _load_ragas_data()

# ---------------------------------------------------------------------------
# Example queries
# ---------------------------------------------------------------------------

EXAMPLES = [
    "My FIFO worker has a mental health condition and wants to reduce hours — what are my obligations?",
    "What are the rules around casual employee overtime on public holidays?",
    "What psychosocial hazards must I manage under WHS law?",
    "What are my obligations when a worker is injured and needs to return to work?",
    "How many fatigue-related incidents have we had this year, and what are our obligations to manage fatigue risk?",
]

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
/* ── page chrome ─────────────────────────────────────────────────────────── */
.gradio-container { max-width: 900px !important; margin: 0 auto; }

/* ── header ──────────────────────────────────────────────────────────────── */
#ws-header { text-align: center; padding: 24px 0 8px; }
#ws-title  { font-size: 2rem; font-weight: 700; color: #1B3A6B; margin: 0; }
#ws-sub    { font-size: 1rem; color: #64748B; margin: 4px 0 0; }

/* ── domain badges ───────────────────────────────────────────────────────── */
.ws-badge-row { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
.ws-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    opacity: 0.25;
    border: 2px solid transparent;
    transition: opacity 0.2s, border-color 0.2s;
}
.ws-badge.active { opacity: 1; }

/* SafeShift */
.badge-safeshift         { background:#DBEAFE; color:#1B3A6B; border-color:#DBEAFE; }
.badge-safeshift.active  { border-color:#1B3A6B; }

/* FairDesk */
.badge-fairdesk          { background:#CCFBF1; color:#0F766E; border-color:#CCFBF1; }
.badge-fairdesk.active   { border-color:#0F766E; }

/* HealthNav */
.badge-healthnav         { background:#FEF3C7; color:#B45309; border-color:#FEF3C7; }
.badge-healthnav.active  { border-color:#B45309; }

/* ── confidence pill ─────────────────────────────────────────────────────── */
.ws-confidence-label { font-size:0.75rem; color:#64748B; font-weight:500; }
.ws-confidence {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.05em;
}

/* ── answer panel ────────────────────────────────────────────────────────── */
#answer-panel { min-height: 120px; }
#answer-panel .prose p { line-height: 1.7; }

/* ── citations table ─────────────────────────────────────────────────────── */
#citations-table table { width:100%; border-collapse:collapse; font-size:0.82rem; }
#citations-table th    { background:#F1F5F9; font-weight:600; padding:6px 10px;
                          text-align:left; border-bottom:1px solid #E2E8F0; }
#citations-table td    { padding:5px 10px; vertical-align:top;
                          border-bottom:1px solid #F1F5F9; }
#citations-table tr:last-child td { border-bottom:none; }

/* ── model status table ──────────────────────────────────────────────────── */
#model-status table { width:100%; border-collapse:collapse; font-size:0.82rem; }
#model-status th    { background:#F1F5F9; font-weight:600; padding:6px 10px;
                       text-align:left; border-bottom:1px solid #E2E8F0; }
#model-status td    { padding:5px 10px; border-bottom:1px solid #F1F5F9; }
#model-status tr:last-child td { border-bottom:none; }

/* ── RAGAS evaluation table ──────────────────────────────────────────────── */
#ragas-table table { width:100%; border-collapse:collapse; font-size:0.82rem; }
#ragas-table th    { background:#F1F5F9; font-weight:600; padding:6px 10px;
                     text-align:left; border-bottom:1px solid #E2E8F0; }
#ragas-table td    { padding:5px 10px; border-bottom:1px solid #F1F5F9; }
#ragas-table tr:last-child td { border-bottom:none; }

/* ── example buttons ─────────────────────────────────────────────────────── */
.ws-example { font-size:0.82rem !important; }

/* ── submit button ───────────────────────────────────────────────────────── */
#submit-btn { background:#1B3A6B !important; color:#fff !important; }
#submit-btn:hover { background:#16305A !important; }
"""

# ---------------------------------------------------------------------------
# Helpers — HTML renderers
# ---------------------------------------------------------------------------

def _badge_html(active_domains: list[str]) -> str:
    parts = ['<div class="ws-badge-row">']
    for key, (fg, bg, label) in _DOMAIN_COLOURS.items():
        active_cls = " active" if key in active_domains else ""
        parts.append(
            f'<span class="ws-badge badge-{key}{active_cls}">{label}</span>'
        )
    parts.append("</div>")
    return "".join(parts)


def _confidence_html(level: str) -> str:
    label, fg, bg = _CONFIDENCE_STYLES.get(level, ("—", "#64748B", "#F1F5F9"))
    return (
        '<span class="ws-confidence-label">Confidence&nbsp;</span>'
        f'<span class="ws-confidence" style="background:{bg};color:{fg};">'
        f"{label}</span>"
    )


def _citations_html(citations: list[dict]) -> str:
    if not citations:
        return "<p style='color:#94A3B8;font-size:0.85rem;'>No citations available.</p>"
    rows = []
    for c in citations:
        domain_key = c.get("domain", "")
        _, bg, label = _DOMAIN_COLOURS.get(domain_key, ("#64748B", "#F8FAFC", domain_key))
        domain_chip = (
            f'<span style="background:{bg};padding:2px 8px;border-radius:6px;'
            f'font-size:0.75rem;font-weight:600;">{label}</span>'
        )
        doc   = c.get("doc_title", c.get("doc_id", ""))
        sec   = c.get("section", "") or "—"
        excpt = c.get("excerpt", "")
        if len(excpt) > 120:
            excpt = excpt[:120] + "…"
        rows.append(
            f"<tr><td>{domain_chip}</td>"
            f"<td><strong>[{c.get('doc_id','')}]</strong> {doc}</td>"
            f"<td>{sec}</td>"
            f"<td style='color:#475569;font-style:italic;'>{excpt}</td></tr>"
        )
    header = (
        "<table><thead><tr>"
        "<th>Domain</th><th>Document</th><th>Section</th><th>Excerpt</th>"
        "</tr></thead><tbody>"
    )
    return f'<div id="citations-table">{header}{"".join(rows)}</tbody></table></div>'


def _ragas_html() -> str:
    """Render the static RAGAS scores table + Phoenix tracing link."""
    agg = _RAGAS_DATA.get("aggregates", {})
    rows = []
    for key, (label, target) in _RAGAS_TARGETS.items():
        val = agg.get(key)
        if val is None:
            score_str   = "N/A"
            status_html = (
                '<span style="background:#F1F5F9;color:#64748B;padding:2px 8px;'
                'border-radius:6px;font-size:0.75rem;">N/A</span>'
            )
        else:
            score_str = f"{val:.2f}"
            if val >= target:
                status_html = (
                    '<span style="background:#DCFCE7;color:#166534;padding:3px 10px;'
                    'border-radius:6px;font-size:0.78rem;font-weight:700;">✅ Pass</span>'
                )
            else:
                status_html = (
                    '<span style="background:#FEF3C7;color:#92400E;padding:3px 10px;'
                    'border-radius:6px;font-size:0.78rem;font-weight:700;">⚠️ Low</span>'
                )
        rows.append(
            f"<tr>"
            f"<td style='font-weight:500;'>{label}</td>"
            f"<td style='font-family:monospace;font-size:0.88rem;'>{score_str}</td>"
            f"<td style='color:#64748B;'>&gt;{target:.2f}</td>"
            f"<td>{status_html}</td>"
            f"</tr>"
        )

    table = (
        '<div id="ragas-table">'
        "<table><thead><tr>"
        "<th>Metric</th><th>Score</th><th>Target</th><th>Status</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )

    phoenix_block = (
        '<div style="margin-top:16px;padding:12px 14px;background:#F8FAFC;'
        'border-radius:8px;border:1px solid #E2E8F0;">'
        '<p style="margin:0 0 6px;font-weight:600;font-size:0.85rem;color:#1B3A6B;">'
        'Live Tracing</p>'
        '<a href="http://192.168.100.10:6006" target="_blank" '
        'style="color:#0F766E;font-weight:600;font-size:0.88rem;text-decoration:none;">'
        'View live trace for this query in Phoenix →</a>'
        '<p style="margin:8px 0 0;font-size:0.78rem;color:#64748B;">'
        'Every query generates traces showing router decisions, retrieval scores '
        'per domain, and synthesis token usage.'
        '</p>'
        '</div>'
    )

    return table + phoenix_block


def _methodology_html() -> str:
    """Render the static evaluation methodology note."""
    judge    = _RAGAS_DATA.get("judge_llm", "gpt-4o-mini (OpenAI)")
    ts       = _RAGAS_DATA.get("timestamp", "")
    ts_label = ts[:10] if ts else "—"
    return (
        '<div style="font-size:0.85rem;line-height:1.75;color:#475569;">'
        '<p style="margin:0 0 6px;">Scores generated using the '
        '<strong>RAGAS framework</strong> against an 8-query golden dataset '
        'covering all three domains (SafeShift · FairDesk · HealthNav).</p>'
        f'<p style="margin:0 0 6px;"><strong>Judge model:</strong> {judge} &nbsp;·&nbsp; '
        f'<strong>Run date:</strong> {ts_label}</p>'
        '<p style="margin:0;"><strong>Application stack:</strong> '
        'Claude Haiku (routing) + Claude Sonnet (synthesis). '
        'Judge kept on a separate OpenAI stack so evaluation inference is '
        'independent of the production stack.</p>'
        '</div>'
    )


def _model_status_html(
    detected_domains: list[str] | None = None,
    confidence: str = "",
) -> str:
    """Render a compact summary table showing the active model stack and run metadata."""
    mf              = ModelFactory()
    provider        = mf.active_provider()
    router_model    = mf.router_model_name()
    synthesis_model = mf.synthesis_model_name()
    stack_label     = provider.capitalize()

    domains_str = ", ".join(detected_domains) if detected_domains else "—"
    conf_str    = confidence.upper() if confidence else "—"

    rows = [
        ("Stack",           stack_label,     ""),
        ("Router model",    router_model,    provider),
        ("Synthesis model", synthesis_model, provider),
        ("Domains hit",     domains_str,     ""),
        ("Confidence",      conf_str,        ""),
    ]

    html_rows = "".join(
        f"<tr><td style='color:#64748B;'>{label}</td>"
        f"<td><strong>{value}</strong></td>"
        f"<td style='color:#94A3B8;font-size:0.78rem;'>{extra}</td></tr>"
        for label, value, extra in rows
    )

    return (
        '<div id="model-status">'
        "<table><thead><tr>"
        "<th>Parameter</th><th>Value</th><th>Provider</th>"
        "</tr></thead><tbody>"
        f"{html_rows}"
        "</tbody></table></div>"
    )


# ---------------------------------------------------------------------------
# Graph invocation
# ---------------------------------------------------------------------------

def _run_query(query: str):
    """Invoke the compiled graph and yield UI component updates."""
    if not query or not query.strip():
        yield (
            "",
            gr.HTML(_badge_html([])),
            gr.HTML(_confidence_html("")),
            gr.HTML("<p style='color:#94A3B8;'>Submit a query to see sources.</p>"),
            gr.HTML(_model_status_html()),
        )
        return

    # Yield a "thinking" state immediately
    yield (
        "*Searching compliance documents…*",
        gr.HTML(_badge_html([])),
        gr.HTML(_confidence_html("")),
        gr.HTML("<p style='color:#94A3B8;'>Retrieving sources…</p>"),
        gr.HTML(_model_status_html()),
    )

    try:
        result = graph.invoke({"query": query.strip()})
    except Exception as exc:
        yield (
            f"**Error:** {exc}",
            gr.HTML(_badge_html([])),
            gr.HTML(_confidence_html("low")),
            gr.HTML("<p style='color:#EF4444;'>Graph invocation failed.</p>"),
            gr.HTML(_model_status_html()),
        )
        return

    # Parse final_answer (JSON string containing the answer markdown)
    raw_final = result.get("final_answer", "")
    try:
        parsed    = json.loads(raw_final)
        answer_md = parsed.get("answer", raw_final)
    except (json.JSONDecodeError, TypeError):
        answer_md = raw_final

    citations        = result.get("citations", [])
    confidence       = result.get("confidence", "")
    detected_domains = result.get("detected_domains", [])

    yield (
        answer_md,
        gr.HTML(_badge_html(detected_domains)),
        gr.HTML(_confidence_html(confidence)),
        gr.HTML(_citations_html(citations)),
        gr.HTML(_model_status_html(detected_domains, confidence)),
    )


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="WorkerShield") as demo:

    # Header
    gr.HTML(
        '<div id="ws-header">'
        '<p id="ws-title">WorkerShield</p>'
        '<p id="ws-sub">Australian Workplace Compliance Assistant</p>'
        '<p style="font-size:0.82rem;color:#94A3B8;margin:6px 0 0;">'
        'Powered by Claude Haiku (routing) + Claude Sonnet (synthesis)'
        '</p>'
        "</div>"
    )

    with gr.Row():
        with gr.Column(scale=1):

            query_box = gr.Textbox(
                label="Ask a workplace compliance question",
                placeholder="e.g. What are my WHS obligations for remote workers?",
                lines=2,
                elem_id="query-box",
            )

            with gr.Row():
                submit_btn = gr.Button("Ask WorkerShield", variant="primary", elem_id="submit-btn")
                clear_btn  = gr.Button("Clear", variant="secondary")

            # Example queries
            gr.Markdown("**Try an example:**", elem_classes=["ws-example"])
            for ex in EXAMPLES:
                btn = gr.Button(ex, size="sm", elem_classes=["ws-example"])
                btn.click(fn=lambda q=ex: q, outputs=query_box)

            gr.Markdown("---")

            # Domain badges
            gr.Markdown("**Agent Route**", elem_classes=["ws-example"])
            badge_html = gr.HTML(_badge_html([]))

            # Confidence
            gr.Markdown("**Confidence**", elem_classes=["ws-example"])
            confidence_html = gr.HTML(_confidence_html(""))

    with gr.Row():
        with gr.Column():
            answer_out = gr.Markdown(
                value="",
                label="Answer",
                elem_id="answer-panel",
            )

            with gr.Accordion("Sources", open=False):
                citations_html = gr.HTML(
                    "<p style='color:#94A3B8;'>Submit a query to see sources.</p>"
                )

            with gr.Accordion("Run summary", open=False):
                model_status_html = gr.HTML(_model_status_html())

            with gr.Accordion("Evaluation & Observability", open=False):
                gr.HTML(_ragas_html())

            with gr.Accordion("About this evaluation", open=False):
                gr.HTML(_methodology_html())

    # ── Event wiring ────────────────────────────────────────────────────────

    outputs = [answer_out, badge_html, confidence_html, citations_html, model_status_html]

    submit_btn.click(
        fn=_run_query,
        inputs=[query_box],
        outputs=outputs,
    )

    query_box.submit(
        fn=_run_query,
        inputs=[query_box],
        outputs=outputs,
    )

    clear_btn.click(
        fn=lambda: (
            "",
            _badge_html([]),
            _confidence_html(""),
            "<p style='color:#94A3B8;'>Submit a query to see sources.</p>",
            _model_status_html(),
        ),
        inputs=[],
        outputs=outputs,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True, css=_CSS)
