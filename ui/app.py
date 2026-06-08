"""
WorkerShield Gradio demo interface.

Compiles the LangGraph graph once at startup, then invokes it on each query.
Displays the synthesised markdown answer, a citations table, domain badges,
and a confidence indicator.
"""

from __future__ import annotations

import json

from dotenv import load_dotenv

load_dotenv("/projects/workershield-v1/.env")

import gradio as gr  # noqa: E402 — must come after load_dotenv

from agents.graph import build_graph  # noqa: E402

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

_DOMAIN_KEYS = ["safeshift", "fairdesk", "healthnav"]

_CONFIDENCE_STYLES = {
    "high":   ("HIGH",   "#166534", "#DCFCE7"),
    "medium": ("MEDIUM", "#92400E", "#FEF3C7"),
    "low":    ("LOW",    "#991B1B", "#FEE2E2"),
}

# ---------------------------------------------------------------------------
# Example queries
# ---------------------------------------------------------------------------

EXAMPLES = [
    "My FIFO worker has a mental health condition and wants to reduce hours — what are my obligations?",
    "What are the rules around casual employee overtime on public holidays?",
    "What psychosocial hazards must I manage under WHS law?",
    "What are my obligations when a worker is injured and needs to return to work?",
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


# ---------------------------------------------------------------------------
# Graph invocation
# ---------------------------------------------------------------------------

def _run_query(query: str):
    """Invoke the compiled graph and return UI component updates."""
    if not query or not query.strip():
        yield (
            "",
            gr.HTML(_badge_html([])),
            gr.HTML(_confidence_html("")),
            gr.HTML("<p style='color:#94A3B8;'>Submit a query to see sources.</p>"),
        )
        return

    # Yield a "thinking" state immediately
    yield (
        "*Searching compliance documents…*",
        gr.HTML(_badge_html([])),
        gr.HTML(_confidence_html("")),
        gr.HTML("<p style='color:#94A3B8;'>Retrieving sources…</p>"),
    )

    try:
        result = graph.invoke({"query": query.strip()})
    except Exception as exc:
        yield (
            f"**Error:** {exc}",
            gr.HTML(_badge_html([])),
            gr.HTML(_confidence_html("low")),
            gr.HTML("<p style='color:#EF4444;'>Graph invocation failed.</p>"),
        )
        return

    # Parse final_answer (it's a JSON string containing the answer markdown)
    raw_final = result.get("final_answer", "")
    try:
        parsed = json.loads(raw_final)
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

    # ── Event wiring ────────────────────────────────────────────────────────

    outputs = [answer_out, badge_html, confidence_html, citations_html]

    submit_btn.click(
        fn=_run_query,
        inputs=query_box,
        outputs=outputs,
    )

    query_box.submit(
        fn=_run_query,
        inputs=query_box,
        outputs=outputs,
    )

    clear_btn.click(
        fn=lambda: ("", _badge_html([]), _confidence_html(""),
                    "<p style='color:#94A3B8;'>Submit a query to see sources.</p>"),
        outputs=outputs,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True, css=_CSS)
