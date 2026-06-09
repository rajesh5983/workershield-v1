"""
WorkerShield domain router — first node in the LangGraph graph.

Primary:  LLM classification via ModelFactory (local Ollama or Anthropic API).
Fallback: keyword matching when the LLM call fails.

Returns a partial WorkerShieldState dict:
  detected_domains: list[str]  — subset of {safeshift, fairdesk, healthnav}
  cross_domain:     bool        — True when 2+ domains detected
"""

from __future__ import annotations

import json
import logging
from typing import Any

from dotenv import load_dotenv

from utils.model_factory import ModelFactory

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOMAINS = ("safeshift", "fairdesk", "healthnav")

_SYSTEM_PROMPT = """You are a workplace compliance query classifier.
Classify the query into one or more of these domains:
- safeshift: WHS legislation, workplace safety, hazardous tasks, work environment, incident reporting
- fairdesk: Fair Work, employment conditions, casual work, NES entitlements, flexible working, overtime
- healthnav: mental health, fatigue, psychological safety, workers compensation, return to work, occupational health

Return ONLY valid JSON: {"detected_domains": [...], "cross_domain": bool}"""

# Keyword fallback — lists are intentionally broad to maximise recall
_KEYWORDS: dict[str, list[str]] = {
    "safeshift": [
        "whs", "work health and safety", "ohs", "occupational health and safety",
        "hazard", "hazardous", "incident", "near miss", "risk assessment",
        "pcbu", "duty of care", "safe work", "worksafe", "code of practice",
        "ppe", "personal protective equipment", "manual task", "musculoskeletal",
        "work environment", "workplace safety", "safety duty", "safety obligation",
        "dangerous", "notifiable incident", "inspector", "improvement notice",
        "prohibition notice", "penalty", "prosecution", "whs act",
    ],
    "fairdesk": [
        "fair work", "nes", "national employment standards", "award",
        "enterprise agreement", "casual", "overtime", "penalty rate",
        "public holiday", "annual leave", "sick leave", "carer leave",
        "parental leave", "flexible working", "unfair dismissal", "redundancy",
        "notice of termination", "pay slip", "minimum wage", "superannuation",
        "super", "entitlement", "employment contract", "permanent employee",
        "part-time", "full-time", "rostering", "shift", "hours of work",
        "fwo", "fair work ombudsman", "fair work commission",
        "reduce hours", "reduced hours", "change hours", "part time request",
    ],
    "healthnav": [
        "mental health", "psychological", "psychosocial", "wellbeing",
        "fatigue", "burnout", "stress", "anxiety", "depression",
        "workers compensation", "workcover", "return to work", "rtw",
        "injury", "rehabilitation", "incapacity", "permanent impairment",
        "occupational health", "eap", "employee assistance",
        "reasonable adjustment", "disability", "bullying", "harassment",
        "modified duties", "fit for work",
    ],
}


# ---------------------------------------------------------------------------
# LLM classifier
# ---------------------------------------------------------------------------

def _classify_with_llm(query: str, model_id: str | None = None) -> dict[str, Any]:
    """Classify via the configured LLM. Raises on any failure."""
    llm = ModelFactory().get_router_llm(model_id)
    logger.debug("Router using provider=%s model=%s", llm.provider, llm.model)
    raw = llm.chat(_SYSTEM_PROMPT, query).strip()

    # Strip markdown fences if the model wraps the JSON
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    result = json.loads(raw)

    # Normalise: keep only known domains, recompute cross_domain
    domains = [d for d in result.get("detected_domains", []) if d in DOMAINS]
    if not domains:
        domains = ["safeshift"]  # safe default — shouldn't happen in practice
    return {
        "detected_domains": domains,
        "cross_domain": len(domains) >= 2,
    }


# ---------------------------------------------------------------------------
# Keyword fallback
# ---------------------------------------------------------------------------

def _classify_with_keywords(query: str) -> dict[str, Any]:
    """Simple keyword scan — used when the API is unavailable."""
    q = query.lower()
    domains = [
        domain
        for domain, keywords in _KEYWORDS.items()
        if any(kw in q for kw in keywords)
    ]
    if not domains:
        # No match — default to all three so retrieval casts a wide net
        logger.warning("No keyword match for query; defaulting to all domains.")
        domains = list(DOMAINS)
    return {
        "detected_domains": domains,
        "cross_domain": len(domains) >= 2,
    }


# ---------------------------------------------------------------------------
# Public interface — LangGraph node
# ---------------------------------------------------------------------------

def router_node(state: dict[str, Any]) -> dict[str, Any]:
    """LangGraph node: classify query → detected_domains + cross_domain."""
    query    = state["query"]
    model_id = state.get("router_model_id")  # injected by UI; None → config default
    try:
        result = _classify_with_llm(query, model_id)
        logger.debug("LLM classification: %s", result)
    except Exception as exc:
        logger.warning("LLM classification failed (%s); using keyword fallback.", exc)
        result = _classify_with_keywords(query)
        logger.debug("Keyword classification: %s", result)
    return result


# ---------------------------------------------------------------------------
# Standalone unit test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    test_cases = [
        ("fatigue risk for FIFO workers",
         {"cross_domain": False, "domains": {"safeshift", "healthnav"}}),
        ("casual employee overtime on public holidays",
         {"cross_domain": False, "domains": {"fairdesk"}}),
        ("mental health support obligations for employers",
         {"cross_domain": False, "domains": {"healthnav"}}),
        ("FIFO worker wants to reduce hours due to mental health condition",
         {"cross_domain": True,  "domains": None}),  # must be multi-domain
        ("incident reporting requirements under WHS Act",
         {"cross_domain": False, "domains": {"safeshift"}}),
        ("national dietary guidelines for workplace wellness",
         {"cross_domain": False, "domains": None}),  # any domain acceptable
    ]

    print(f"\n{'='*70}")
    print(f"{'#':<3} {'Query':<50} {'Domains':<28} {'XD':<5} Result")
    print(f"{'='*70}")

    passes = 0
    for i, (query, expected) in enumerate(test_cases, 1):
        result    = router_node({"query": query})
        domains   = result["detected_domains"]
        cross     = result["cross_domain"]

        # Evaluate
        xd_ok     = (cross == expected["cross_domain"])
        domain_ok = (
            expected["domains"] is None
            or bool(set(domains) & expected["domains"])
        )
        ok = xd_ok and domain_ok
        passes += ok

        tag   = "PASS" if ok else "FAIL"
        label = f"{', '.join(domains)}"
        note  = ""
        if not xd_ok:
            note = f" [cross_domain expected {expected['cross_domain']}, got {cross}]"
        if not domain_ok:
            note += f" [expected one of {expected['domains']}, got {set(domains)}]"

        print(f"{i:<3} {query[:48]:<50} {label:<28} {'T' if cross else 'F':<5} {tag}{note}")

    print(f"{'='*70}")
    print(f"Result: {passes}/{len(test_cases)} passed\n")
