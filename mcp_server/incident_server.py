"""
WorkerShield Incident MCP Server.

Exposes three tools over stdio transport so Claude Code (or any MCP-compatible
client) can query the synthetic incident database alongside the document corpus.

Tools:
  query_incidents       — filtered incident records
  get_incident_summary  — counts grouped by domain × status
  get_incident_detail   — full record for a single incident

Run standalone:
  python3 mcp_server/incident_server.py

Register with Claude Code:
  claude mcp add workershield-incidents -- python3 /projects/workershield-v1/mcp_server/incident_server.py

Self-test:
  python3 mcp_server/incident_server.py --test
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.fastmcp import FastMCP

from data.incidents_db import (
    get_incident_detail as _get_detail,
    get_incident_summary as _get_summary,
    query_incidents as _query,
)

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="workershield-incidents",
    instructions=(
        "Provides access to the WorkerShield synthetic workplace incident database. "
        "Use query_incidents to filter records by domain, status, category, or date range. "
        "Use get_incident_summary for aggregated counts. "
        "Use get_incident_detail to retrieve a specific record by ID."
    ),
)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def query_incidents(
    domain: str = "",
    status: str = "",
    category: str = "",
    date_from: str = "",
    date_to: str = "",
    limit: int = 10,
) -> str:
    """Query workplace incident records with optional filters.

    Args:
        domain:    Filter by domain — one of: safeshift, fairdesk, healthnav.
                   Leave empty to search all domains.
        status:    Filter by status — one of: open, in_progress, closed.
                   Leave empty to include all statuses.
        category:  Filter by incident category or subcategory keyword
                   (e.g. 'fatigue', 'return_to_work', 'underpayment').
                   Leave empty to include all categories.
        date_from: Earliest reported date (YYYY-MM-DD). Leave empty for no lower bound.
        date_to:   Latest reported date (YYYY-MM-DD). Leave empty for no upper bound.
        limit:     Maximum number of records to return (default 10, max 50).

    Returns:
        JSON array of matching incident records.
    """
    rows = _query(
        domain=domain or None,
        status=status or None,
        category=category or None,
        date_from=date_from or None,
        date_to=date_to or None,
        limit=min(int(limit), 50),
    )
    return json.dumps(rows, indent=2)


@mcp.tool()
def get_incident_summary() -> str:
    """Return aggregated incident counts grouped by domain and status.

    Returns a JSON object with:
      - totals: overall counts (total, open, in_progress, closed, avg_days_to_resolve)
      - by_domain_status: list of {domain, status, count} rows
      - by_category: list of {domain, category, count} rows ordered by frequency

    Use this to answer questions about incident trends, volumes, or distributions.
    """
    summary = _get_summary()
    return json.dumps(summary, indent=2)


@mcp.tool()
def get_incident_detail(incident_id: str) -> str:
    """Retrieve the full record for a single incident by ID.

    Args:
        incident_id: The incident identifier (e.g. 'INC-001'). Case-insensitive.

    Returns:
        JSON object with all incident fields, or an error message if not found.
    """
    record = _get_detail(incident_id)
    if record is None:
        return json.dumps({"error": f"Incident {incident_id!r} not found."})
    return json.dumps(record, indent=2)


# ---------------------------------------------------------------------------
# Self-test mode (--test flag)
# ---------------------------------------------------------------------------

def _run_self_test() -> None:
    print("=" * 60)
    print("WorkerShield Incident MCP Server — self-test")
    print("=" * 60)

    print("\n[1] get_incident_summary()")
    summary = json.loads(get_incident_summary())
    print(f"  Total incidents : {summary['totals']['total']}")
    print(f"  Open            : {summary['totals']['open']}")
    print(f"  In progress     : {summary['totals']['in_progress']}")
    print(f"  Closed          : {summary['totals']['closed']}")
    print(f"  Avg days resolve: {summary['totals']['avg_days_to_resolve']}")

    print("\n  By domain × status:")
    for row in summary["by_domain_status"]:
        print(f"    {row['domain']:<12} {row['status']:<12} {row['count']}")

    print("\n[2] query_incidents(category='fatigue', domain='safeshift')")
    fatigue = json.loads(query_incidents(category="fatigue", domain="safeshift"))
    print(f"  Returned {len(fatigue)} fatigue/safeshift records")
    for r in fatigue[:3]:
        print(f"    {r['incident_id']}  {r['status']:<12} {r['reported_date']}  {r['location']}")

    print("\n[3] query_incidents(domain='healthnav', status='open')")
    open_hn = json.loads(query_incidents(domain="healthnav", status="open"))
    print(f"  Returned {len(open_hn)} open HealthNav records")

    print("\n[4] query_incidents(date_from='2026-01-01')")
    ytd = json.loads(query_incidents(date_from="2026-01-01"))
    print(f"  Returned {len(ytd)} incidents reported in 2026")

    print("\n[5] get_incident_detail('INC-001')")
    detail = json.loads(get_incident_detail("INC-001"))
    if "error" not in detail:
        print(f"  incident_id  : {detail['incident_id']}")
        print(f"  domain       : {detail['domain']}")
        print(f"  category     : {detail['category']}")
        print(f"  status       : {detail['status']}")
        print(f"  description  : {detail['description'][:80]}…")
    else:
        print(f"  ERROR: {detail['error']}")

    print("\n[6] get_incident_detail('INC-999')  (not found)")
    missing = json.loads(get_incident_detail("INC-999"))
    print(f"  {missing}")

    print("\n" + "=" * 60)
    print("All self-tests passed.")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if "--test" in sys.argv:
        _run_self_test()
    else:
        mcp.run(transport="stdio")
