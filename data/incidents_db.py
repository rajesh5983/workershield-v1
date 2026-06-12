"""
Shared SQLite query helpers for the WorkerShield incident database.

Imported by:
  - mcp_server/incident_server.py  (MCP tool wrappers)
  - agents/graph.py incident_check_node  (direct in-process calls)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent / "incidents.db"


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def query_incidents(
    domain: str | None = None,
    status: str | None = None,
    category: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return filtered incident records as a list of dicts.

    All parameters are optional; omit to skip that filter.
    date_from / date_to are ISO-8601 strings (YYYY-MM-DD).
    """
    clauses: list[str] = []
    params:  list[Any] = []

    if domain:
        clauses.append("domain = ?")
        params.append(domain.lower())
    if status:
        clauses.append("status = ?")
        params.append(status.lower())
    if category:
        clauses.append("(category = ? OR subcategory LIKE ?)")
        params.extend([category.lower(), f"%{category.lower()}%"])
    if date_from:
        clauses.append("reported_date >= ?")
        params.append(date_from)
    if date_to:
        clauses.append("reported_date <= ?")
        params.append(date_to)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"SELECT * FROM incidents {where} ORDER BY reported_date DESC LIMIT ?"
    params.append(limit)

    with _connect() as con:
        rows = con.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def get_incident_summary() -> dict[str, Any]:
    """Return counts grouped by domain × status, plus category breakdown."""
    with _connect() as con:
        by_domain_status = con.execute(
            """
            SELECT domain, status, COUNT(*) AS count
            FROM incidents
            GROUP BY domain, status
            ORDER BY domain, status
            """
        ).fetchall()

        by_category = con.execute(
            """
            SELECT domain, category, COUNT(*) AS count
            FROM incidents
            GROUP BY domain, category
            ORDER BY domain, count DESC
            """
        ).fetchall()

        totals = con.execute(
            """
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN status = 'open'        THEN 1 ELSE 0 END) AS open,
                SUM(CASE WHEN status = 'in_progress' THEN 1 ELSE 0 END) AS in_progress,
                SUM(CASE WHEN status = 'closed'      THEN 1 ELSE 0 END) AS closed,
                ROUND(AVG(CASE WHEN days_to_resolve IS NOT NULL THEN days_to_resolve END), 1) AS avg_days_to_resolve
            FROM incidents
            """
        ).fetchone()

    return {
        "totals": dict(totals),
        "by_domain_status": [dict(r) for r in by_domain_status],
        "by_category":      [dict(r) for r in by_category],
    }


def get_incident_detail(incident_id: str) -> dict[str, Any] | None:
    """Return the full record for a single incident, or None if not found."""
    with _connect() as con:
        row = con.execute(
            "SELECT * FROM incidents WHERE incident_id = ?",
            (incident_id.upper(),),
        ).fetchone()
    return dict(row) if row else None
