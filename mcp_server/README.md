# WorkerShield Incident MCP Server

A Model Context Protocol (MCP) server that exposes the synthetic WorkerShield incident
database as queryable tools. Enables Claude Code and the WorkerShield LangGraph agent
to combine document-based compliance obligations with real incident statistics.

---

## Tools

| Tool | Description |
|------|-------------|
| `query_incidents` | Filtered query — by domain, status, category, and/or date range |
| `get_incident_summary` | Aggregated counts grouped by domain × status plus category breakdown |
| `get_incident_detail` | Full record for a single incident by ID |

---

## Quick start

```bash
# Ensure the database exists
python3 data/generate_incidents.py

# Self-test (no MCP client needed)
python3 mcp_server/incident_server.py --test

# Run as MCP server (stdio transport — used by Claude Code)
python3 mcp_server/incident_server.py
```

---

## Register with Claude Code

### Project-level (recommended for this repo)

```bash
claude mcp add workershield-incidents \
  -- python3 /projects/workershield-v1/mcp_server/incident_server.py
```

This writes an entry to `.claude/mcp.json` (project scope).

### User-level (available in all projects)

```bash
claude mcp add --scope user workershield-incidents \
  -- python3 /projects/workershield-v1/mcp_server/incident_server.py
```

### Verify registration

```bash
claude mcp list
```

You should see `workershield-incidents` in the list with a `✓ connected` status.

### Remove

```bash
claude mcp remove workershield-incidents
```

---

## Register with the WorkerShield app

The WorkerShield LangGraph agent calls the same underlying database functions
directly (via `data/incidents_db.py`) rather than spawning an MCP subprocess —
this avoids IPC overhead inside the graph while keeping the tools available to
external MCP clients.

To wire the MCP server into a production WorkerShield deployment (e.g. a
containerised API), use the `mcp` Python client:

```python
import asyncio
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

SERVER_PATH = "/projects/workershield-v1/mcp_server/incident_server.py"

async def call_mcp_tool(tool_name: str, arguments: dict) -> str:
    params = StdioServerParameters(command="python3", args=[SERVER_PATH])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            return result.content[0].text if result.content else ""

# Example
summary = asyncio.run(call_mcp_tool("get_incident_summary", {}))
```

---

## Database

- **Location:** `data/incidents.db` (SQLite)
- **Schema:** `data/incidents_schema.md`
- **Generator:** `data/generate_incidents.py` — re-run to regenerate (deterministic, seed=42)
- **Records:** 50 synthetic incidents across SafeShift (18), FairDesk (16), HealthNav (16)
- **Date range:** June 2025 – June 2026

---

## Architecture notes

```
Claude Code  ──MCP stdio──▶  mcp_server/incident_server.py
                                       │
WorkerShield agent                     │  (both import)
  incident_check_node  ──direct──▶  data/incidents_db.py
                                       │
                                  data/incidents.db
```

The shared `data/incidents_db.py` module means the MCP server and the agent node
use identical query logic — no duplication, easy to test.
