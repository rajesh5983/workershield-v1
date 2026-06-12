# WorkerShield Incident Database — Schema Design

## Overview

SQLite database (`data/incidents.db`) storing synthetic workplace incident records
across all three WorkerShield domains. Designed to support realistic operational
queries such as "how many fatigue-related incidents this quarter" or "show all open
return-to-work cases".

---

## Table: `incidents`

| Column | Type | Nullable | Description |
|--------|------|----------|-------------|
| `incident_id` | TEXT | NOT NULL (PK) | Unique identifier — format `INC-NNN` |
| `domain` | TEXT | NOT NULL | `safeshift` / `fairdesk` / `healthnav` |
| `category` | TEXT | NOT NULL | Primary incident category (see below) |
| `subcategory` | TEXT | YES | More granular classification |
| `description` | TEXT | NOT NULL | Plain-English summary of the incident |
| `status` | TEXT | NOT NULL | `open` / `in_progress` / `closed` |
| `severity` | TEXT | YES | `low` / `medium` / `high` / `critical` — used mainly for SafeShift WHS incidents |
| `reported_date` | TEXT | NOT NULL | ISO-8601 date the incident was reported |
| `resolved_date` | TEXT | YES | ISO-8601 date resolved — NULL when open/in_progress |
| `worker_role` | TEXT | YES | Job role of the affected worker |
| `location` | TEXT | YES | Site or region (fictionalised) |
| `outcome` | TEXT | YES | Narrative outcome or action taken |
| `days_to_resolve` | INTEGER | YES | Calendar days from report to resolution (NULL if unresolved) |

---

## Categories by Domain

### SafeShift (WHS incidents)
| Category | Subcategory examples |
|----------|---------------------|
| `fatigue` | shift_fatigue, heat_fatigue, FIFO_fatigue |
| `manual_handling` | lifting_injury, repetitive_strain |
| `slip_trip_fall` | wet_floor, uneven_surface, height |
| `machinery` | lockout_failure, guarding_missing |
| `chemical_exposure` | spill, inhalation |
| `psychosocial` | workload, violence, workplace_stress |
| `ppe_non_compliance` | missing_helmet, improper_footwear |

### FairDesk (Workplace relations disputes)
| Category | Subcategory examples |
|----------|---------------------|
| `unfair_dismissal` | procedural_fairness, summary_dismissal |
| `casual_conversion` | eligibility_dispute, employer_refusal |
| `flexible_working` | denied_request, reasonable_grounds_dispute |
| `underpayment` | overtime_unpaid, allowance_missing |
| `bullying` | repeated_behaviour, management_action |
| `discrimination` | protected_attribute, adverse_action |

### HealthNav (Occupational health / return-to-work)
| Category | Subcategory examples |
|----------|---------------------|
| `return_to_work` | modified_duties, RTW_plan_breach |
| `workers_compensation` | claim_disputed, insurer_delay |
| `mental_health` | psychological_injury, burnout |
| `musculoskeletal` | back_injury, shoulder_injury |
| `occupational_disease` | noise_induced_hearing_loss, dermatitis |

---

## Status Lifecycle

```
reported → open → in_progress → closed
                ↑_______________|  (reopened)
```

- **open**: incident logged, investigation not yet started
- **in_progress**: investigation / treatment / negotiation underway
- **closed**: resolved, outcome documented

---

## Supported Query Patterns

| Query intent | SQL pattern |
|---|---|
| Fatigue incidents this year | `WHERE category = 'fatigue' AND reported_date >= '2026-01-01'` |
| Open RTW cases | `WHERE domain = 'healthnav' AND category = 'return_to_work' AND status = 'open'` |
| Severity breakdown | `GROUP BY severity` |
| Domain × status counts | `GROUP BY domain, status` |
| Last quarter | `WHERE reported_date >= date('now', '-3 months')` |
| Average resolution time | `AVG(days_to_resolve) WHERE resolved_date IS NOT NULL` |

---

## Design Notes

- Dates span the 12 months prior to June 2026 to give temporal breadth for "last quarter" queries.
- Worker roles and locations are clearly fictional (no real company names, generic QLD/remote sites).
- Severity is stored only for SafeShift records; FairDesk/HealthNav use NULL.
- `days_to_resolve` is pre-computed at insert time (not a virtual column) for simple SQLite compatibility.
