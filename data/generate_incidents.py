"""
Generate synthetic WorkerShield incident records and write to data/incidents.db.

Produces 50 clearly-fictitious incident records across SafeShift, FairDesk, and
HealthNav domains, spread across the 12 months prior to June 2026.

Run:
    python3 data/generate_incidents.py
"""

from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DB_PATH   = Path(__file__).parent / "incidents.db"
REFERENCE_DATE = date(2026, 6, 12)   # today in the project timeline
START_DATE     = date(2025, 6, 12)   # 12 months back

random.seed(42)   # reproducible output


# ---------------------------------------------------------------------------
# Incident templates per domain
# ---------------------------------------------------------------------------

SAFESHIFT_INCIDENTS = [
    # (category, subcategory, severity, description_template, role, location)
    ("fatigue", "shift_fatigue", "high",
     "Worker reported extreme fatigue after completing a 12-hour night shift at the {loc} processing facility. No adequate rest break schedule was in place.",
     "Process operator", True),
    ("fatigue", "FIFO_fatigue", "medium",
     "FIFO worker flagged deteriorating alertness during week 3 of a 3:1 roster at the {loc} mine site. Supervisor observed micro-sleep episodes.",
     "FIFO drill operator", True),
    ("fatigue", "heat_fatigue", "high",
     "Outdoor worker experienced heat exhaustion symptoms after 6 hours of continuous work in 38°C conditions at {loc} without adequate hydration breaks.",
     "Civil construction worker", True),
    ("fatigue", "shift_fatigue", "critical",
     "Near-miss incident — forklift operator experienced fatigue-related lapse in concentration at {loc} warehouse, narrowly avoiding collision with pedestrian.",
     "Forklift operator", True),
    ("fatigue", "FIFO_fatigue", "low",
     "Workers raised fatigue concerns via toolbox talk at {loc} site. Fatigue management plan review initiated after roster extended without consultation.",
     "Site maintenance crew", True),
    ("manual_handling", "lifting_injury", "medium",
     "Warehouse worker sustained lower back strain lifting a 32 kg item without mechanical assistance at {loc} distribution centre.",
     "Warehouse picker", True),
    ("manual_handling", "repetitive_strain", "low",
     "Office worker at {loc} reported wrist pain consistent with repetitive strain injury after extended keyboard use without ergonomic assessment.",
     "Data entry officer", False),
    ("manual_handling", "lifting_injury", "high",
     "Worker required hospitalisation after attempting to reposition heavy machinery component at {loc} without team-lift protocol.",
     "Plant maintenance technician", True),
    ("slip_trip_fall", "wet_floor", "medium",
     "Employee slipped on unmarked wet floor near the {loc} amenities block, sustaining knee laceration. No wet floor signage was deployed.",
     "Administration officer", False),
    ("slip_trip_fall", "height", "critical",
     "Worker fell approximately 2.1 metres from scaffold at {loc} construction site. Inadequate edge protection identified as root cause.",
     "Scaffolder", True),
    ("slip_trip_fall", "uneven_surface", "low",
     "Worker tripped on raised concrete lip at {loc} site entrance during shift changeover. Incident reported with no injury.",
     "Electrical tradesperson", True),
    ("machinery", "lockout_failure", "critical",
     "Maintenance worker received electric shock at {loc} plant after isolation procedures were not applied prior to work on live panel.",
     "Maintenance electrician", True),
    ("machinery", "guarding_missing", "high",
     "Rotating machinery guard found removed at {loc} production line. Incident reported before injury occurred — guard had been removed for cleaning.",
     "Production line operator", True),
    ("chemical_exposure", "inhalation", "medium",
     "Worker reported respiratory irritation after inhaling cleaning chemical fumes at {loc} due to inadequate ventilation in confined space.",
     "Cleaning contractor", True),
    ("psychosocial", "workplace_stress", "low",
     "Worker lodged formal complaint citing excessive workload and management pressure at {loc} office. No physical injury reported.",
     "Project coordinator", False),
    ("ppe_non_compliance", "missing_helmet", "medium",
     "Safety audit at {loc} site identified three workers in designated hard-hat zone without PPE. Corrective action notices issued.",
     "Multiple roles", True),
    ("fatigue", "shift_fatigue", "medium",
     "Night-shift supervisor at {loc} depot raised concerns about staffing levels leading to mandatory overtime for consecutive nights.",
     "Logistics supervisor", True),
    ("manual_handling", "repetitive_strain", "medium",
     "Packing line worker at {loc} facility reported shoulder pain after six months of repetitive overhead reaching. Ergonomic review commenced.",
     "Packing operator", True),
]

FAIRDESK_INCIDENTS = [
    # (category, subcategory, description, role, location_office)
    ("unfair_dismissal", "procedural_fairness",
     "Employee terminated without opportunity to respond to performance concerns at {loc} office. Application lodged with Fair Work Commission alleging procedural unfairness.",
     "Sales representative", False),
    ("unfair_dismissal", "summary_dismissal",
     "Employee dismissed on the spot following workplace altercation at {loc} site. Employer did not conduct investigation prior to termination.",
     "Machine operator", True),
    ("casual_conversion", "eligibility_dispute",
     "Casual employee at {loc} facility requested conversion to permanent part-time after 18 months regular engagement. Employer disputed regular and systematic pattern.",
     "Retail assistant", False),
    ("casual_conversion", "employer_refusal",
     "Employer at {loc} office refused casual conversion request citing business uncertainty without providing written reasons within the required 21-day period.",
     "Administrative officer", False),
    ("flexible_working", "denied_request",
     "Employee caring for a child under school age submitted flexible working request at {loc}. Employer verbally refused without written response within 21 days.",
     "Customer service officer", False),
    ("flexible_working", "reasonable_grounds_dispute",
     "Employee with a disability requested modified start times at {loc} to attend medical appointments. Employer cited operational requirements without evidence.",
     "IT support technician", False),
    ("underpayment", "overtime_unpaid",
     "Workers at {loc} warehouse discovered overtime worked on three public holidays was paid at single time rather than the applicable penalty rate.",
     "Warehouse team members", True),
    ("underpayment", "allowance_missing",
     "FIFO workers at {loc} site were not receiving the applicable site allowance under their enterprise agreement for the duration of the wet season.",
     "FIFO maintenance workers", True),
    ("bullying", "repeated_behaviour",
     "Employee at {loc} office made formal bullying complaint citing repeated exclusion from meetings, undermining of work, and belittling comments from line manager.",
     "Project officer", False),
    ("bullying", "management_action",
     "Supervisor at {loc} facility issued performance management after team raised safety concerns. Workers allege adverse action under general protections provisions.",
     "Production team lead", True),
    ("discrimination", "protected_attribute",
     "Job applicant at {loc} alleges they were not shortlisted for promotion due to pregnancy, constituting adverse action on the basis of a protected attribute.",
     "Administration team member", False),
    ("discrimination", "adverse_action",
     "Worker at {loc} made a complaint to the Fair Work Ombudsman and alleges subsequent reduction in rostered hours constituted adverse action.",
     "Hospitality worker", False),
    ("unfair_dismissal", "procedural_fairness",
     "Long-serving employee at {loc} distribution centre dismissed for alleged misconduct. Investigation conducted by the same manager who lodged the complaint.",
     "Logistics coordinator", True),
    ("casual_conversion", "eligibility_dispute",
     "Agency-engaged casual at {loc} plant sought conversion. Dispute arose over whether the engagement period met the 12-month threshold under the NES.",
     "Process technician (labour hire)", True),
    ("flexible_working", "denied_request",
     "Employee returning from parental leave requested four-day week at {loc}. Employer offered one day per week remote only, without genuine consideration of the request.",
     "Marketing coordinator", False),
    ("underpayment", "overtime_unpaid",
     "Audit at {loc} identified systematic non-payment of meal break allowances for shifts exceeding 5 hours. Underpayment estimated across 12 workers over 8 months.",
     "Retail store workers", False),
]

HEALTHNAV_INCIDENTS = [
    # (category, subcategory, description, role, location_site)
    ("return_to_work", "modified_duties",
     "Worker recovering from lumbar spine surgery at {loc} — RTW plan requires modified duties for 8 weeks. Employer struggling to identify suitable alternative tasks.",
     "Heavy vehicle operator", True),
    ("return_to_work", "RTW_plan_breach",
     "Treating doctor's RTW restrictions at {loc} were not communicated to the work supervisor. Worker returned to full duties 3 weeks early, aggravating the injury.",
     "Warehouse supervisor", True),
    ("workers_compensation", "claim_disputed",
     "WorkCover QLD rejected the psychological injury claim lodged by worker at {loc} alleging the injury did not arise out of employment. Decision under review.",
     "Emergency services support officer", False),
    ("workers_compensation", "insurer_delay",
     "Insurer failed to respond to medical certificate lodged by injured worker at {loc} within the statutory 20-business-day period, causing financial hardship.",
     "Construction labourer", True),
    ("mental_health", "psychological_injury",
     "Worker at {loc} lodged workers compensation claim for psychological injury following prolonged workplace bullying. Claim accepted; RTW with same employer deemed not suitable.",
     "Customer contact centre worker", False),
    ("mental_health", "burnout",
     "Senior manager at {loc} presented to GP with burnout symptoms after 18 months without leave. Employer had not provided adequate cover during extended understaffing.",
     "Operations manager", False),
    ("mental_health", "psychological_injury",
     "FIFO worker at {loc} mine site returned from 28-day swing with acute psychological distress. Access to Employee Assistance Programme not communicated at site induction.",
     "FIFO geologist", True),
    ("musculoskeletal", "back_injury",
     "Nurse at {loc} sustained acute lower back injury during patient transfer. Hospital had not deployed mechanical patient lifting aids on the ward.",
     "Registered nurse", False),
    ("musculoskeletal", "shoulder_injury",
     "Worker at {loc} fruit packing shed sustained rotator cuff injury. RTW plan in place but modified duties limited to 2 hours per day, creating viability concerns.",
     "Packing line worker", True),
    ("return_to_work", "modified_duties",
     "Injured worker at {loc} is 6 weeks into RTW plan. Employer requesting WorkCover fund an additional ergonomic workstation. Approval pending.",
     "Data analyst", False),
    ("workers_compensation", "claim_disputed",
     "WorkCover QLD dispute panel reviewing claim by worker at {loc} who alleges acoustic trauma from sustained loud machinery noise over 4-year employment.",
     "Printing press operator", True),
    ("mental_health", "burnout",
     "School teacher on workers compensation at {loc} for stress-related burnout. RTW plan includes graduated return; employer disputes number of modified hours proposed.",
     "Secondary school teacher", False),
    ("musculoskeletal", "back_injury",
     "Courier driver at {loc} depot on RTW after disc herniation. Employer raising concerns about driver's capacity to meet delivery KPIs on modified duties.",
     "Courier driver", True),
    ("occupational_disease", "noise_induced_hearing_loss",
     "Retired tradesperson claiming WorkCover for noise-induced hearing loss (NIHL) attributable to 20 years of unprotected exposure at {loc} manufacturing site.",
     "Boilermaker (retired)", True),
    ("return_to_work", "RTW_plan_breach",
     "WorkCover notified that host employer at {loc} had assigned injured labour-hire worker tasks beyond their certified capacity, triggering a formal non-compliance notice.",
     "Labour-hire production worker", True),
    ("workers_compensation", "insurer_delay",
     "Insurer at {loc} has not provided a liability decision 35 business days after the claim. Worker has not received weekly payments and is facing mortgage stress.",
     "Aged care worker", False),
]


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------

SITE_LOCATIONS = [
    "Townsville", "Mackay", "Mt Isa", "Gladstone", "Rockhampton",
    "Toowoomba", "Cairns", "Bundaberg", "Emerald", "Bowen Basin",
]
OFFICE_LOCATIONS = [
    "Brisbane CBD", "South Brisbane", "Fortitude Valley", "Ipswich",
    "Gold Coast", "Sunshine Coast", "Logan", "Redlands",
]


def _pick_location(is_site: bool) -> str:
    pool = SITE_LOCATIONS if is_site else OFFICE_LOCATIONS
    return random.choice(pool)


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def _random_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))


def _resolve_date(reported: date, status: str) -> tuple[str | None, int | None]:
    if status == "open":
        return None, None
    if status == "in_progress":
        return None, None
    # closed — resolve 7–90 days after report
    days = random.randint(7, 90)
    resolved = reported + timedelta(days=days)
    if resolved > REFERENCE_DATE:
        resolved = REFERENCE_DATE - timedelta(days=1)
    return resolved.isoformat(), (resolved - reported).days


# ---------------------------------------------------------------------------
# Build incident rows
# ---------------------------------------------------------------------------

def _build_safeshift_rows() -> list[dict]:
    rows = []
    statuses = (["open"] * 6 + ["in_progress"] * 5 + ["closed"] * 7)
    random.shuffle(statuses)

    for i, tmpl in enumerate(SAFESHIFT_INCIDENTS):
        category, subcat, severity, desc_tmpl, role, is_site = tmpl
        loc = _pick_location(is_site)
        reported = _random_date(START_DATE, REFERENCE_DATE - timedelta(days=7))
        status = statuses[i % len(statuses)]
        resolved_str, days = _resolve_date(reported, status)

        rows.append({
            "incident_id":    f"INC-{len(rows)+1:03d}",
            "domain":         "safeshift",
            "category":       category,
            "subcategory":    subcat,
            "description":    desc_tmpl.format(loc=loc),
            "status":         status,
            "severity":       severity,
            "reported_date":  reported.isoformat(),
            "resolved_date":  resolved_str,
            "worker_role":    role,
            "location":       loc,
            "outcome":        _safeshift_outcome(status, category),
            "days_to_resolve": days,
        })
    return rows


def _build_fairdesk_rows(offset: int) -> list[dict]:
    rows = []
    statuses = (["open"] * 5 + ["in_progress"] * 5 + ["closed"] * 6)
    random.shuffle(statuses)

    for i, tmpl in enumerate(FAIRDESK_INCIDENTS):
        category, subcat, desc_tmpl, role, is_site = tmpl
        loc = _pick_location(is_site)
        reported = _random_date(START_DATE, REFERENCE_DATE - timedelta(days=7))
        status = statuses[i % len(statuses)]
        resolved_str, days = _resolve_date(reported, status)

        rows.append({
            "incident_id":    f"INC-{offset + len(rows)+1:03d}",
            "domain":         "fairdesk",
            "category":       category,
            "subcategory":    subcat,
            "description":    desc_tmpl.format(loc=loc),
            "status":         status,
            "severity":       None,
            "reported_date":  reported.isoformat(),
            "resolved_date":  resolved_str,
            "worker_role":    role,
            "location":       loc,
            "outcome":        _fairdesk_outcome(status, category),
            "days_to_resolve": days,
        })
    return rows


def _build_healthnav_rows(offset: int) -> list[dict]:
    rows = []
    statuses = (["open"] * 5 + ["in_progress"] * 6 + ["closed"] * 5)
    random.shuffle(statuses)

    for i, tmpl in enumerate(HEALTHNAV_INCIDENTS):
        category, subcat, desc_tmpl, role, is_site = tmpl
        loc = _pick_location(is_site)
        reported = _random_date(START_DATE, REFERENCE_DATE - timedelta(days=7))
        status = statuses[i % len(statuses)]
        resolved_str, days = _resolve_date(reported, status)

        rows.append({
            "incident_id":    f"INC-{offset + len(rows)+1:03d}",
            "domain":         "healthnav",
            "category":       category,
            "subcategory":    subcat,
            "description":    desc_tmpl.format(loc=loc),
            "status":         status,
            "severity":       None,
            "reported_date":  reported.isoformat(),
            "resolved_date":  resolved_str,
            "worker_role":    role,
            "location":       loc,
            "outcome":        _healthnav_outcome(status, category),
            "days_to_resolve": days,
        })
    return rows


# ---------------------------------------------------------------------------
# Outcome generators (brief narrative per domain/status)
# ---------------------------------------------------------------------------

_SS_OUTCOMES = {
    "closed": {
        "fatigue":           "Fatigue risk assessment completed; revised rest break schedule implemented and roster reviewed.",
        "manual_handling":   "Manual handling risk control installed; worker received physiotherapy; return to work on modified duties.",
        "slip_trip_fall":    "Hazard remediated; wet-floor signage and non-slip matting installed at location.",
        "machinery":         "Isolation procedure updated; LOTO signage improved; toolbox talk delivered to all maintenance staff.",
        "chemical_exposure": "Ventilation upgraded; SDS reviewed; PPE (respirator) issued to relevant workers.",
        "psychosocial":      "Workload review conducted; additional resource allocated; employee referred to EAP.",
        "ppe_non_compliance":"Corrective action notices issued; mandatory refresher training completed by all affected workers.",
    },
    "in_progress": "Investigation underway; interim controls implemented pending root cause analysis.",
    "open":        "Incident logged; supervisor notified; hazard control pending assignment.",
}

_FD_OUTCOMES = {
    "closed": {
        "unfair_dismissal":  "Fair Work Commission conciliation resulted in agreed compensation settlement.",
        "casual_conversion": "Employer provided written response; conversion granted following review of engagement records.",
        "flexible_working":  "Written response issued; request granted with agreed trial period.",
        "underpayment":      "Back-pay calculated and issued; payroll process corrected prospectively.",
        "bullying":          "Anti-bullying order sought; mediation conducted; management action plan implemented.",
        "discrimination":    "Complaint resolved through Fair Work Commission conciliation; no admission of liability.",
    },
    "in_progress": "Matter lodged with Fair Work Commission; initial conciliation conference scheduled.",
    "open":        "Complaint received; employer notified; response period open.",
}

_HN_OUTCOMES = {
    "closed": {
        "return_to_work":        "RTW plan completed; worker returned to full duties without further restriction.",
        "workers_compensation":  "Claim accepted; weekly payments and medical expenses covered; worker recovered.",
        "mental_health":         "Psychological treatment completed; graduated RTW plan successfully concluded.",
        "musculoskeletal":       "Worker completed rehabilitation; returned to pre-injury duties with ergonomic modifications.",
        "occupational_disease":  "Claim assessed; permanent impairment determination made; lump-sum settlement approved.",
    },
    "in_progress": "RTW plan or claim assessment currently in progress; next review scheduled.",
    "open":        "Claim or RTW referral received; initial medical assessment pending.",
}


def _safeshift_outcome(status: str, category: str) -> str:
    if status == "closed":
        return _SS_OUTCOMES["closed"].get(category, "Incident resolved; corrective actions implemented.")
    return _SS_OUTCOMES.get(status, "")


def _fairdesk_outcome(status: str, category: str) -> str:
    if status == "closed":
        return _FD_OUTCOMES["closed"].get(category, "Matter resolved.")
    return _FD_OUTCOMES.get(status, "")


def _healthnav_outcome(status: str, category: str) -> str:
    if status == "closed":
        return _HN_OUTCOMES["closed"].get(category, "Case closed; outcome documented.")
    return _HN_OUTCOMES.get(status, "")


# ---------------------------------------------------------------------------
# SQLite writer
# ---------------------------------------------------------------------------

DDL = """
CREATE TABLE IF NOT EXISTS incidents (
    incident_id     TEXT PRIMARY KEY,
    domain          TEXT NOT NULL,
    category        TEXT NOT NULL,
    subcategory     TEXT,
    description     TEXT NOT NULL,
    status          TEXT NOT NULL,
    severity        TEXT,
    reported_date   TEXT NOT NULL,
    resolved_date   TEXT,
    worker_role     TEXT,
    location        TEXT,
    outcome         TEXT,
    days_to_resolve INTEGER
);
"""


def write_db(rows: list[dict]) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()

    con = sqlite3.connect(DB_PATH)
    con.execute(DDL)
    con.executemany(
        """
        INSERT INTO incidents VALUES (
            :incident_id, :domain, :category, :subcategory,
            :description, :status, :severity,
            :reported_date, :resolved_date,
            :worker_role, :location, :outcome, :days_to_resolve
        )
        """,
        rows,
    )
    con.commit()

    count = con.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
    print(f"Written {count} incident records to {DB_PATH}")

    # Print a quick summary
    print("\nSummary by domain × status:")
    for row in con.execute(
        "SELECT domain, status, COUNT(*) AS n FROM incidents GROUP BY domain, status ORDER BY domain, status"
    ).fetchall():
        print(f"  {row[0]:<12} {row[1]:<12} {row[2]}")

    print("\nFatigue-related incidents:")
    for row in con.execute(
        "SELECT incident_id, status, reported_date, location FROM incidents WHERE category = 'fatigue'"
    ).fetchall():
        print(f"  {row[0]}  {row[1]:<12} {row[2]}  {row[3]}")

    con.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ss_rows = _build_safeshift_rows()
    fd_rows = _build_fairdesk_rows(offset=len(ss_rows))
    hn_rows = _build_healthnav_rows(offset=len(ss_rows) + len(fd_rows))
    all_rows = ss_rows + fd_rows + hn_rows
    print(f"Generated {len(all_rows)} synthetic incident records  "
          f"(SS={len(ss_rows)}  FD={len(fd_rows)}  HN={len(hn_rows)})")
    write_db(all_rows)
