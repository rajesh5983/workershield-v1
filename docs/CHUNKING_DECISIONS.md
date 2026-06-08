# WorkerShield Chunking Decisions

Strategies assigned per document after validation against actual PDF content
(first 3 pages sampled per document, June 2026).

| doc_id | title | strategy | rationale |
|--------|-------|----------|-----------|
| FD01 | Introduction to NES | recursive | Short 2-page fact sheet, flowing prose with bullet points — no clause hierarchy |
| FD02 | Casual Employment Statement | recursive | Short 3-page fact sheet, flowing prose |
| FD03 | Flexible Working Best Practice Guide | recursive | Primarily prose narrative — no pipe/tab tables detected in PDF extraction |
| HN01 | Work-Related Psychological Health | section_header | 43-page guide with clear section structure beyond TOC |
| HN02 | Fatigue Fact Sheet | recursive | Short 3-page fact sheet, flowing prose |
| HN03 | Workers Compensation Entitlements | section_header | 12-page structured guide with headed sections |
| SS01 | Managing Work Environment CoP | section_header | 42-page Code of Practice with numbered section headings |
| SS02 | Hazardous Manual Tasks CoP | section_header | 71-page Code of Practice with numbered section headings |
| SS03a | Queensland WHS Act 2011 | clause_boundary | 308-page legislation with strict numbered clause hierarchy |
| SS03b | Guide to Model WHS Act | clause_boundary | 42-page duties guide structured around legislative clauses |

---

## Corpus Map

```mermaid
%% WorkerShield — Corpus and Chunk Strategy by Domain
flowchart TD
    subgraph safeshift["SafeShift — WHS Law"]
        SS01["SS01\nManaging Work Environment CoP\nsection_header"]
        SS02["SS02\nHazardous Manual Tasks CoP\nsection_header"]
        SS03a["SS03a\nQLD WHS Act 2011\nclause_boundary"]
        SS03b["SS03b\nGuide to Model WHS Act\nclause_boundary"]
    end
    subgraph fairdesk["FairDesk — Fair Work"]
        FD01["FD01\nIntroduction to NES\nrecursive"]
        FD02["FD02\nCasual Employment Statement\nrecursive"]
        FD03["FD03\nFlexible Working Best Practice\nrecursive"]
    end
    subgraph healthnav["HealthNav — Occupational Health"]
        HN01["HN01\nWork-Related Psychological Health\nsection_header"]
        HN02["HN02\nFatigue Fact Sheet\nrecursive"]
        HN03["HN03\nWorkers Compensation Entitlements\nsection_header"]
    end
    Qdrant[("Qdrant\ncollection: workershield\n1,268 vectors")]

    SS01 & SS02 & SS03a & SS03b --> Qdrant
    FD01 & FD02 & FD03 --> Qdrant
    HN01 & HN02 & HN03 --> Qdrant
```
