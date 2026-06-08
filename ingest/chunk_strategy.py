"""
Chunking strategies for WorkerShield corpus ingestion.

Each strategy accepts raw text and a metadata dict, and returns a list of
chunk dicts with keys: text, chunk_type, section, page_estimate.

Dispatch via get_chunker(doc_id) which reads docs/corpus_registry.yaml.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Callable

import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

ChunkDict = dict  # keys: text, chunk_type, section, page_estimate

_REGISTRY_PATH = Path(__file__).resolve().parents[1] / "docs" / "corpus_registry.yaml"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_APPROX_CHARS_PER_PAGE = 2_500


def _page_estimate(char_offset: int, total_chars: int, total_pages: int) -> int:
    """Return a 1-based page estimate from character position."""
    if total_chars == 0 or total_pages == 0:
        return 1
    fraction = char_offset / total_chars
    return max(1, math.ceil(fraction * total_pages))


# ---------------------------------------------------------------------------
# Strategy 1 — recursive
# ---------------------------------------------------------------------------

def chunk_recursive(text: str, meta: dict) -> list[ChunkDict]:
    """LangChain RecursiveCharacterTextSplitter, ~512 tokens / 50 overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512 * 4,   # ~4 chars per token
        chunk_overlap=50 * 4,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    splits = splitter.split_text(text)
    total_pages = meta.get("pages", 1)
    total_chars = len(text)

    chunks: list[ChunkDict] = []
    offset = 0
    for split in splits:
        idx = text.find(split, offset)
        if idx != -1:
            offset = idx
        chunks.append({
            "text": split,
            "chunk_type": "recursive",
            "section": "",
            "page_estimate": _page_estimate(offset, total_chars, total_pages),
        })
        offset += len(split)
    return chunks


# ---------------------------------------------------------------------------
# Strategy 2 — clause_boundary
# ---------------------------------------------------------------------------

# Matches clause numbers like: 1.  /  1.1  /  1.1.1  /  12.  at line start
_CLAUSE_RE = re.compile(
    r"(?m)^(\d+(?:\.\d+)*\.?\s)",
)


def chunk_clause_boundary(text: str, meta: dict) -> list[ChunkDict]:
    """Split on numbered clause patterns; keep clause number in section field."""
    total_pages = meta.get("pages", 1)
    total_chars = len(text)

    boundaries = [(m.start(), m.group(1).strip()) for m in _CLAUSE_RE.finditer(text)]

    if not boundaries:
        # No clauses detected — fall back to recursive
        return chunk_recursive(text, meta)

    chunks: list[ChunkDict] = []
    for i, (start, clause_num) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        chunks.append({
            "text": body,
            "chunk_type": "clause_boundary",
            "section": clause_num,
            "page_estimate": _page_estimate(start, total_chars, total_pages),
        })
    return chunks


# ---------------------------------------------------------------------------
# Strategy 3 — table_aware
# ---------------------------------------------------------------------------

# Detects lines that look like pipe-delimited or multi-space-separated table rows
_TABLE_ROW_RE = re.compile(r"^.{0,200}(\|.+\||\t.+\t).{0,200}$", re.MULTILINE)
_TABLE_BLOCK_RE = re.compile(
    r"((?:^.{0,200}(?:\|.+\||\t.+\t).{0,200}\n?){2,})",
    re.MULTILINE,
)


def _extract_headers(table_block: str) -> str:
    """Return the first row of a table block as a header prefix."""
    first_line = table_block.strip().splitlines()[0]
    cells = re.split(r"\||\t{2,}", first_line)
    headers = [c.strip() for c in cells if c.strip()]
    return "Headers: " + " | ".join(headers)


def chunk_table_aware(text: str, meta: dict) -> list[ChunkDict]:
    """Detect table blocks, prepend column headers to each row chunk; split
    prose sections with recursive splitter."""
    total_pages = meta.get("pages", 1)
    total_chars = len(text)
    chunks: list[ChunkDict] = []

    last_end = 0
    for match in _TABLE_BLOCK_RE.finditer(text):
        prose_before = text[last_end:match.start()].strip()
        if prose_before:
            for c in chunk_recursive(prose_before, meta):
                c["chunk_type"] = "table_aware_prose"
                chunks.append(c)

        table_block = match.group(0)
        lines = table_block.strip().splitlines()
        header_prefix = _extract_headers(table_block)

        for row in lines[1:]:   # skip header row itself
            row = row.strip()
            if not row:
                continue
            chunks.append({
                "text": f"{header_prefix}\n{row}",
                "chunk_type": "table_row",
                "section": header_prefix,
                "page_estimate": _page_estimate(match.start(), total_chars, total_pages),
            })

        last_end = match.end()

    tail = text[last_end:].strip()
    if tail:
        for c in chunk_recursive(tail, meta):
            c["chunk_type"] = "table_aware_prose"
            chunks.append(c)

    if not chunks:
        return chunk_recursive(text, meta)
    return chunks


# ---------------------------------------------------------------------------
# Strategy 4 — section_header
# ---------------------------------------------------------------------------

# Matches: ALL CAPS headings, numbered headings (1 Introduction), underlined lines
_SECTION_RE = re.compile(
    r"(?m)"
    r"(?:^([A-Z][A-Z\s]{4,})\s*$"           # ALL CAPS line
    r"|^(\d+(?:\.\d+)*\s+[A-Z][^\n]{3,})\s*$"  # 1.2 Heading text
    r"|^([^\n]{5,})\n[-=]{4,}\s*$)",            # Underlined heading
)


def chunk_section_header(text: str, meta: dict) -> list[ChunkDict]:
    """Split on heading patterns; carry section title into each chunk."""
    total_pages = meta.get("pages", 1)
    total_chars = len(text)

    boundaries: list[tuple[int, str]] = []
    for m in _SECTION_RE.finditer(text):
        title = (m.group(1) or m.group(2) or m.group(3) or "").strip()
        boundaries.append((m.start(), title))

    if not boundaries:
        return chunk_recursive(text, meta)

    chunks: list[ChunkDict] = []

    # Text before the first heading
    preamble = text[: boundaries[0][0]].strip()
    if preamble:
        chunks.append({
            "text": preamble,
            "chunk_type": "section_header",
            "section": "",
            "page_estimate": _page_estimate(0, total_chars, total_pages),
        })

    for i, (start, title) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        # Sub-split long sections so chunks stay within ~512-token budget
        if len(body) > 512 * 4:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=512 * 4, chunk_overlap=50 * 4
            )
            for sub in splitter.split_text(body):
                chunks.append({
                    "text": sub,
                    "chunk_type": "section_header",
                    "section": title,
                    "page_estimate": _page_estimate(start, total_chars, total_pages),
                })
        else:
            chunks.append({
                "text": body,
                "chunk_type": "section_header",
                "section": title,
                "page_estimate": _page_estimate(start, total_chars, total_pages),
            })
    return chunks


# ---------------------------------------------------------------------------
# Strategy 5 — recommendation
# ---------------------------------------------------------------------------

_OBLIGATION_RE = re.compile(
    r"(?i)\b(must not|must|shall not|shall|should|required to|is required|"
    r"obligation|duty|duties|responsible for)\b"
)


def chunk_recommendation(text: str, meta: dict) -> list[ChunkDict]:
    """Split on obligation/duty signal words; each chunk anchors one directive."""
    total_pages = meta.get("pages", 1)
    total_chars = len(text)

    # Split into sentences first, then group by obligation signal
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks: list[ChunkDict] = []
    current: list[str] = []
    char_offset = 0

    for sentence in sentences:
        if _OBLIGATION_RE.search(sentence):
            if current:
                body = " ".join(current).strip()
                if body:
                    chunks.append({
                        "text": body,
                        "chunk_type": "recommendation",
                        "section": "",
                        "page_estimate": _page_estimate(char_offset, total_chars, total_pages),
                    })
            current = [sentence]
            char_offset = text.find(sentence, char_offset)
        else:
            current.append(sentence)
        char_offset += len(sentence) + 1

    if current:
        body = " ".join(current).strip()
        if body:
            chunks.append({
                "text": body,
                "chunk_type": "recommendation",
                "section": "",
                "page_estimate": _page_estimate(char_offset, total_chars, total_pages),
            })

    if not chunks:
        return chunk_recursive(text, meta)
    return chunks


# ---------------------------------------------------------------------------
# Strategy map & dispatch
# ---------------------------------------------------------------------------

_STRATEGY_MAP: dict[str, Callable[[str, dict], list[ChunkDict]]] = {
    "recursive": chunk_recursive,
    "clause_boundary": chunk_clause_boundary,
    "table_aware": chunk_table_aware,
    "section_header": chunk_section_header,
    "recommendation": chunk_recommendation,
}


def _load_registry() -> dict[str, str]:
    """Return {doc_id: chunk_strategy} from corpus_registry.yaml."""
    with open(_REGISTRY_PATH, "r") as fh:
        data = yaml.safe_load(fh)
    return {doc["id"]: doc["chunk_strategy"] for doc in data["documents"]}


def get_chunker(doc_id: str) -> Callable[[str, dict], list[ChunkDict]]:
    """Return the chunking function registered for doc_id.

    Raises KeyError if doc_id is not in the registry.
    """
    registry = _load_registry()
    strategy = registry[doc_id]
    return _STRATEGY_MAP[strategy]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SAMPLE_PROSE = (
        "Workers must follow all reasonable safety instructions. "
        "The employer shall provide adequate PPE. "
        "Casual employees should be informed of conversion rights after 12 months."
    )

    SAMPLE_CLAUSES = (
        "1. General Duties\n"
        "1.1 An employer must ensure, so far as is reasonably practicable, the "
        "health and safety of workers.\n"
        "1.1.1 This includes providing safe systems of work.\n"
        "2. Consultation\n"
        "2.1 Workers must be consulted on matters affecting their health and safety."
    )

    SAMPLE_TABLE = (
        "Entitlement | Minimum | Condition\n"
        "Annual leave | 4 weeks | Full-time employees\n"
        "Personal leave | 10 days | All employees\n"
        "Parental leave | 12 months | 12 months service\n"
        "\nEmployers must meet all NES minimums regardless of award coverage."
    )

    SAMPLE_SECTIONS = (
        "INTRODUCTION\n"
        "This code applies to all workplaces in Queensland.\n\n"
        "1 SCOPE AND APPLICATION\n"
        "The code covers managing the work environment and facilities.\n\n"
        "2 DUTIES OF PCBU\n"
        "A PCBU must manage risks to health and safety."
    )

    meta = {"pages": 2}

    tests = [
        ("recursive",        chunk_recursive,        SAMPLE_PROSE),
        ("clause_boundary",  chunk_clause_boundary,  SAMPLE_CLAUSES),
        ("table_aware",      chunk_table_aware,       SAMPLE_TABLE),
        ("section_header",   chunk_section_header,    SAMPLE_SECTIONS),
        ("recommendation",   chunk_recommendation,    SAMPLE_PROSE),
    ]

    for name, fn, sample in tests:
        print(f"\n{'=' * 60}")
        print(f"Strategy: {name}  |  input length: {len(sample)} chars")
        print("=" * 60)
        results = fn(sample, meta)
        for i, chunk in enumerate(results):
            print(f"  [{i}] type={chunk['chunk_type']}  section={repr(chunk['section'])}  "
                  f"page≈{chunk['page_estimate']}")
            print(f"      text={repr(chunk['text'][:80])}")

    print("\n--- dispatch test ---")
    for doc_id in ["FD01", "FD02", "FD03", "HN01", "HN02", "HN03", "SS01", "SS02", "SS03a", "SS03b"]:
        fn = get_chunker(doc_id)
        print(f"  {doc_id} -> {fn.__name__}")
