"""Load DOCX files into the shared 'DocumentData' shape.

Walks every '.docx' in the configured data folder, pulls out headers, footers, paragraphs,
and tables, parses the filename and front-page metadata, 
and finally splits the body into sections and merges runs of text and lists.
The single entry point is 'load_and_segment'.
"""

import os
import re
import json
import time
from typing import List, Dict, Any, Optional

from docx import Document as DocxDocument
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
from docx.table import Table

from graphrag import config as default_config


# ── Filename parsing ──────────────────────────────────────────────────────
def parse_filename_flexible(filename: str) -> Dict[str, str]:
    """Parse filenames in one of two formats.

      Standard    (7 dash-parts): PROJECT-SUPPLIER-CLIENT-DOCTYPE-SEQ-REV-RUNNUM_ATTACH
      Tech_Report (6 dash-parts): PROJECT-DISCIPLINE-DOCTYPE-SEQ-REV-RUNNUM_ATTACH

    The underscore separates the running number from the optional attachment suffix.
    ATTACH is everything after the underscore which is absent when the filename ends with '_'.

    The two formats are told apart by position 2 (0-indexed position 1):
    a single letter → discipline code (Tech_Report variant);
    anything else → supplier abbreviation (Standard variant).
    """
    STANDARD_FIELDS = [
        "project_number", "supplier", "client", "document_type",
        "document_sequence", "revision", "running_number",
    ]
    TECH_REPORT_FIELDS = [
        "project_number", "discipline", "document_type",
        "document_sequence", "revision", "running_number",
    ]

    name = os.path.splitext(filename)[0]

    # Split on the first underscore: everything after is the attachment (if present)
    if "_" in name:
        main, attach_raw = name.split("_", 1)
        attachment = attach_raw.strip("-_") or None
    else:
        main = name.rstrip("-")
        attachment = None

    parts = [p for p in re.split(r'-+', main) if p]
    if not parts:
        return {}

    is_discipline_variant = (len(parts) >= 2 and len(parts[1]) == 1 and parts[1].isalpha())
    fields = TECH_REPORT_FIELDS if is_discipline_variant else STANDARD_FIELDS

    metadata = {field: value for field, value in zip(fields, parts)}

    if attachment:
        metadata["attachment_number"] = attachment
    if "discipline" in metadata:
        metadata["discipline"] = metadata["discipline"].upper()

    return metadata


# ── Headers & footers ─────────────────────────────────────────────────────
def _table_lines(tbl) -> List[str]:
    """Return one ' | '-joined line per non-empty row of a DOCX table to separate values."""
    lines = []
    for row in tbl.rows:
        cells = [(cell.text or "").strip() for cell in row.cells]
        cells = [c for c in cells if c]
        if cells:
            lines.append(" | ".join(cells))
    return lines


def _extract_text_from_container(container) -> str:
    """Pull visible text out of a DOCX container (header/footer/body)."""
    parts = []
    for paragraph in getattr(container, "paragraphs", []):
        text = (paragraph.text or "").strip()
        if text:
            parts.append(text)

    for tbl in getattr(container, "tables", []):
        parts.extend(_table_lines(tbl))

    return "\n".join(parts).strip()

def extract_headers_footers(docx: DocxDocument) -> Dict[str, List[str]]:
    """Return all headers and footers across the document, deduplicated."""
    headers = []
    footers = []

    for section in docx.sections:
        for head in [
            section.header,
            getattr(section, "first_page_header", None),
            getattr(section, "even_page_header", None),
        ]:
            if head is None:
                continue
            text = _extract_text_from_container(head)
            if text:
                headers.append(text)

        for foot in [
            section.footer,
            getattr(section, "first_page_footer", None),
            getattr(section, "even_page_footer", None),
        ]:
            if foot is None:
                continue
            text = _extract_text_from_container(foot)
            if text:
                footers.append(text)

    def dedupe(seq):
        seen = set()
        result = []
        for item in seq:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    return {"headers": dedupe(headers), "footers": dedupe(footers)}


# ── Paragraph & block helpers ────────────────────────────────────────────
# Matches a word split across two lines with a trailing hyphen, e.g. "under-\nstanding".
# normalize_paragraph_text uses this to rejoin the two halves into "understanding".
HYPHEN_LINEBREAK = re.compile(r"(\w)-\s*\n\s*(\w)")

def normalize_paragraph_text(text: str) -> str:
    """Tidy paragraph text (fix hyphenated line breaks, collapse whitespace)."""
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = HYPHEN_LINEBREAK.sub(r"\1-\2", text)
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()

def get_list_level(paragraph: Paragraph) -> Optional[int]:
    """Return the list nesting level (0, 1, 2, ...) or None if not a list."""
    # A .docx file is really a zip of XML files. 
    # Word stores each paragraph as an XML element named <w:p>, and inside it puts smaller 
    # elements with their own short names (pPr, numPr, ...).
    # The "w:" part is just a Word-specific prefix on the tag names. 
    # python-docx exposes these short names as attributes, which we walk through to find the list level.
    try:
        # _p is the underlying <w:p> XML element behind the python-docx Paragraph object.
        # We drop down to it to read formatting that python-docx doesn't expose directly.
        para = paragraph._p
        # pPr is the <w:pPr> paragraph-properties element which holds styling and (for list items) numbering info. 
        # It's None when the paragraph has no explicit properties.
        properties = para.pPr
        # numPr is the <w:numPr> numbering-properties element; only present when the paragraph
        # is part of a numbered or bulleted list.
        if properties is None or properties.numPr is None:
            return None

        # ilvl is the <w:ilvl> indent-level element. 
        # Its .val is the 0-based nesting depth (0 = top-level bullet, 1 = first sub-bullet, and so on).
        indent_level = properties.numPr.ilvl
        if indent_level is None:
            return 0
        return int(indent_level.val)
    except Exception:
        return None

def iter_block_items(docx: DocxDocument):
    """Yield Paragraph and Table objects in document order."""
    body = docx.element.body
    # qn("w:p") expands the short tag "w:p" into the full namespaced form Word actually uses in the XML. 
    # We need the full form to compare against child.tag when walking the body in document order.
    for child in body.iterchildren():
        if child.tag == qn("w:p"):       # paragraph
            yield Paragraph(child, docx)
        elif child.tag == qn("w:tbl"):   # table
            yield Table(child, docx)

def _read_first_run_font(paragraph: Paragraph) -> tuple:
    """Read the font size and color of the paragraph's first character span.

    A DOCX paragraph's text is split into runs — spans of text that share formatting.
    We read the formatting of the first run as a rough proxy for the paragraph's overall look,
    which other heuristics (e.g. heading detection) can hint off.
    Returns (None, None) for empty paragraphs or when the color attribute can't be read.
    """
    if not paragraph.runs:
        return None, None

    first_run = paragraph.runs[0]
    # .size is a length object representing the font size, but we want to convert it to a simple number of points.;
    # the .pt attribute converts that to a plain number of points (e.g. 11.0, 14.0).
    font_size_pt = first_run.font.size.pt if first_run.font.size else None

    try:
        if first_run.font.color and first_run.font.color.rgb:
            font_color_hex = str(first_run.font.color.rgb)
        else:
            font_color_hex = None
    except Exception:
        font_color_hex = None

    return font_size_pt, font_color_hex


def _paragraph_to_block(
    paragraph: Paragraph,
    block_index: int,
    header_footer_fingerprints: set,
    toc_styles: set
) -> Optional[Dict]:
    """Turn one DOCX paragraph into a block dict.

    Returns None if the paragraph should be skipped 
    (empty, a duplicate of a header/footer line, or a table-of-contents entry).
    """
    text = normalize_paragraph_text(paragraph.text or "")
    if not text:
        return None
    if text in header_footer_fingerprints:
        return None

    style_name = paragraph.style.name if paragraph.style else None
    if style_name and style_name.lower() in toc_styles:
        return None

    level = get_list_level(paragraph)
    if level is None and style_name == "List Paragraph":
        level = 0

    font_size_pt, font_color_hex = _read_first_run_font(paragraph)

    return {
        "block_index": block_index,
        "type": "paragraph",
        "text": text,
        "style": style_name,
        "is_list": level is not None,
        "list_level": level,
        "font_size_pt": font_size_pt,
        "font_color_hex": font_color_hex,
    }


def _table_to_block(table: Table, block_index: int) -> Dict:
    """Turn one DOCX table into a block dict."""
    rows = []
    for row in table.rows:
        cells = [(cell.text or "").strip() for cell in row.cells]
        rows.append(cells)

    return {
        "block_index": block_index,
        "type": "table",
        "rows": rows,
    }


# ── Front-page metadata extraction ───────────────────────────────────────
# The fields the LLM is asked to extract from the front page.
# Anything else it returns is dropped.
_FRONTPAGE_METADATA_FIELDS = {"title", "date", "rig", "reference_persons"}

# Matches the first top-level {...} block in a string.
# We use this because the LLM is asked for JSON only but occasionally wraps its output in extra
# prose (this picks the JSON back out).
_JSON_OBJECT_RE = re.compile(r'\{[^{}]*\}', re.DOTALL)

_FRONTPAGE_EXTRACTION_PROMPT = """\
Extract metadata from this engineering document.

Find these fields (return "" if not found):
- title: The document title or subject. This is typically found on the frontpage as a prominent heading,
  sometimes styled as "Title" or "Front Page". 
  It may also be the first prominent text on the page, or follow a label like "Subject:" or "Title:".
  If there is both a main title and a subtitle (e.g. a second line directly below the title), 
  combine them into one string separated by " - ".
- date: The document date (any format)
- rig: The rig, installation, or vessel name. May be a full name (e.g. "Deepsea Stavanger") or an abbreviation (e.g. "DSS").
  Look in headers, project titles, and tables.
- reference_persons: Names of Odfjell personnel involved in this document.
  Look in the DOCUMENT SIGNATURES section for names under "Prepared by", "Controlled by", or "Approved by". Return as comma-separated names.

IMPORTANT:
- For rig: abbreviations are typically 3-4 capital letters found in project titles or headers.
- For reference_persons: These are the people from Odfjell who prepared, controlled, or approved the document. 
  Only include actual person names, not titles or departments.

Return ONLY valid JSON:
{{"title": "...", "date": "...", "rig": "...", "reference_persons": "..."}}

Document content:
\"\"\"
{context}
\"\"\"

JSON:"""

_FRONTPAGE_EXTRACTION_SYSTEM_MSG = (
    "You extract structured metadata from engineering documents. "
    "Output ONLY valid JSON, nothing else."
)

def _is_present(value: Optional[str]) -> bool:
    """A value counts as present only if it has actual content.

    The LLM sometimes returns the literal string '""' for fields it didn't find. 
    We treat that the same as an empty string.
    """
    if not value:
        return False
    stripped = value.strip()
    return bool(stripped) and stripped != '""'

def build_frontpage_context(blocks: List[Dict], headers: List[str], cfg=None) -> str:
    """Stitch the document's front-page content into one string for the LLM."""
    if cfg is None:
        cfg = default_config
    max_chars = getattr(cfg, "FRONTPAGE_MAX_CONTEXT_CHARS", 2500)

    parts = []
    if headers:
        parts.append("DOCUMENT HEADERS:")
        for h in headers:
            parts.append(f"  {h}")
        parts.append("")

    parts.append("DOCUMENT FRONTPAGE CONTENT:")
    block_count = 0
    for block in blocks:
        if block['type'] == 'table':
            table_lines = ["  TABLE:"]
            for row in block['rows']:
                table_lines.append("    | " + " | ".join(row))
            parts.append("\n".join(table_lines))
            block_count += 1
        elif block['type'] == 'paragraph':
            parts.append(f"  {block['text']}")
            block_count += 1
        if block_count >= 12 or len('\n'.join(parts)) > max_chars:
            break
    return '\n'.join(parts)


def find_signature_context(blocks: List[Dict], cfg=None) -> str:
    """Find the 'Document Signatures' section near the front of the document."""
    if cfg is None:
        cfg = default_config
    frontpage_limit = getattr(cfg, "FRONTPAGE_BLOCK_LIMIT", 20)

    frontpage = blocks[:frontpage_limit]
    signature_tables = []

    # Tables under a "Document Signatures" heading have one of these labels as a column header.
    # This is how we recognise them.
    signature_labels = ['prepared by', 'approved by', 'controlled by']

    for i, block in enumerate(frontpage):
        if block['type'] == 'paragraph':
            if 'signature' in block['text'].lower():
                for j in range(i + 1, min(i + 3, len(frontpage))):
                    if frontpage[j]['type'] == 'table':
                        signature_tables.append(frontpage[j])
                    elif frontpage[j]['type'] == 'paragraph' and frontpage[j]['text'].strip():
                        break
        if block['type'] == 'table':
            all_text = ' '.join(
                cell for row in block['rows'] for cell in row
            ).lower()
            if any(label in all_text for label in signature_labels):
                if block not in signature_tables:
                    signature_tables.append(block)

    if not signature_tables:
        return ""

    parts = ["DOCUMENT SIGNATURES:"]
    for tbl in signature_tables:
        for row in tbl['rows']:
            parts.append("  | " + " | ".join(row))
    return '\n'.join(parts)


def _parse_llm_metadata_response(response: str) -> Dict[str, str]:
    """Pull a JSON object out of the LLM's reply and keep only the fields we asked for."""
    match = _JSON_OBJECT_RE.search(response)
    if match is None:
        return {}

    raw = json.loads(match.group(0))
    return {
        field: value.strip()
        for field, value in raw.items()
        if field in _FRONTPAGE_METADATA_FIELDS and _is_present(value)
    }


def extract_frontpage_metadata_with_llm(blocks: List[Dict], headers: List[str], llm, cfg=None) -> Dict[str, str]:
    """Use the LLM to read metadata off the document's front page."""
    frontpage_context = build_frontpage_context(blocks, headers, cfg=cfg)
    signature_context = find_signature_context(blocks, cfg=cfg)

    context = frontpage_context
    if signature_context:
        context += "\n\n" + signature_context

    if not context.strip():
        return {}

    prompt = _FRONTPAGE_EXTRACTION_PROMPT.format(context=context)

    try:
        response = llm.invoke(
            prompt,
            system_msg=_FRONTPAGE_EXTRACTION_SYSTEM_MSG,
        )
        return _parse_llm_metadata_response(response)
    except Exception as e:
        print(f"  LLM extraction failed: {e}")
        return {}


# ── Abbreviation table extraction ─────────────────────────────────────────
# Matches the word 'abbreviation' or 'abbreviations' on a word boundary.
# Used to spot paragraph headings like "Abbreviations" or "List of Abbreviations".
_ABBREV_TITLE_RE = re.compile(r'\babbreviations?\b', re.IGNORECASE)

def _is_abbreviation_heading(block: Dict) -> bool:
    """True if the block is a paragraph titled 'Abbreviations' or similar."""
    return (
        block["type"] == "paragraph" and bool(_ABBREV_TITLE_RE.search(block.get("text", "")))
    )

def _find_table_after_heading(blocks: List[Dict], start: int, lookahead: int = 6) -> Optional[Dict]:
    """Look forward from 'start' for the next table, skipping blank paragraphs.

    Stops (returning None) if a non-blank paragraph appears before any table.
    This means the abbreviation heading isn't directly followed by its table.
    """
    end = min(start + lookahead, len(blocks))
    for j in range(start, end):
        candidate = blocks[j]
        if candidate["type"] == "table":
            return candidate
        if candidate["type"] == "paragraph" and candidate.get("text", "").strip():
            return None
    return None

def _parse_abbreviation_table(table_block: Dict) -> Dict[str, str]:
    """Read abbreviation/meaning pairs out of a two-column table block."""
    pairs: Dict[str, str] = {}
    for row in table_block["rows"]:
        if len(row) < 2:
            continue
        abbr = row[0].strip().upper()
        meaning = row[1].strip()
        if abbr and meaning:
            pairs[abbr] = meaning
    return pairs

def extract_abbreviations(blocks: List[Dict]) -> Dict[str, str]:
    """Pull abbreviation → meaning pairs from sections titled 'Abbreviations'.

    Looks for a paragraph whose text matches the title (e.g. 'Abbreviations', 'List of Abbreviations') 
    and reads the immediately following table.
    The table is expected to have two columns: abbreviation and meaning. Rows with an empty first column are skipped.

    Returns a dict {ABBREVIATION: 'meaning'} with upper-cased keys.
    """
    abbrevs: Dict[str, str] = {}
    for i, block in enumerate(blocks):
        if not _is_abbreviation_heading(block):
            continue
        table = _find_table_after_heading(blocks, i + 1)
        if table is not None:
            abbrevs.update(_parse_abbreviation_table(table))
    return abbrevs


# ── Section detection & segmentation ─────────────────────────────────────
# Fallback heading detector, used only when a document has no styled headings to lean on.
# Matches a whole paragraph that looks like one of:
#   - a dot-numbered heading: "1 Scope", "1.2 Findings", "1.2.3 Details"
#   - an appendix label:       "Appendix A", "Appendix B1 Diagrams"
#   - a section label:         "Section 1", "Section 12 Notes"
HEADING_REGEX = re.compile(
    r"""
    ^(
        (\d+(\.\d+)*\s+.+) |
        (appendix\s+[a-z0-9]+.*) |
        (section\s+\d+.*)
    )$
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Pulls a leading dot-separated number off a heading (e.g. '1.2.3 Title')
_NUMBERED_HEADING_RE = re.compile(r'^(\d+(?:\.\d+)*)\s')

def _get_heading_level(block: Dict) -> Optional[int]:
    """Return the heading level (1, 2, 3, ...) from the DOCX style name.

    Returns 0 for Title style, None for non-headings.
    """
    if block["type"] != "paragraph":
        return None
    style = (block.get("style") or "").lower()
    if style == "title":
        return 0
    if style.startswith("heading"):
        try:
            return int(style.split()[-1])
        except (ValueError, IndexError):
            return 1
    return None

def _get_regex_heading_depth(block: Dict) -> Optional[int]:
    """Guess heading depth from dot-separated numbering ('1.2.3' → 3).

    Returns None if the block is not a regex-detected heading.
    """
    if block["type"] != "paragraph":
        return None
    text = (block.get("text") or "").strip()
    if not text or not HEADING_REGEX.match(text):
        return None
    m = _NUMBERED_HEADING_RE.match(text)
    if m:
        return m.group(1).count('.') + 1  # '1' → 1, '1.2' → 2, '1.2.3' → 3
    return 1  # 'Appendix A' / 'Section 1' → depth 1

def segment_into_sections(document: Dict) -> List[Dict]:
    """Split the ordered blocks into sections based on headings.

    Every heading starts a new section, but each section also tracks its place in the heading hierarchy
    via heading_level, parent_section_number, and section_path.

    section_path is a breadcrumb string like 'Scope of Work > BOP Stack > Findings' built from the heading chain.
    This gives the entity-extraction LLM the full context of where a chunk sits in the document outline 
    without merging sections together.

    When no styled headings are present, it falls back to regex detection with depth read off any 
    dot-separated numbering at the start of the heading.
    """
    blocks = document["blocks"]

    has_styled_headings = any(
        b["type"] == "paragraph"
        and (b.get("style") or "").lower().startswith("heading")
        for b in blocks
    )
    use_regex = not has_styled_headings

    sections = []
    current_section = None
    section_counter = 0

    # Stack of (section_number, heading_level, title) tracking the current heading chain.
    heading_stack: List[tuple] = []

    def start_new_section(title: str, level: int) -> Dict:
        nonlocal section_counter
        section_counter += 1

        # Level 0 (Title style) is pre-content (e.g. "Distribution"), which we treat as a standalone section that 
        # never becomes a parent of H1+ content.
        if level < 1:
            heading_stack.clear()
            return {
                "section_number": section_counter,
                "title": title,
                "heading_level": level,
                "parent_section_number": None,
                "section_path": title,
                "blocks": [],
            }

        # Pop the stack until we find a parent at a strictly higher level.
        while heading_stack and heading_stack[-1][1] >= level:
            heading_stack.pop()

        # Build section_path from the current heading chain + the new title.
        parent_titles = [entry[2] for entry in heading_stack]
        section_path = " > ".join(parent_titles + [title])

        parent_num = heading_stack[-1][0] if heading_stack else None

        heading_stack.append((section_counter, level, title))

        return {
            "section_number": section_counter,
            "title": title,
            "heading_level": level,
            "parent_section_number": parent_num,
            "section_path": section_path,
            "blocks": [],
        }

    for block in blocks:
        if not use_regex:
            level = _get_heading_level(block)
        else:
            level = _get_regex_heading_depth(block)

        if level is not None:
            if current_section:
                sections.append(current_section)
            current_section = start_new_section(block["text"], level)
            continue

        if current_section is None:
            current_section = start_new_section("UNSECTIONED", 0)
        current_section["blocks"].append(block)

    if current_section:
        sections.append(current_section)
    return sections


# ── List grouping & text merging ──────────────────────────────────────────
def _can_bridge_list_gap(block: Dict, blocks: List[Dict], index: int) -> bool:
    """Decide whether a non-list paragraph between list items should be bridged."""
    if block["type"] != "paragraph":
        return False
    style = (block.get("style") or "").lower()
    if style.startswith("heading") or style == "title":
        return False
    for j in range(index + 1, min(index + 3, len(blocks))):
        if blocks[j]["type"] == "paragraph" and blocks[j].get("is_list"):
            return True
    return False

def group_lists_in_section(section: Dict) -> None:
    """Merge consecutive list paragraphs into a single list block."""
    new_blocks = []
    buffer = []

    def flush_buffer():
        nonlocal buffer
        if buffer:
            list_block = {
                "type": "list",
                "items": [
                    {
                        "text": b["text"],
                        "level": b.get("list_level") or 0,
                        "block_index": b["block_index"],
                    } for b in buffer
                ],
                "block_indices": [b["block_index"] for b in buffer],
            }
            new_blocks.append(list_block)
            buffer = []

    blocks = section["blocks"]
    i = 0
    while i < len(blocks):
        b = blocks[i]
        if b["type"] == "paragraph" and b.get("is_list"):
            buffer.append(b)
        elif buffer and _can_bridge_list_gap(b, blocks, i):
            buffer.append(b)
        else:
            flush_buffer()
            new_blocks.append(b)
        i += 1

    flush_buffer()
    section["blocks"] = new_blocks

def merge_consecutive_text_in_section(section: Dict) -> None:
    """Merge consecutive text paragraphs into a single text block."""
    new_blocks = []
    text_buffer = []

    def flush_text():
        nonlocal text_buffer
        if text_buffer:
            merged_block = {
                "type": "paragraph",
                "text": "\n".join(b["text"] for b in text_buffer),
                "block_indices": [b["block_index"] for b in text_buffer],
            }
            new_blocks.append(merged_block)
            text_buffer = []

    for block in section["blocks"]:
        if block["type"] == "paragraph":
            text_buffer.append(block)
        else:
            flush_text()
            new_blocks.append(block)

    flush_text()
    section["blocks"] = new_blocks

# ── Per-document loader & main entry point ───────────────────────────────
def _load_one_document(filename: str, file_path: str, cfg, llm) -> Dict:
    """Load and segment one DOCX file into a DocumentData dict."""
    toc_styles = getattr(cfg, "TOC_STYLES", {"toc heading", "toc 1", "toc 2", "toc 3", "toc 4"}) # Table of Contents styles to ignore as headings
    docx = DocxDocument(file_path)

    # ── Headers, footers, and the fingerprint set used to drop duplicates ──
    hf = extract_headers_footers(docx)
    headers = hf["headers"]
    footers = hf["footers"]
    header_footer_fingerprints = set(headers + footers)

    # ── Walk the body and convert each paragraph/table into a block dict ──
    blocks: List[Dict[str, Any]] = []
    for raw_block in iter_block_items(docx):
        if isinstance(raw_block, Paragraph):
            converted = _paragraph_to_block(raw_block, len(blocks), header_footer_fingerprints, toc_styles)
            if converted is not None:
                blocks.append(converted)
        elif isinstance(raw_block, Table):
            blocks.append(_table_to_block(raw_block, len(blocks)))

    # ── Metadata: filename, LLM front-page extraction, abbreviations ─────
    file_metadata = parse_filename_flexible(filename)
    llm_metadata = (
        extract_frontpage_metadata_with_llm(blocks, headers, llm, cfg=cfg)
        if llm is not None else {}
    )
    abbreviations = extract_abbreviations(blocks)

    doc = {
        "source_file": filename,
        **file_metadata,
        **llm_metadata,
        "abbreviations": abbreviations,
        "headers": headers,
        "footers": footers,
        "blocks": blocks,
    }

    # ── Per-document log line ────────────────────────────────────────────
    llm_fields = [f for f in ["date", "rig", "reference_persons"] if f in llm_metadata]
    print(
        f"   {len(blocks)} blocks | "
        f"filename: {len(file_metadata)} fields | "
        f"LLM: {llm_fields} | "
        f"abbrevs: {len(abbreviations)}"
    )

    # ── Sections and within-section grouping ─────────────────────────────
    doc["sections"] = segment_into_sections(doc)
    for section in doc["sections"]:
        group_lists_in_section(section)
        merge_consecutive_text_in_section(section)

    return doc

def load_and_segment(cfg=None, llm=None) -> List[Dict]:
    """Load every DOCX in the data folder and return a list of DocumentData dicts.

    This is the single entry point for DOCX ingestion.
    """
    if cfg is None:
        cfg = default_config

    data_folder = cfg.DATA_FOLDER

    documents = []
    loader_start = time.time()

    for filename in os.listdir(data_folder):
        if not filename.lower().endswith(".docx"):
            continue

        file_path = os.path.join(data_folder, filename)
        doc_start = time.time()
        print(f"Loading {filename}")

        doc = _load_one_document(filename, file_path, cfg, llm)
        doc_elapsed = time.time() - doc_start
        print(f"   {doc_elapsed:.1f}s")

        documents.append(doc)

    total_elapsed = time.time() - loader_start
    print(
        f"\nLoaded {len(documents)} DOCX documents in {total_elapsed:.0f}s (avg {total_elapsed/max(len(documents),1):.1f}s/doc)"
    )
    return documents
