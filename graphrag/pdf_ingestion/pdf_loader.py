"""PDF document ingestion: load, segment, and fingerprint PDF files.

Uses gmft (PyPDFium2) for PDF parsing and table detection. 
Intentionally made as a parallel of docx_loader.py where the returned DocumentData dict has the same
shape so chunking, graph build, entity extraction, and retrieval do not need to branch on input format.

The DOCX path is the production one. 
This file is kept as a functional (but not optimal and generally deprecated) alternative 
for runs where 'pdf' is included in config.DOCUMENT_FORMATS.

Entry point: load_and_segment(cfg, llm) -> List[DocumentData]
"""

import os
import re
import math
import time
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import Counter

from gmft.pdf_bindings import PyPDFium2Document
from gmft.auto import AutoTableDetector, AutoTableFormatter, AutoFormatConfig

from graphrag import config as default_config
from graphrag.ingestion.docx_loader import (
    parse_filename_flexible,
    extract_frontpage_metadata_with_llm,
    segment_into_sections,
    group_lists_in_section,
    merge_consecutive_text_in_section,
    HYPHEN_LINEBREAK,
)


# ── Text repair prompt ───────────────────────────────────────────────────
REPAIR_PROMPT = """You are given text extracted from a PDF document.
The text may have broken or scrambled word order because the PDF reader extracted words in the wrong sequence (e.g., multi-column layouts read left-to-right instead of column-by-column).

Rules:
- Reconstruct the intended reading order of the text
- Fix broken sentences where words are clearly in the wrong order
- Do NOT add information that is not present in the original text
- Do NOT remove any information
- Do NOT summarize or paraphrase
- Preserve list markers (bullets, numbers) and their structure
- Keep form-style fields as-is (e.g., "Name: John Smith")
- If the text is too garbled to reconstruct confidently, return it unchanged

Return only the repaired text."""


# ── List item detection ──────────────────────────────────────────────────
LIST_ITEM_REGEX = re.compile(
    r"^(?:"
    r"[•\-–—►▪▸●○◆]\s+"            # bullet markers
    r"|\(?[0-9]+[.):\-]\s+"         # numbered: 1. 1) 1: (1)
    r"|\(?[a-zA-Z][.):\-]\s+"       # lettered: a. a) (a)
    r")"
)


def looks_like_list_item(text: str) -> bool:
    """Detect list items by bullet/number marker only."""
    return bool(LIST_ITEM_REGEX.match(text.strip()))


# ── PDF text extraction helpers ──────────────────────────────────────────
def _normalize_text(text) -> str:
    """Ensure text is a string."""
    if isinstance(text, str):
        return text
    if isinstance(text, list):
        return " ".join(map(str, text))
    return str(text)


def _normalize_pdf_paragraph(text: str) -> str:
    """Normalize PDF paragraph text: fix hyphenated line breaks, collapse whitespace."""
    if not text:
        return ""
    text = text.replace("\r", "\n")
    text = HYPHEN_LINEBREAK.sub(r"\1-\2", text)
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text.strip()


def _is_inside_any_table(bbox, table_bboxes, margin: float = 5) -> bool:
    """Check if a text block's center falls inside any table bounding box."""
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    for tx0, ty0, tx1, ty1 in table_bboxes:
        if tx0 - margin <= cx <= tx1 + margin and ty0 - margin <= cy <= ty1 + margin:
            return True
    return False


def extract_page_text(page, table_bboxes=None, exclude_texts=None, bbox_margin: float = 5) -> List[Dict]:
    """Extract word-level text blocks, filtering table regions and header/footer text."""
    blocks = []
    for x0, y0, x1, y1, text in page.get_positions_and_text():
        text = _normalize_text(text).strip()
        if not text:
            continue
        if table_bboxes and _is_inside_any_table((x0, y0, x1, y1), table_bboxes, margin=bbox_margin):
            continue
        if exclude_texts and text in exclude_texts:
            continue
        blocks.append({"bbox": (x0, y0, x1, y1), "text": text})
    return blocks


def detect_headers_footers(doc, threshold: float = 0.4) -> Tuple[Set[str], List[Dict[str, Any]]]:
    """Scan all pages to find recurring text (headers/footers).

    Returns:
        (filtered_set, details_list) where details_list contains {"text", "count", "fraction"} 
        for every filtered item — useful for inspection in the notebook.
    """
    text_page_counts = Counter()
    num_pages = len(doc)

    for page in doc:
        seen_on_page = set()
        for x0, y0, x1, y1, text in page.get_positions_and_text():
            text = _normalize_text(text).strip()
            # 3-char minimum filters out page-number noise that would otherwise dominate the recurring-text counts.
            if text and len(text) >= 3 and text not in seen_on_page:
                seen_on_page.add(text)
                text_page_counts[text] += 1

    # We've set a floor of 2 so a single-page hit on a tiny PDF is never classified as recurring.
    min_count = max(2, int(num_pages * threshold))
    filtered = {text for text, count in text_page_counts.items() if count >= min_count}

    details = [
        {"text": text, "count": count, "fraction": round(count / num_pages, 2)}
        for text, count in text_page_counts.most_common()
        if count >= min_count
    ]

    return filtered, details


# ── Line and paragraph grouping ──────────────────────────────────────────
def _blocks_to_lines(blocks: List[Dict], line_threshold: float = 5) -> List[List[Dict]]:
    """Group word-level blocks into x-sorted lines based on y-proximity."""
    if not blocks:
        return []
    sorted_blocks = sorted(blocks, key=lambda b: (b["bbox"][1], b["bbox"][0]))
    lines = []
    current_line = [sorted_blocks[0]]

    for b in sorted_blocks[1:]:
        anchor_y = current_line[0]["bbox"][1]
        if abs(b["bbox"][1] - anchor_y) <= line_threshold:
            current_line.append(b)
        else:
            current_line.sort(key=lambda bl: bl["bbox"][0])
            lines.append(current_line)
            current_line = [b]

    if current_line:
        current_line.sort(key=lambda bl: bl["bbox"][0])
        lines.append(current_line)
    return lines


def _line_text(line: List[Dict]) -> str:
    """Join blocks in a line into text."""
    return " ".join(b["text"].strip() for b in line if b["text"].strip())


def _blocks_to_paragraphs(
    blocks: List[Dict],
    y_threshold: float = 12,
    line_threshold: float = 5,
) -> List[List[Dict]]:
    """Group text blocks into paragraphs via line-based grouping.

    Lines starting with list markers always start a new paragraph.
    """
    lines = _blocks_to_lines(blocks, line_threshold)
    paragraphs = []
    current_blocks = []
    last_line_y = None

    for line in lines:
        text = _line_text(line)
        min_y = min(b["bbox"][1] for b in line)
        is_list = looks_like_list_item(text)

        if last_line_y is None:
            current_blocks.extend(line)
        elif is_list:
            if current_blocks:
                paragraphs.append(current_blocks)
            current_blocks = list(line)
        elif abs(min_y - last_line_y) < y_threshold:
            current_blocks.extend(line)
        else:
            if current_blocks:
                paragraphs.append(current_blocks)
            current_blocks = list(line)
        last_line_y = min_y

    if current_blocks:
        paragraphs.append(current_blocks)
    return paragraphs


# ── Structured block splitting ───────────────────────────────────────────
def _split_into_structured_blocks(paragraphs: List[List[Dict]]) -> List[Dict]:
    """Split paragraphs into text blocks and list blocks."""
    structured = []
    current_list = []

    def flush_list():
        nonlocal current_list
        if not current_list:
            return
        # A single bullet point stays a text block.
        # Only two or more consecutive bullet lines become a real list block.
        if len(current_list) >= 2:
            structured.append({"type": "list", "items": current_list})
        else:
            structured.append({"type": "text", "text": current_list[0]["text"]})
        current_list = []

    for para in paragraphs:
        if not para:
            continue
        text = " ".join(
            _normalize_text(p["text"]).strip()
            for p in para
            if _normalize_text(p["text"]).strip()
        )
        if not text:
            continue

        x0 = para[0]["bbox"][0]

        if looks_like_list_item(text):
            current_list.append({"text": text, "x0": x0})
        else:
            flush_list()
            structured.append({"type": "text", "text": text})

    flush_list()
    return structured


def _infer_list_levels(items: List[Dict], indent_threshold: float = 8) -> List[Dict]:
    """Estimate list nesting depth from x-offset."""
    if not items:
        return items
    base_x = min(i["x0"] for i in items)
    for item in items:
        delta = item["x0"] - base_x
        item["level"] = max(0, int(round(delta / indent_threshold)))
    return items


# ── Text repair ──────────────────────────────────────────────────────────
BATCH_REPAIR_PROMPT = """You are given numbered text blocks extracted from a PDF.
Each block may have broken or scrambled word order from incorrect PDF extraction.

Rules:
- Reconstruct the intended reading order of EACH block independently
- Fix broken sentences where words are clearly in the wrong order
- Do NOT add information that is not present
- Do NOT remove any information
- Do NOT summarize or paraphrase
- Keep form-style fields as-is (e.g., "Name: John Smith")
- If a block is too garbled, return it unchanged

Return ALL blocks in the EXACT same numbered format:
[1] repaired text
[2] repaired text
...
"""

def _parse_batch_response(response: str, count: int) -> List[Optional[str]]:
    """Parse numbered [1] ... [2] ... response back into individual texts."""
    results = [None] * count
    # re.split with a capture group interleaves separators and content, so
    # the result looks like ['', '1', 'text1', '2', 'text2', ...]. 
    # We start at index 1 and step by 2 to pair each "[N]" index marker with the
    # text that immediately follows it.
    parts = re.split(r'\[(\d+)\]\s*', response)
    for i in range(1, len(parts) - 1, 2):
        try:
            idx = int(parts[i]) - 1  # 1-based to 0-based
            if 0 <= idx < count:
                results[idx] = parts[i + 1].strip()
        except (ValueError, IndexError):
            continue
    return results


def _batch_repair_texts(
    texts: List[str], llm, min_words: int = 5, repair_log: List[Dict] = None
) -> List[str]:
    """Repair multiple text blocks in batched LLM calls.

    Groups texts into batches up to PDF_REPAIR_BATCH_MAX_CHARS, sending one LLM call per batch. 
    Falls back to original text for any block that fails validation.
    """
    if llm is None or not texts:
        return texts

    batch_max_chars = getattr(default_config, "PDF_REPAIR_BATCH_MAX_CHARS", 3000)

    # Build index of texts that need repair vs those that are too short
    needs_repair = []  # (original_index, text)
    results = list(texts)  # start with originals

    for i, text in enumerate(texts):
        if len(text.split()) >= min_words:
            needs_repair.append((i, text))

    if not needs_repair:
        return results

    # Group into batches
    batches = []  # List of List[(original_index, text)]
    current_batch = []
    current_len = 0

    for idx, text in needs_repair:
        entry_len = len(text) + 10  # overhead for the prefix
        if current_batch and current_len + entry_len > batch_max_chars:
            batches.append(current_batch)
            current_batch = []
            current_len = 0
        current_batch.append((idx, text))
        current_len += entry_len

    if current_batch:
        batches.append(current_batch)

    # Process each batch
    for batch in batches:
        # Build numbered prompt
        numbered_lines = []
        for pos, (_, text) in enumerate(batch):
            numbered_lines.append(f"[{pos + 1}] {text}")
        prompt = BATCH_REPAIR_PROMPT + "\n" + "\n".join(numbered_lines)

        try:
            response = llm.invoke(
                prompt,
                system_msg="You repair scrambled PDF text. Return ALL numbered blocks.",
            )
            parsed = _parse_batch_response(response, len(batch))
        except Exception:
            parsed = [None] * len(batch)

        # Validate and apply each repaired block
        for pos, (original_idx, original_text) in enumerate(batch):
            repaired = parsed[pos]
            original_words = original_text.split()

            if repaired and repaired.strip():
                repaired = repaired.strip()
                repaired_words = repaired.split()

                # Revert if the repair lost more than roughly 20% of the words.
                # That usually means the LLM summarised or paraphrased
                # instead of just reordering, which is not what we asked for.
                if len(repaired_words) < len(original_words) * 0.8:
                    if repair_log is not None:
                        repair_log.append({
                            "status": "reverted",
                            "original": original_text,
                            "repaired": repaired,
                            "reason": f"word count dropped {len(original_words)} -> {len(repaired_words)}",
                        })
                elif repaired != original_text:
                    results[original_idx] = repaired
                    if repair_log is not None:
                        repair_log.append({
                            "status": "applied",
                            "original": original_text,
                            "repaired": repaired,
                        })
            # If parsed[pos] is None, results[original_idx] stays as original

    return results


# ── Table detection ──────────────────────────────────────────────────────
# gmft's detector and formatter are expensive to construct, so we build them
# once per Python process and reuse the same instances across every PDF in the run.
_table_detector = None
_table_formatter = None


def _get_table_tools():
    """Lazily initialize and cache gmft table detector/formatter."""
    global _table_detector, _table_formatter
    if _table_detector is None:
        _table_detector = AutoTableDetector()
        table_config = AutoFormatConfig()
        table_config.semantic_spanning_cells = True
        table_config.enable_multi_header = True
        # Silence gmft's per-table progress prints.
        table_config.verbosity = 0
        _table_formatter = AutoTableFormatter(table_config)
    return _table_detector, _table_formatter


# ── Table conversion ─────────────────────────────────────────────────────
def _sanitize_value(val) -> str:
    """Convert a cell value to string, replacing NaN with empty string."""
    if isinstance(val, float) and math.isnan(val):
        return ""
    return str(val).strip() if val is not None else ""


def _convert_table(table, formatter, filename: str = "", page_no: int = -1) -> Optional[Dict]:
    """Convert a gmft table to a block dict matching the pipeline format.

    Logs warnings on failure and validates extracted content.
    """
    try:
        ft = formatter.extract(table)
        df = ft.df()
        if df.empty:
            print(f"  WARNING: Empty table on page {page_no} of {filename}")
            return None
        header_row = [_sanitize_value(c) for c in df.columns.tolist()]
        data_rows = [
            [_sanitize_value(cell) for cell in row]
            for row in df.values.tolist()
        ]
        rows = [header_row] + data_rows

        # Validate: reject if all cells are empty
        all_cells = [cell for row in rows for cell in row]
        if all(c == "" for c in all_cells):
            print(f"  WARNING: All-empty table on page {page_no} of {filename}")
            return None

        return {"type": "table", "rows": rows}
    except Exception as e:
        print(f"  WARNING: Table extraction failed on page {page_no} of {filename}: {e}")
        return None


# ── Single PDF processing ────────────────────────────────────────────────
def process_single_pdf(file_path: str, cfg, llm=None) -> Dict[str, Any]:
    """Process a single PDF file into a DocumentData dict.

    Two-pass approach:
      Pass 1: Detect recurring header/footer text
      Pass 2: Per-page extraction → flat block list

    Stores processing stats and inspection data on the returned dict
    under keys prefixed with '_pdf_'. 
    The prefix gets ignored by downstream modules.
    """
    filename = os.path.basename(file_path)

    # ── Read config thresholds ───────────────────────────────────────────
    hf_threshold = getattr(cfg, "PDF_HEADER_FOOTER_THRESHOLD", 0.4)
    enable_repair = getattr(cfg, "ENABLE_PDF_TEXT_REPAIR", True)
    line_threshold = getattr(cfg, "PDF_LINE_THRESHOLD", 5)
    para_threshold = getattr(cfg, "PDF_PARAGRAPH_THRESHOLD", 12)
    indent_threshold = getattr(cfg, "PDF_LIST_INDENT_THRESHOLD", 8)
    bbox_margin = getattr(cfg, "PDF_TABLE_BBOX_MARGIN", 5)
    repair_min_words = getattr(cfg, "PDF_REPAIR_MIN_WORDS", 5)

    # ── Stats tracking ───────────────────────────────────────────────────
    stats = {
        "pages": 0,
        "blocks_text": 0,
        "blocks_list": 0,
        "blocks_table": 0,
        "tables_detected": 0,
        "tables_extracted": 0,
        "tables_failed": 0,
        "repairs_attempted": 0,
        "repairs_applied": 0,
        "repairs_reverted": 0,
        "headers_filtered": 0,
        "sections": 0,
    }
    repair_log = []  # [{status, original, repaired, reason?}, ...]

    # ── Two passes: header detection across the whole PDF, then per-page extraction ──
    doc = PyPDFium2Document(file_path)
    stats["pages"] = len(doc)

    # Header/footer detection (scans all pages)
    header_footer_texts, hf_details = detect_headers_footers(doc, threshold=hf_threshold)
    stats["headers_filtered"] = len(header_footer_texts)

    if header_footer_texts:
        print(f"  Filtered {len(header_footer_texts)} header/footer patterns")

    # Reuse shared table detector/formatter
    table_detector, table_formatter = _get_table_tools()

    # Extract all structured blocks (no LLM calls)
    # Each entry: ("text", normalized_text) | ("list", items) | ("table_block", converted_dict)
    raw_blocks = []

    for page_no, page in enumerate(doc):

        # 1) Detect tables for bbox filtering
        tables_on_page = table_detector.extract(page)
        table_bboxes = [t.bbox for t in tables_on_page]
        stats["tables_detected"] += len(tables_on_page)

        # 2) Extract text, filtering table regions and headers/footers
        text_blocks = extract_page_text(
            page,
            table_bboxes=table_bboxes,
            exclude_texts=header_footer_texts,
            bbox_margin=bbox_margin,
        )

        # 3) Build paragraphs
        paragraphs = _blocks_to_paragraphs(
            text_blocks,
            y_threshold=para_threshold,
            line_threshold=line_threshold,
        )

        # 4) Split into structured blocks (text vs list)
        structured = _split_into_structured_blocks(paragraphs)

        # 5) Collect raw blocks (text repair deferred)
        for sblock in structured:
            if sblock["type"] == "text":
                text = _normalize_pdf_paragraph(sblock["text"])
                if text.strip():
                    raw_blocks.append(("text", text))
            elif sblock["type"] == "list":
                items = _infer_list_levels(sblock["items"], indent_threshold=indent_threshold)
                raw_blocks.append(("list", items))

        # 6) Convert tables now (while doc is still open)
        for table in tables_on_page:
            table_block = _convert_table(table, table_formatter, filename=filename, page_no=page_no)
            if table_block is not None:
                raw_blocks.append(("table_block", table_block))
                stats["tables_extracted"] += 1
            else:
                stats["tables_failed"] += 1

    doc.close()

    # ── Batch text repair (single set of LLM calls) ─────────────────────
    if enable_repair:
        # Gather all text blocks that need repair
        text_indices = [i for i, raw_block in enumerate(raw_blocks) if raw_block[0] == "text"]
        text_originals = [raw_blocks[i][1] for i in text_indices]
        stats["repairs_attempted"] = len(text_originals)

        repaired_texts = _batch_repair_texts(
            text_originals, llm, min_words=repair_min_words, repair_log=repair_log,
        )

        # Write repaired texts back
        for idx, repaired in zip(text_indices, repaired_texts):
            raw_blocks[idx] = ("text", repaired)

    # ── Build final block list ────────────────────────────────────────────
    all_blocks = []
    block_index = 0

    for raw_block in raw_blocks:
        if raw_block[0] == "text":
            text = raw_block[1]
            if text.strip():
                all_blocks.append({
                    "block_index": block_index,
                    "type": "paragraph",
                    "text": text,
                    "style": None,
                    "is_list": False,
                    "list_level": None,
                    "font_size_pt": None,
                    "font_color_hex": None,
                })
                block_index += 1
                stats["blocks_text"] += 1

        elif raw_block[0] == "list":
            items = raw_block[1]
            for item in items:
                all_blocks.append({
                    "block_index": block_index,
                    "type": "paragraph",
                    "text": _normalize_pdf_paragraph(item["text"]),
                    "style": None,
                    "is_list": True,
                    "list_level": item.get("level", 0),
                    "font_size_pt": None,
                    "font_color_hex": None,
                })
                block_index += 1
                stats["blocks_list"] += 1

        elif raw_block[0] == "table_block":
            table_block = raw_block[1]
            table_block["block_index"] = block_index
            all_blocks.append(table_block)
            block_index += 1
            stats["blocks_table"] += 1

    # ── Repair stats ─────────────────────────────────────────────────────
    stats["repairs_applied"] = sum(1 for r in repair_log if r["status"] == "applied")
    stats["repairs_reverted"] = sum(1 for r in repair_log if r["status"] == "reverted")

    # ── Metadata ─────────────────────────────────────────────────────────
    headers = list(header_footer_texts)
    footers = []

    file_metadata = parse_filename_flexible(filename)

    llm_metadata = {}
    if llm is not None:
        llm_metadata = extract_frontpage_metadata_with_llm(all_blocks, headers, llm)

    doc_data = {
        "source_file": filename,
        **file_metadata,
        **llm_metadata,
        "headers": headers,
        "footers": footers,
        "blocks": all_blocks,
    }

    # ── Segmentation ─────────────────────────────────────────────────────
    doc_data["sections"] = segment_into_sections(doc_data)
    for section in doc_data["sections"]:
        group_lists_in_section(section)
        merge_consecutive_text_in_section(section)

    stats["sections"] = len(doc_data["sections"])

    # ── Store inspection data ────────────────────────────────────────────
    doc_data["_pdf_processing_stats"] = stats
    doc_data["_pdf_header_footer_details"] = hf_details
    doc_data["_pdf_repair_log"] = repair_log

    # ── Print summary ────────────────────────────────────────────────────
    total_blocks = stats["blocks_text"] + stats["blocks_list"] + stats["blocks_table"]
    print(
        f"  {stats['pages']} pages | "
        f"{total_blocks} blocks ({stats['blocks_text']} text, {stats['blocks_list']} list, {stats['blocks_table']} table) | "
        f"{stats['repairs_applied']} repaired"
        + (f", {stats['repairs_reverted']} reverted" if stats["repairs_reverted"] else "")
        + f" | {stats['sections']} sections | "
        f"{stats['headers_filtered']} headers filtered"
    )
    if stats["tables_failed"]:
        print(f"  WARNING: {stats['tables_failed']}/{stats['tables_detected']} tables failed extraction")

    llm_fields = [f for f in ["date", "rig", "reference_persons"] if f in llm_metadata]
    print(f"  Metadata — filename: {len(file_metadata)} fields | LLM: {llm_fields}")

    return doc_data


# ── Main entry point ─────────────────────────────────────────────────────
def load_and_segment(cfg=None, llm=None) -> List[Dict]:
    """
    Load all PDF documents, extract metadata, segment into sections,
    group lists, merge text, and build fingerprints.

    This is the single entry point for PDF ingestion.

    Args:
        cfg: Config module (defaults to graphrag.config)
        llm: LLM instance for frontpage metadata extraction and text repair

    Returns:
        List of DocumentData dicts ready for chunking.
    """
    if cfg is None:
        cfg = default_config

    data_folder = getattr(cfg, "PDF_DATA_FOLDER", "bids_pdf_data/")

    pdf_files = [
        f for f in os.listdir(data_folder)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print("No PDF files found in", data_folder)
        return []

    documents = []
    loader_start = time.time()

    for filename in pdf_files:
        file_path = os.path.join(data_folder, filename)
        doc_start = time.time()
        print(f"Loading {filename}")
        doc_data = process_single_pdf(file_path, cfg, llm)
        doc_elapsed = time.time() - doc_start
        print(f"  -> {doc_elapsed:.1f}s")
        documents.append(doc_data)

    total_elapsed = time.time() - loader_start
    avg = total_elapsed / len(documents) if documents else 0
    print(f"\nLoaded {len(documents)} PDF documents in {total_elapsed:.0f}s (avg {avg:.1f}s/doc)")
    return documents
