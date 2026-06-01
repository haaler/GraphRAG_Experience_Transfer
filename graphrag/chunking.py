"""Split documents into retrieval-ready chunks.

Chunks are not embedded — retrieval is graph-primary, so chunks are reached through the 
Entity graph rather than vector search.
"""

import hashlib
import json
import os
from typing import List, Dict

from graphrag import config as default_config


# ── Chunking utilities ───────────────────────────────────────────────────
def make_chunk_id(source_file: str, section_number: int, chunk_number: int, chunk_type: str) -> str:
    """Deterministic chunk ID from a SHA-1 hash of the inputs."""
    raw = f"{source_file}|{section_number}|{chunk_number}|{chunk_type}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()

def chunk_text_by_size(
    text: str,
    max_characters: int = None,
    overlap: int = None,
    cfg=None,
) -> List[str]:
    """Split text into chunks of at most max_characters, with overlap.

    - If the text fits in one chunk, return it as-is.
    - When splitting, prefer a sentence boundary (.!?), then any whitespace, then a hard cut.
    """
    if cfg is None:
        cfg = default_config
    if max_characters is None:
        max_characters = cfg.TEXT_CHUNK_SIZE
    if overlap is None:
        overlap = cfg.TEXT_CHUNK_OVERLAP

    search_frac = getattr(cfg, "CHUNK_BREAK_SEARCH_FRAC", 0.7)

    text = text.strip()
    if not text:
        return []
    if len(text) <= max_characters:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_characters

        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        # Finding a clean place to split:
        # Prefer a sentence boundary ('.!?' followed by whitespace), fall back to any whitespace, fall back to a hard cut at 'end'.
        # The search only goes back as far as 'search_from' so we don't move the split too far from the target chunk size.
        search_from = start + int(max_characters * search_frac)
        break_at = None
        for i in range(end - 1, search_from - 1, -1):
            if text[i] in '.!?' and i + 1 < len(text) and text[i + 1] in ' \n\r\t':
                break_at = i + 1
                break

        if break_at is None:
            for i in range(end - 1, search_from - 1, -1):
                if text[i] in ' \n\r\t':
                    break_at = i
                    break

        if break_at is None:
            break_at = end

        chunks.append(text[start:break_at].strip())
        start = max(break_at - overlap, start + 1)

    return [c for c in chunks if c]

def chunk_to_text(chunk: Dict) -> str:
    """Render a chunk as plain text.

    Each chunk type has its own way of being flattened:
    - text: the raw text content
    - table: JSON-serialised rows
    - list: indented bullet format
    """
    if chunk["chunk_type"] == "text":
        return chunk["content"]["text"]
    elif chunk["chunk_type"] == "table":
        return json.dumps(chunk["content"]["rows"], ensure_ascii=False)
    elif chunk["chunk_type"] == "list":
        lines = []
        for item in chunk["content"]["items"]:
            indent = "  " * item["level"]
            lines.append(f"{indent}- {item['text']}")
        return "\n".join(lines)
    return ""


# ── Document chunking ────────────────────────────────────────────────────
def chunk_document(document: Dict, cfg=None) -> List[Dict]:
    """Build retrieval-ready chunks from a segmented document.

    - Tables → 1 atomic chunk each
    - Lists → 1 atomic chunk each
    - Paragraphs → kept whole if under TEXT_CHUNK_SIZE, else split with overlap
    """
    if cfg is None:
        cfg = default_config

    chunks = []
    chunk_number = 0
    source_file = document["source_file"]

    for section in document["sections"]:
        section_num = section["section_number"]
        section_title = section["title"]
        section_path = section.get("section_path", section_title)

        for block in section["blocks"]:

            if block["type"] == "table":
                chunk_number += 1
                chunks.append({
                    "chunk_id": make_chunk_id(source_file, section_num, chunk_number, "table"),
                    "chunk_type": "table",
                    "section_number": section_num,
                    "section_title": section_title,
                    "section_path": section_path,
                    "source_file": source_file,
                    "content": {"rows": block["rows"]},
                    "block_indices": [block["block_index"]],
                })

            elif block["type"] == "list":
                chunk_number += 1
                chunks.append({
                    "chunk_id": make_chunk_id(source_file, section_num, chunk_number, "list"),
                    "chunk_type": "list",
                    "section_number": section_num,
                    "section_title": section_title,
                    "section_path": section_path,
                    "source_file": source_file,
                    "content": {"items": block["items"]},
                    "block_indices": block["block_indices"],
                })

            elif block["type"] == "paragraph":
                text = block["text"]
                text_chunks = chunk_text_by_size(text, cfg=cfg)
                block_indices = block.get("block_indices", [block.get("block_index")])

                for txt in text_chunks:
                    chunk_number += 1
                    chunks.append({
                        "chunk_id": make_chunk_id(source_file, section_num, chunk_number, "text"),
                        "chunk_type": "text",
                        "section_number": section_num,
                        "section_title": section_title,
                        "section_path": section_path,
                        "source_file": source_file,
                        "content": {"text": txt},
                        "block_indices": block_indices,
                    })

    # Merge small adjacent text chunks in the same section
    merged = []
    max_chars = cfg.TEXT_CHUNK_SIZE
    for chunk in chunks:
        if (merged
                and chunk["chunk_type"] == "text"
                and merged[-1]["chunk_type"] == "text"
                and chunk["section_title"] == merged[-1]["section_title"]
                and len(merged[-1]["content"]["text"]) + len(chunk["content"]["text"]) + 1 <= max_chars):
            prev = merged[-1]
            prev["content"]["text"] += "\n" + chunk["content"]["text"]
            prev["block_indices"] = prev["block_indices"] + chunk["block_indices"]
        else:
            merged.append(chunk)

    document["chunks"] = merged
    return merged

def dump_chunks(documents: List[Dict], cfg=None) -> None:
    """Write each document's chunks to a JSON file in CHUNK_DUMP_FOLDER.

    Debugging helper — not called from the runtime pipeline. Useful from a notebook to inspect chunking output.
    """
    if cfg is None:
        cfg = default_config
        
    out_dir = getattr(cfg, "CHUNK_DUMP_FOLDER", "chunk_dumps_new/")
    os.makedirs(out_dir, exist_ok=True)

    for doc in documents:
        chunks = doc.get("chunks", [])
        if not chunks:
            continue
        fname = f"chunks_{doc['source_file']}.json"
        path = os.path.join(out_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"Dumped chunks for {len(documents)} documents to {out_dir}")

def chunk_all(documents: List[Dict], cfg=None) -> None:
    """Chunk every document in-place. Main entry point for the chunking phase."""
    if cfg is None:
        cfg = default_config

    total_chunks = 0
    for doc in documents:
        chunks = chunk_document(doc, cfg=cfg)
        total_chunks += len(chunks)

    print(f"Chunked {len(documents)} documents ({total_chunks} total chunks)")
