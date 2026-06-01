"""Shared shape every document moves through the pipeline with.

Loaders (DOCX, PDF, ...) return a list of 'DocumentData' dicts. 
Each contains a list of 'SectionData' (its sections) and, after the chunking step,
a list of 'ChunkData' (the smaller pieces sent to the LLM).
The rest of the pipeline reads from this shape and doesn't care which loader produced it.
"""
from typing import TypedDict, List, Dict, Any, Optional

# All three classes use 'total=False' so loaders can build them up field-by-field (not every field is set at every stage).

class ChunkData(TypedDict, total=False):
    """One unit of content from a document — a paragraph of text, a table, or a list.
    Produced by the chunking step.
    """
    chunk_id: str
    chunk_type: str          # "text" | "table" | "list"
    section_number: int
    section_title: str
    section_path: str        # for example: "Scope of Work > BOP Stack > Findings"
    source_file: str
    content: Dict[str, Any]  # {"text": ...} or {"rows": ...} or {"items": ...}
    block_indices: List[int]


class SectionData(TypedDict, total=False):
    """One section of a document, with its heading, its place in the section tree,
    and the blocks of content it contains.
    """
    section_number: int
    title: str
    heading_level: int
    parent_section_number: Optional[int]
    section_path: str               # for example: "Scope of Work > BOP Stack > Findings"
    blocks: List[Dict[str, Any]]


class DocumentData(TypedDict, total=False):
    """A whole document with its metadata, structure, and (after chunking) its chunks.

    Every loader (DOCX, PDF, ...) returns a list of these.
    Downstream modules (chunking, graph, entities, retrieval) only read from this shape.
    They never look at the original file.
    """
    source_file: str

    # ── Metadata (filename-derived + LLM-extracted) ──────────────────
    project_number: str
    supplier: str
    client: str
    discipline: str              
    document_type: str
    document_sequence: str
    revision: str
    attachment_number: str
    rig: str
    date: str
    odt_reference_persons: str
    abbreviations: Dict[str, str]

    # ── Structural ───────────────────────────────────────────────────
    headers: List[str]
    footers: List[str]
    blocks: List[Dict[str, Any]]
    sections: List[SectionData]

    # ── Chunks (populated by the chunking module) ────────────────────
    chunks: List[ChunkData]
