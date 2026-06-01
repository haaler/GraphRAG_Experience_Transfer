"""Write helpers for the indexing pipeline.

Each function takes a Neo4j transaction and the document dict produced by the loaders, and writes one piece 
of the graph (Documents, Sections, Chunks, the project / type / discipline nodes, plus revision supersession edges).
"""

import re
import json
from typing import List, Dict

from graphrag.entities.schemas import detect_document_type, detect_discipline


def compare_revisions(rev_a: str, rev_b: str) -> int:
    """Compare two revision strings.

    A letter suffix (e.g. '01A') is a draft that precedes the clean revision ('01').
    Ordering: 01A < 01 < 02A < 02 < 03.

    Returns 1 if rev_a > rev_b, -1 if less, 0 if equal.
    """
    def parse(r):
        """Parse a revision string into a tuple that sorts correctly. 
        Nested function to avoid exposing the parsing logic outside this comparison.
        """
        m = re.match(r'^0*(\d+)([A-Za-z]?)$', (r or "").strip())
        if not m:
            return (0, 0)
        num = int(m.group(1))
        has_letter_suffix = bool(m.group(2))

        # Encode so tuple comparison does the right thing:
        # "01A" -> (1, 0), "01" -> (1, 1), "02A" -> (2, 0), "02" -> (2, 1). 
        # Drafts (numbers with a letter suffix) sort before the final of the same number.
        return (num, 0 if has_letter_suffix else 1)

    pa, pb = parse(rev_a), parse(rev_b)
    return 1 if pa > pb else (-1 if pa < pb else 0)

def make_base_doc_id(doc: Dict) -> str:
    """Stable document family ID shared across revisions.

    Includes supplier so that different suppliers' bids for the same project are treated as separate 
    document families (not revisions of each other). 
    Includes attachment_number so that attachments are treated as separate documents from the main bid.

    Revision is intentionally excluded. This is what we compare across to decide supersession within a family.
    """
    parts = [
        doc.get("project_number", ""),
        doc.get("supplier", ""),
        doc.get("document_type", ""),
        doc.get("document_sequence", ""),
        doc.get("attachment_number", ""),
    ]
    # Empty string when no metadata at all — the caller treats that as "always ingest, never supersedes anyone".
    return "|".join(parts) if any(parts) else ""

def check_supersession(tx, doc: Dict):
    """Decide whether this document should be ingested and whether it supersedes an existing one.

    Returns (should_ingest: bool, superseded_source_file: str | None).
    """
    base_id = make_base_doc_id(doc)

    # No family ID (legacy / metadata-less filename). Always ingest fresh.
    if not base_id:
        return True, None

    # Look for an existing 'current' document in the same family.
    result = tx.run(
        "MATCH (d:Document {base_doc_id: $base_id, status: 'current'}) "
        "RETURN d.source_file AS source_file, d.revision AS revision "
        "LIMIT 1",
        base_id=base_id,
    ).single()

    # First document in its family — ingest fresh, nothing to supersede.
    if result is None:
        return True, None

    existing_rev = result["revision"] or ""
    new_rev = doc.get("revision", "")

    # New revision wins → ingest, and mark the old one as superseded.
    if compare_revisions(new_rev, existing_rev) > 0:
        return True, result["source_file"]
    # Old revision is newer or equal → skip; keep what's already in the graph.
    else:
        return False, None

def write_document_node(tx, doc: Dict) -> None:
    """Create or update the Document node with all available metadata."""
    # All new docs start with status='current'.
    # supersede_document() flips this to 'superseded' on the older revision when a newer one arrives.
    tx.run(
        "MERGE (d:Document {source_file: $source_file}) "
        "SET d.base_doc_id        = $base_doc_id, "
            "d.status             = 'current', "
            "d.project_number     = $project_number, "
            "d.supplier           = $supplier, "
            "d.client             = $client, "
            "d.discipline         = $discipline, "
            "d.document_type      = $document_type, "
            "d.document_sequence  = $document_sequence, "
            "d.revision           = $revision, "
            "d.attachment_number  = $attachment_number, "
            "d.rig                = $rig, "
            "d.date               = $date, "
            "d.title              = $title",
        source_file=doc["source_file"],
        base_doc_id=make_base_doc_id(doc),
        project_number=doc.get("project_number"),
        supplier=doc.get("supplier"),
        client=doc.get("client"),
        discipline=doc.get("discipline"),
        document_type=doc.get("document_type"),
        document_sequence=doc.get("document_sequence"),
        revision=doc.get("revision"),
        attachment_number=doc.get("attachment_number"),
        rig=doc.get("rig"),
        date=doc.get("date"),
        title=doc.get("title"),
    )

def write_sections_and_chunks(tx, doc: Dict) -> None:
    """Write Section and Chunk nodes, wiring HAS_SECTION and HAS_CHUNK edges."""
    source_file = doc["source_file"]

    # ── Sections + HAS_SECTION edges ──
    for section in doc["sections"]:
        # section_id = "<source_file>|<section_number>" — composite ID, since
        # section numbers only need to be unique within their own document.
        section_id = f"{source_file}|{section['section_number']}"
        tx.run(
            "MERGE (s:Section {section_id: $section_id}) "
            "SET s.section_number = $section_number, "
                "s.title          = $title, "
                "s.section_path   = $section_path, "
                "s.heading_level  = $heading_level, "
                "s.source_file    = $source_file "
            "WITH s "
            "MATCH (d:Document {source_file: $source_file}) "
            "MERGE (d)-[:HAS_SECTION]->(s)",
            section_id=section_id,
            section_number=section["section_number"],
            title=section["title"],
            section_path=section.get("section_path", section["title"]),
            heading_level=section.get("heading_level"),
            source_file=source_file,
        )

    # ── PARENT_SECTION edges between child and parent sections ──
    # Sections form a tree (1 → 1.1 → 1.1.1), so each section has at most one parent.
    # We link to the parent section if it exists, otherwise we just have a standalone section node.
    for section in doc["sections"]:
        parent_num = section.get("parent_section_number")
        if parent_num is not None:
            # Wire this section to its parent in the outline tree.
            child_id = f"{source_file}|{section['section_number']}"
            parent_id = f"{source_file}|{parent_num}"
            tx.run(
                "MATCH (child:Section {section_id: $child_id}), "
                "      (parent:Section {section_id: $parent_id}) "
                "MERGE (child)-[:PARENT_SECTION]->(parent)",
                child_id=child_id,
                parent_id=parent_id,
            )

    # ── Chunks + HAS_CHUNK edges ──
    for chunk in doc.get("chunks", []):
        section_id = f"{source_file}|{chunk['section_number']}"
        # Neo4j can't store nested structures as a property, so we flatten the content dict to JSON.
        # Retrieval reads it back via json.loads().
        content_str = json.dumps(chunk["content"], ensure_ascii=False)
        tx.run(
            "MERGE (c:Chunk {chunk_id: $chunk_id}) "
            "SET c.chunk_type     = $chunk_type, "
                "c.section_number = $section_number, "
                "c.section_title  = $section_title, "
                "c.section_path   = $section_path, "
                "c.source_file    = $source_file, "
                "c.content        = $content "
            "WITH c "
            "MATCH (s:Section {section_id: $section_id}) "
            "MERGE (s)-[:HAS_CHUNK]->(c)",
            chunk_id=chunk["chunk_id"],
            chunk_type=chunk["chunk_type"],
            section_number=chunk["section_number"],
            section_title=chunk["section_title"],
            section_path=chunk.get("section_path", chunk["section_title"]),
            source_file=source_file,
            content=content_str,
            section_id=section_id,
        )

def write_describes_edges(tx, doc: Dict) -> None:
    """Add DESCRIBES edges between text chunks that precede table/list chunks."""
    chunks = doc.get("chunks", [])
    for i in range(len(chunks) - 1):
        curr, nxt = chunks[i], chunks[i + 1]
        # Many documents have a paragraph that introduces a table or list ("The following table summarises...").
        # The DESCRIBES edge captures that pairing so retrieval can pull the lead-in 
        # text along with the table or list it refers to.
        if (
            curr["chunk_type"] == "text"
            and nxt["chunk_type"] in ("table", "list")
            and curr["section_number"] == nxt["section_number"]
        ):
            tx.run(
                "MATCH (a:Chunk {chunk_id: $a_id}), (b:Chunk {chunk_id: $b_id}) "
                "MERGE (a)-[:DESCRIBES]->(b)",
                a_id=curr["chunk_id"],
                b_id=nxt["chunk_id"],
            )

def supersede_document(tx, old_source_file: str, new_source_file: str) -> None:
    """Mark the old document as superseded and link it to the replacement."""
    # Superseded docs aren't deleted. Instead, retrieval de-prioritises them so we
    # don't surface stale revisions of the same content while keeping the history queryable.
    tx.run(
        "MATCH (old:Document {source_file: $old_sf}), "
              "(new:Document {source_file: $new_sf}) "
        "SET old.status = 'superseded' "
        "MERGE (old)-[:SUPERSEDED_BY]->(new)",
        old_sf=old_source_file,
        new_sf=new_source_file,
    )

# The three "write_*_node" functions below (Project, DocumentType, Discipline) all create a category node and link the document to it.
# These category nodes let queries pivot on metadata without scanning every Document (e.g.
# "all docs for project 96132", "all bid documents").
def write_project_node(tx, doc: Dict) -> None:
    """Create the Project node and link the Document to it via HAS_DOCUMENT."""
    pn = doc.get("project_number")
    if not pn:
        return
    tx.run(
        "MERGE (p:Project {project_number: $pn}) "
        "WITH p "
        "MATCH (d:Document {source_file: $sf}) "
        "MERGE (p)-[:HAS_DOCUMENT]->(d)",
        pn=pn, sf=doc["source_file"],
    )

def write_document_type_node(tx, doc: Dict) -> None:
    """Create the DocumentType node and link the Document via HAS_TYPE.

    Both the code (e.g. 'BP') and the full name (e.g. 'Bid/Proposal') are stored so queries can filter by either.
    Documents with an unrecognised or absent code get type 'Typeless'.
    """
    code = (doc.get("document_type") or "").upper() or "UNKNOWN"
    name = detect_document_type(doc)
    tx.run(
        "MERGE (dt:DocumentType {code: $code}) "
        "SET dt.name = $name "
        "WITH dt "
        "MATCH (d:Document {source_file: $sf}) "
        "MERGE (d)-[:HAS_TYPE]->(dt)",
        code=code, name=name, sf=doc["source_file"],
    )

def write_discipline_node(tx, doc: Dict) -> None:
    """Create the Discipline node and link the Document via HAS_DISCIPLINE.

    Only written when the document has a discipline code (the discipline-variant filenames). 
    Both the code (e.g. 'J') and the full name from DISCIPLINE_CODES are stored.
    If no entry exists in DISCIPLINE_CODES, the raw code is used as the name as well so the node is still created.
    """
    code = (doc.get("discipline") or "").upper()
    if not code:
        return
    name = detect_discipline(doc) or code
    tx.run(
        "MERGE (disc:Discipline {code: $code}) "
        "SET disc.name = $name "
        "WITH disc "
        "MATCH (d:Document {source_file: $sf}) "
        "MERGE (d)-[:HAS_DISCIPLINE]->(disc)",
        code=code, name=name, sf=doc["source_file"],
    )
