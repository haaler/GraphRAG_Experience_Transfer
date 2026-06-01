"""Context assembly for the retrieval stage.

Four public functions, all called from ask.py to build the prompt the answer-generation LLM eventually sees:

  - fetch_chunks: given a list of chunk_ids, load each chunk's content and its document metadata from Neo4j.
  - gather_entity_context: walk the entity graph from query terms to build the Knowledge graph context block (Stage 2b).
  - gather_community_context: mix local entities from traversal hits with semantic embedding
    search to build the Community context block (also Stage 2b).
  - assemble_context: sort surviving chunks by _score, picking those that fit under CONTEXT_MAX_CHARS, 
    group by source document, and render each chunk with a document header (Stage 3 input).
"""

import json
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from graphrag import config as default_config
from graphrag.retrieval.chunk_retrieval import _extract_query_terms


def _content_to_text(content_str: str, chunk_type: str) -> str:
    """Convert a chunk's stored content (JSON) to readable text."""
    # Fall back to the raw string when the content is missing or not valid JSON, 
    # so a single malformed chunk does not crash retrieval.
    try:
        content = json.loads(content_str or "{}")
    except (json.JSONDecodeError, TypeError):
        return content_str or ""

    if chunk_type == "text":
        return content.get("text", "")
    elif chunk_type == "table":
        rows = content.get("rows", [])
        if not rows:
            return ""
        lines = []
        for row in rows:
            if isinstance(row, dict):
                lines.append("  " + " | ".join(f"{key}: {val}" for key, val in row.items()))
            else:
                lines.append("  " + str(row))
        return "\n".join(lines)
    elif chunk_type == "list":
        items = content.get("items", [])
        lines = [
            "  " * item.get("level", 0) + "- " + item.get("text", "")
            for item in items
        ]
        return "\n".join(lines)
    return str(content)


def fetch_chunks(chunk_ids: List[str], driver) -> List[Dict]:
    """Fetch chunk content and document metadata from Neo4j, preserving order."""
    if not chunk_ids:
        return []

    with driver.session() as session:
        # OPTIONAL MATCH on Document because we pull document metadata for the header line built by _build_doc_header. 
        # We keep it optional to keep potential orphan chunks (possible after a deletion) 
        # in the result rather than dropping it silently.
        rows = session.run(
            "MATCH (c:Chunk) WHERE c.chunk_id IN $ids "
            "OPTIONAL MATCH (d:Document {source_file: c.source_file}) "
            "RETURN c.chunk_id      AS chunk_id, "
            "       c.content       AS content, "
            "       c.chunk_type    AS chunk_type, "
            "       c.section_title AS section_title, "
            "       c.source_file   AS source_file, "
            "       d.project_number AS project_number, "
            "       d.rig           AS rig, "
            "       d.status        AS status, "
            "       d.revision      AS revision, "
            "       d.date          AS date, "
            "       d.title         AS title",
            ids=chunk_ids,
        ).data()

    # Map chunk_id -> chunk dict with the JSON content already decoded to text,
    # so the rest of the pipeline gets a single row dict per chunk.
    row_map = {}
    for row in rows:
        chunk_id = row["chunk_id"]
        row_map[chunk_id] = {**row, "text": _content_to_text(row["content"], row["chunk_type"])}
    return [row_map[cid] for cid in chunk_ids if cid in row_map]


def gather_entity_context(question: str, driver, doc_filter: Optional[List[str]] = None) -> str:
    """Query the entity graph for cross-document information.

    Traverses typed relationships (SUPPLIER_FOR, LOCATED_AT, etc.) to build a rich knowledge graph context.
    """
    terms = _extract_query_terms(question)
    # Drops short bigrams (e.g., "to it"). 
    # _extract_query_terms already filters single tokens by length, 
    # but bigrams of two very short words can slip through.
    terms = [term for term in terms if len(term) >= 4]
    if not terms:
        return ""

    # Track entity info: {entity_name: {type, relationships, documents}}
    entity_info: dict = defaultdict(lambda: {
        "entity_type": "Unknown",
        "relationships": [],  # (rel_type, target_name, target_type)
        "documents": defaultdict(set),  # doc_label -> set of roles
    })

    with driver.session() as session:
        for term in terms:
            doc_clause = "WHERE c.source_file IN $doc_filter " if doc_filter else ""
            params = {"term": term}
            if doc_filter:
                params["doc_filter"] = doc_filter

            # Get matching entities with typed relationships
            cypher_rels = (
                "MATCH (e:Entity) "
                "WHERE toLower(e.name) CONTAINS $term "
                "OPTIONAL MATCH (e)-[r]-(related:Entity) "
                "WHERE NOT type(r) IN ['IS_A', 'HAS_ENTITY', 'MEMBER_OF'] "
                "RETURN e.name AS entity, e.entity_type AS entity_type, "
                "       type(r) AS rel_type, related.name AS related_name, "
                "       related.entity_type AS related_type, "
                "       startNode(r) = e AS is_outgoing"
            )
            for row in session.run(cypher_rels, **params):
                entity_name = row["entity"]
                info = entity_info[entity_name]
                info["entity_type"] = row["entity_type"] or "Unknown"
                if row["rel_type"]:
                    info["relationships"].append((
                        row["rel_type"],
                        row["related_name"],
                        row["related_type"] or "Unknown",
                        row["is_outgoing"],
                    ))

            # Get document associations for matching entities
            cypher_docs = (
                "MATCH (e:Entity) "
                "WHERE toLower(e.name) CONTAINS $term "
                "WITH e AS ent "
                "MATCH (ent)<-[r:HAS_ENTITY]-(c:Chunk) "
                + doc_clause +
                "WITH DISTINCT ent, r, c "
                "OPTIONAL MATCH (d:Document {source_file: c.source_file}) "
                "RETURN ent.name AS entity, r.entity_type AS role, "
                "       c.source_file AS doc, d.project_number AS project, "
                "       d.status AS status"
            )
            for row in session.run(cypher_docs, **params):
                entity_name = row["entity"]
                doc = row["doc"] or "unknown"
                role = row["role"] or "entity"
                proj = row.get("project") or ""
                is_superseded = row.get("status") == "superseded"
                doc_label = doc + (f" (project {proj})" if proj else "")
                if is_superseded:
                    doc_label += " [SUPERSEDED]"
                entity_info[entity_name]["documents"][doc_label].add(role)

    if not entity_info:
        return ""

    lines = ["Knowledge graph context:"]
    for entity_name, info in sorted(entity_info.items()):
        lines.append(f"  '{entity_name}' ({info['entity_type']})")

        # Show typed relationships
        seen_rels = set()
        for rel_type, target, target_type, is_outgoing in info["relationships"]:
            rel_key = (rel_type, target)
            if rel_key in seen_rels:
                continue
            seen_rels.add(rel_key)
            arrow = f"--{rel_type}-->" if is_outgoing else f"<--{rel_type}--"
            lines.append(f"    {arrow} '{target}' ({target_type})")

        # Show document associations
        docs = info["documents"]
        if docs:
            doc_strs = sorted(docs.keys())
            lines.append(f"    Found in: {', '.join(doc_strs)}")

    return "\n".join(lines)


def gather_community_context(question: str, driver, embed_fn) -> str:
    """Retrieve community summaries relevant to the question.

    Combines two sources, deduped by community_id:
      1. Local: entities matching query terms -> MEMBER_OF -> community summaries.
      2. Global: semantic search over Community.embedding (cosine) via the
         community_embedding_index, so broad questions still reach relevant
         clusters even when no entity name matches.
    """
    max_summaries = getattr(default_config, "COMMUNITY_SUMMARIES_MAX", 5)
    semantic_top_k = getattr(default_config, "COMMUNITY_SEMANTIC_TOP_K", 5)

    # Dedup by community_id; local results take precedence
    seen_ids: set = set()
    ordered: list = []  # list of (community_id, summary)

    # ── Local (entity-traversal) ──────────────────────────────────────
    terms = _extract_query_terms(question)
    # Drops short bigrams. 
    # _extract_query_terms already filters single tokens by length, but bigrams of two very short words can slip through.
    terms = [term for term in terms if len(term) >= 4]
    if terms:
        with driver.session() as session:
            for term in terms:
                if len(ordered) >= max_summaries:
                    break
                for row in session.run(
                    "MATCH (e:Entity) "
                    "WHERE toLower(e.name) CONTAINS $term "
                    "MATCH (e)-[:MEMBER_OF]->(comm:Community) "
                    "WHERE comm.summary IS NOT NULL "
                    "RETURN DISTINCT comm.community_id AS cid, "
                    "                comm.summary      AS summary",
                    term=term,
                ):
                    cid = row["cid"]
                    if cid in seen_ids:
                        continue
                    seen_ids.add(cid)
                    ordered.append((cid, row["summary"]))
                    if len(ordered) >= max_summaries:
                        break

    # ── Global (semantic search over community embeddings) ────────────
    if len(ordered) < max_summaries:
        try:
            q_emb = embed_fn(question)
            with driver.session() as session:
                rows = session.run(
                    "CALL db.index.vector.queryNodes("
                    "'community_embedding_index', $k, $emb) "
                    "YIELD node, score "
                    "WHERE node.summary IS NOT NULL "
                    "RETURN node.community_id AS cid, "
                    "       node.summary      AS summary",
                    k=semantic_top_k,
                    emb=q_emb,
                ).data()
            for row in rows:
                if len(ordered) >= max_summaries:
                    break
                cid = row["cid"]
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)
                ordered.append((cid, row["summary"]))
        except Exception:
            # Vector index may not exist yet (e.g., communities not built)
            pass

    if not ordered:
        return ""

    lines = ["Community context:"]
    for _, summary in ordered[:max_summaries]:
        lines.append(f"  - {summary}")
    return "\n".join(lines)


def _format_chunk_block(c: Dict) -> str:
    """Format a single chunk as a text block for the context string."""
    section = c.get("section_title") or "--"
    ctype   = c.get("chunk_type")    or "text"
    text    = (c.get("text") or "").strip()
    return f"  [{section} / {ctype}]\n  {text}"


def _build_doc_header(c: Dict) -> str:
    """Build a document header line from a chunk's metadata."""
    header = f"[Document: {c['source_file']}"
    if c.get("title"):
        header += f" | {c['title']}"
    if c.get("project_number"):
        header += f" | Project {c['project_number']}"
    if c.get("rig"):
        header += f" | {c['rig']}"
    if c.get("date"):
        header += f" | Date {c['date']}"
    if c.get("revision"):
        header += f" | Rev {c['revision']}"
    if c.get("status") == "superseded":
        header += " | SUPERSEDED"
    header += "]"
    return header


def _chunk_score(chunk: Dict) -> float:
    """Sort key for assemble_context: a chunk's retrieval score, defaulting
    to 0.0 for chunks that arrived without one (e.g. baseline document-level inclusion).
    """
    return chunk.get("_score", 0.0)


def assemble_context(chunks: List[Dict], cfg=None) -> Tuple[str, List[str]]:
    """Format retrieved chunks as a context string for the LLM.

    Sorts chunks by _score (highest first), selects within CONTEXT_MAX_CHARS,
    then groups by document for readable output. 
    Never truncates a chunk mid-text.

    Returns (context_string, used_doc_files) where used_doc_files lists
    only the documents whose chunks were actually selected into the context.
    """
    if cfg is None:
        cfg = default_config

    max_chars = getattr(cfg, "CONTEXT_MAX_CHARS", 8000)

    if not chunks:
        return "(No relevant chunks found.)", []

    # ── Score-based selection ─────────────────────────────────────────
    sorted_chunks = sorted(chunks, key=_chunk_score, reverse=True)

    selected: list = []
    total_chars = 0
    for chunk in sorted_chunks:
        block = _format_chunk_block(chunk)
        block_len = len(block) + 1  # +1 for newline
        if total_chars + block_len > max_chars and selected:
            break
        selected.append(chunk)
        total_chars += block_len

    # ── Group selected chunks by document (first-seen order) ─────────
    by_doc: dict = defaultdict(list)
    doc_order: list = []
    for chunk in selected:
        source_file = chunk["source_file"]
        if source_file not in by_doc:
            doc_order.append(source_file)
        by_doc[source_file].append(chunk)

    # ── Format with document headers ─────────────────────────────────
    parts = []
    for source_file in doc_order:
        doc_chunks = by_doc[source_file]
        header = _build_doc_header(doc_chunks[0])
        doc_parts = [header]
        for chunk in doc_chunks:
            doc_parts.append(_format_chunk_block(chunk))
        doc_parts.append("")  # trailing blank line
        parts.append("\n".join(doc_parts))

    return "\n".join(parts), doc_order
