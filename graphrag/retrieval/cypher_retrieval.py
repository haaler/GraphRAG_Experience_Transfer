"""Text-to-Cypher retrieval stage (Stage 2c of the query pipeline).

Generates a read-only Cypher query from the user's question, executes it
against Neo4j, and formats the tabular result for the answer-generation LLM.
Runs alongside the keyword retrieval stage rather than replacing it.
When the question is structural (counts, lists, aggregations) the Cypher result
gives the answer LLM authoritative tabular data. 
When the question asks for the contents of a specific document, the query returns chunk_ids so
the pipeline can fetch the chunk text alongside the rest of the context.

Includes a forbid-write-keywords safety check, a schema-validation pass
that confirms every relationship pattern in the generated Cypher matches
a real triple in the live graph, and one retry on either failure.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Dict, List, Optional, Tuple

from graphrag import config as default_config
from graphrag.entities.schemas import (
    ALL_RELATIONSHIP_TYPES,
    CORE_SCHEMA,
    RELATIONSHIPS,
    TYPE_DESCRIPTIONS,
)


# ── Schema description (built once) ───────────────────────────────────────
# Structural rel types we never want to advertise to the Cypher LLM as entity-to-entity edges.
# Used by the schema-off branch when discovering the live rel-type vocabulary from Neo4j.
_NON_ENTITY_REL_TYPES = {
    "HAS_ENTITY", "IS_A", "MEMBER_OF", "SIMILAR_TO",
    "HAS_SECTION", "HAS_CHUNK", "PARENT_SECTION",
    "SUPERSEDED_BY", "HAS_DOCUMENT", "HAS_TYPE",
    "HAS_DISCIPLINE", "DESCRIBES",
}


def _build_schema_description(cfg=None, driver=None) -> str:
    """Build the schema description block that goes into the LLM prompt.

    Schema-on (default): pulls from CORE_SCHEMA / RELATIONSHIPS so the description stays in sync
    with the conceptual schema in graphrag/entities/schemas.py.

    Schema-off: discovers the live entity-type vocabulary and entity-to-entity rel labels from
    Neo4j. Requires a driver. Falls back to the schema-on text if the driver is None
    (e.g. at module import time) since the live data isn't available yet.
    """
    use_schema = True if cfg is None else getattr(cfg, "USE_CONCEPTUAL_SCHEMA", True)

    if not use_schema and driver is not None:
        entity_types_block, typed_rels_block = _build_live_schema_blocks(driver)
    else:
        # Schema-on (or no driver yet): describe the curated schema.
        entity_type_lines = []
        for entity_class, types in CORE_SCHEMA.items():
            entity_type_lines.append(f"  entity_class '{entity_class}':")
            for entity_type in types:
                description = TYPE_DESCRIPTIONS.get(entity_type, "")
                if description:
                    entity_type_lines.append(f"    - {entity_type}: {description}")
                else:
                    entity_type_lines.append(f"    - {entity_type}")
        entity_types_block = "\n".join(entity_type_lines)

        rel_lines = []
        seen = set()
        for src, rel, target, description in RELATIONSHIPS:
            key = (src, rel, target)
            if key in seen:
                continue
            seen.add(key)
            rel_lines.append(f"  (:Entity {{entity_type: '{src}'}}) -[:{rel}]-> (:Entity {{entity_type: '{target}'}})  ({description})")
        typed_rels_block = "\n".join(rel_lines)

    if use_schema:
        types_header = "Entity types (use as the 'entity_type' property value on :Entity nodes):"
        rels_header = "Typed entity-to-entity relationships:"
    else:
        types_header = (
            "Entity types currently present in the graph "
            "(the extraction LLM picked these — they are free-form, not from a curated list):"
        )
        rels_header = (
            "Entity-to-entity relationship labels currently present in the graph "
            "(also free-form, picked by the extraction LLM):"
        )

    return f"""\
GRAPH SCHEMA

Node labels and key properties:
  (:Document {{source_file, project_number, supplier, client, discipline,
              document_type, revision, rig, status, date, title}})
      -- status is 'current' or 'superseded'
      -- title is the document SUBJECT (e.g. "Mooring Analysis Norway"), NOT
         the document classification. Words like "bid", "report", "checklist"
         describe the document_type, not the title. Use DocumentType.name or
         d.document_type for classification matching.
      -- rig contains the full rig/installation name (e.g. "Deepsea Stavanger").
         When the question mentions a specific rig, filter on d.rig (not d.title).
         Common rig abbreviations in filenames: DSS = Deepsea Stavanger,
         DSA = Deepsea Atlantic, DSN = Deepsea Nordkapp.
  (:Project {{project_number}})
  (:Section {{section_id, title, heading_level, source_file}})
  (:Chunk   {{chunk_id, chunk_type, section_title, source_file}})
      -- chunk_type is 'text', 'table', or 'list'
  (:Entity  {{name, entity_type, entity_class, merge_key}})
      -- merge_key is lowercased+whitespace-collapsed name (use for exact match)
      -- entity_type is a PROPERTY on Entity, NOT a node label
  (:EntityType    {{name}})
  (:DocumentType  {{code, name}})
  (:Discipline    {{code, name}})
  (:Community     {{community_id, summary}})

Document-structure edges:
  (Document)-[:HAS_SECTION]->(Section)-[:HAS_CHUNK]->(Chunk)
  (Section)-[:PARENT_SECTION]->(Section)              -- child to parent
  (Document)-[:SUPERSEDED_BY]->(Document)
  (Project)-[:HAS_DOCUMENT]->(Document)
  (Document)-[:HAS_TYPE]->(DocumentType)
  (Document)-[:HAS_DISCIPLINE]->(Discipline)
  (Chunk)-[:DESCRIBES]->(Chunk)                       -- text describes table/list

IMPORTANT — edges that do NOT exist:
  - There is NO direct edge between Document and Chunk.
    HAS_CHUNK connects Section→Chunk ONLY.
    To bridge Chunk↔Document, join on property: c.source_file = d.source_file
    Example: MATCH (c:Chunk) MATCH (d:Document {{source_file: c.source_file}})

Entity linkage edges:
  (Chunk)-[:HAS_ENTITY]->(Entity)   -- the main chunk-to-entity bridge
  (Entity)-[:IS_A]->(EntityType)
  (Entity)-[:MEMBER_OF]->(Community)

{rels_header}
{typed_rels_block}

{types_header}
{entity_types_block}

USAGE HINTS
  - To find all entities of a type, match on the property:
      (e:Entity {{entity_type: 'ContactPerson'}})
    Do NOT use (:ContactPerson) — entity_type is a property, not a label.
  - To list the documents an entity appears in:
      (e:Entity)<-[:HAS_ENTITY]-(c:Chunk) and group by c.source_file
  - When the question asks "which documents mention / use / discuss X",
    match on entities as well via (Entity)<-[:HAS_ENTITY]-(Chunk) and union
    with a section_title content fallback. As entity names are case-sensitive, 
    try with different capitalizations on the first letters of words, 
    or where it otherwise might be natural to have / not have capitalization. 
    The user often does not know whether X is an extracted entity; 
    the entity branch covers the case where it is, the fallback covers the case where it isn't.
  - Filter out old revisions with WHERE d.status = 'current' unless the user
    asked about history or superseded documents.
  - Match entity names fuzzily with: toLower(e.name) CONTAINS 'xxx'
    Or exactly with the normalized key: e.merge_key = 'xxx'
  - Document.source_file is the document identity; Project.project_number
    matches Document.project_number.
  - Documents are classified by discipline via (Document)-[:HAS_DISCIPLINE]->(Discipline).
    Discipline names include: Inspection, Drilling, Procurement, Administration, etc.
    When the user asks about "X documents" (e.g., "inspection documents"), prefer
    matching via the Discipline node first, with content search as a fallback.
  - To retrieve the text of a specific document's chunks:
      MATCH (d:Document {{source_file: '...'}})-[:HAS_SECTION]->(s)-[:HAS_CHUNK]->(c)
      RETURN c.chunk_id AS chunk_id, c.section_title AS section_title
    Include chunk_id in RETURN so the pipeline can fetch the full text.
"""


def _build_live_schema_blocks(driver) -> Tuple[str, str]:
    """Query Neo4j for the entity-type and entity-to-entity rel-label vocabularies.

    Used by the schema-off branch of _build_schema_description. Returns two text blocks ready
    to drop into the prompt. Empty list lines are returned when the graph has no data yet.
    """
    with driver.session() as session:
        type_rows = session.run(
            "MATCH (e:Entity) WHERE e.entity_type IS NOT NULL "
            "RETURN DISTINCT e.entity_type AS entity_type ORDER BY entity_type"
        ).data()
        rel_rows = session.run(
            "MATCH (a:Entity)-[r]-(b:Entity) "
            "RETURN DISTINCT type(r) AS rel_type ORDER BY rel_type"
        ).data()

    entity_type_lines = [f"  - {row['entity_type']}" for row in type_rows if row.get("entity_type")]
    if not entity_type_lines:
        entity_type_lines = ["  (no entities written yet)"]

    rel_lines = []
    for row in rel_rows:
        rel = row.get("rel_type")
        if not rel or rel in _NON_ENTITY_REL_TYPES:
            continue
        rel_lines.append(f"  (:Entity) -[:{rel}]-> (:Entity)")
    if not rel_lines:
        rel_lines = ["  (no entity-to-entity edges written yet)"]

    return "\n".join(entity_type_lines), "\n".join(rel_lines)


# Cache for the prompt schema-description text. Keyed by mode ("on" / "off").
# Schema-on is built once at import. Schema-off is built lazily on first run_cypher_retrieval call
# because it needs a live driver to read the entity-type and rel-label vocabularies from the graph.
_SCHEMA_DESCRIPTION_CACHE: Dict[str, str] = {
    "on": _build_schema_description(),
}


def _get_schema_description(cfg, driver) -> str:
    """Return the prompt schema-description text, building (and caching) the right variant.

    Cached because the same description is reused across retries within a single Stage 2c call.
    The cache survives across calls, so the live-graph variant is queried only once per process.
    """
    use_schema = getattr(cfg, "USE_CONCEPTUAL_SCHEMA", True)
    mode = "on" if use_schema else "off"
    if mode not in _SCHEMA_DESCRIPTION_CACHE:
        _SCHEMA_DESCRIPTION_CACHE[mode] = _build_schema_description(cfg=cfg, driver=driver)
    return _SCHEMA_DESCRIPTION_CACHE[mode]


# Kept as a module-level convenience for callers that want the default (schema-on) block.
SCHEMA_DESCRIPTION = _SCHEMA_DESCRIPTION_CACHE["on"]


# ── Few-shot examples (diverse patterns) ──────────────────────────────────
FEW_SHOT_EXAMPLES = """\
EXAMPLES — X and Y are placeholders; replace them with values from the user's question.

Q: List all entities of type X.
A: {"cypher": "MATCH (e:Entity {entity_type: 'X'}) RETURN e.name AS name ORDER BY name"}

Q: Which entities of type X appear in two or more documents?
A: {"cypher": "MATCH (e:Entity {entity_type: 'X'})<-[:HAS_ENTITY]-(c:Chunk) WITH e, collect(DISTINCT c.source_file) AS docs WHERE size(docs) >= 2 RETURN e.name AS name, docs ORDER BY size(docs) DESC"}

Q: Which documents are X documents?
A: {"cypher": "MATCH (d:Document)-[:HAS_DISCIPLINE]->(disc:Discipline) WHERE toLower(disc.name) CONTAINS 'x' AND d.status = 'current' RETURN d.source_file AS file, d.project_number AS project, d.rig AS rig UNION MATCH (d:Document {status: 'current'})-[:HAS_SECTION]->()-[:HAS_CHUNK]->(c) WHERE toLower(c.section_title) CONTAINS 'x' RETURN DISTINCT d.source_file AS file, d.project_number AS project, d.rig AS rig"}

Q: Which entities of type X are related to entities of type Y?
A: {"cypher": "MATCH (a:Entity {entity_type: 'X'})-[r]-(b:Entity {entity_type: 'Y'}) WHERE NOT type(r) IN ['IS_A', 'HAS_ENTITY', 'MEMBER_OF'] RETURN a.name AS x, type(r) AS relationship, b.name AS y"}

Q: How many documents are in each project?
A: {"cypher": "MATCH (p:Project)-[:HAS_DOCUMENT]->(d:Document) WHERE d.status = 'current' RETURN p.project_number AS project, count(d) AS n_documents ORDER BY n_documents DESC"}

Q: Which other documents are in the same project as entity X?
A: {"cypher": "MATCH (e:Entity)<-[:HAS_ENTITY]-(c:Chunk) WHERE toLower(e.name) CONTAINS 'x' MATCH (d:Document {source_file: c.source_file}) MATCH (p:Project)-[:HAS_DOCUMENT]->(d) MATCH (p)-[:HAS_DOCUMENT]->(d2:Document) WHERE d2.source_file <> d.source_file AND d2.status = 'current' RETURN DISTINCT p.project_number AS project, d2.source_file AS other_document ORDER BY project"}

Q: Which documents mention X?
A: {"cypher": "MATCH (e:Entity)<-[:HAS_ENTITY]-(c:Chunk) WHERE toLower(e.name) CONTAINS 'x' MATCH (d:Document {source_file: c.source_file}) WHERE d.status = 'current' RETURN DISTINCT d.source_file AS file, d.project_number AS project, d.rig AS rig UNION MATCH (d:Document {status: 'current'})-[:HAS_SECTION]->()-[:HAS_CHUNK]->(c:Chunk) WHERE toLower(c.section_title) CONTAINS 'x' RETURN DISTINCT d.source_file AS file, d.project_number AS project, d.rig AS rig"}

Q: What does document X say about Y?
A: {"cypher": "MATCH (d:Document)-[:HAS_SECTION]->(s)-[:HAS_CHUNK]->(c) WHERE toLower(d.source_file) CONTAINS 'x' AND (toLower(c.section_title) CONTAINS 'y' OR toLower(c.content) CONTAINS 'y') RETURN c.chunk_id AS chunk_id, c.section_title AS section_title, c.chunk_type AS chunk_type"}

Q: Which bids are for rig X?
A: {"cypher": "MATCH (d:Document)-[:HAS_TYPE]->(dt:DocumentType) WHERE toLower(d.rig) CONTAINS 'x' AND toLower(dt.name) CONTAINS 'bid' AND d.status = 'current' RETURN d.source_file AS file, d.title AS title, d.project_number AS project, d.rig AS rig ORDER BY d.project_number"}
"""


CYPHER_SYSTEM_MSG = (
    "You are a Cypher expert for a Neo4j knowledge graph about offshore drilling documents."
    "Your job is to translate a natural-language question into ONE read-only Cypher query."
    "For structural questions, return aggregated data."
    "For content questions about specific documents, return chunk_ids so the pipeline can fetch the text. Respond with JSON only."
)


CYPHER_GEN_PROMPT = """\
{schema}

{examples}

TASK
Write ONE read-only Cypher query (MATCH / RETURN / WITH / WHERE / ORDER BY /
aggregations only — NO CREATE / DELETE / SET / REMOVE / MERGE / DROP / CALL /
LOAD / FOREACH) that answers the question.

Return JSON in one of these shapes, nothing else:
  {{"cypher": "<the query as a single line>"}}
  {{"cypher": null, "reason": "<why this question cannot be answered by graph traversal>"}}

When the question asks about the CONTENTS of a specific document or section,
return a query that retrieves chunk_ids so the pipeline can fetch the text:
  MATCH (d:Document)-[:HAS_SECTION]->(s)-[:HAS_CHUNK]->(c)
  WHERE d.source_file = '...'
  RETURN c.chunk_id AS chunk_id, c.section_title AS section_title
You may filter by section_title or chunk content if the question is specific.
Always include chunk_id in the RETURN clause for content retrieval queries.

NEVER filter on chunk_type (e.g. c.chunk_type IN ['text','table','list']) — all
chunks are one of those types, so the condition matches everything and is useless.

Decline (return null) ONLY when:
  - The question has no grounding in the schema above
{doc_filter_block}
Question: {question}
"""


# ── Safety ────────────────────────────────────────────────────────────────

# Write/admin keywords that must NOT appear in a generated query.
# Word-boundary regex so 'SET' doesn't match inside identifiers like 'RESET'.
_FORBIDDEN_PATTERN = re.compile(
    r"\b(CREATE|DELETE|DETACH|SET|REMOVE|MERGE|DROP|LOAD|FOREACH|CALL)\b",
    re.IGNORECASE,
)


def _is_safe_query(cypher: str) -> Tuple[bool, str]:
    """Return (is_safe, reason_if_not)."""
    if not cypher or not cypher.strip():
        return False, "empty query"
    match = _FORBIDDEN_PATTERN.search(cypher)
    if match:
        return False, f"contains forbidden keyword '{match.group(0)}'"
    return True, ""


# ── Schema validation ────────────────────────────────────────────────────

# Regex to extract relationship patterns from generated Cypher.
# Matches: (var:Label ...)-[:REL]->(var:Label ...) and reverse/undirected variants.
# Captures: left_label, rel_type, direction_arrow, right_label.
_REL_PATTERN = re.compile(
    r'\((?:\w+)?:(\w+)'          # left node, capture label
    r'[^)]*\)'                    # rest of left node (properties, etc.)
    r'\s*'
    r'(<-\[:(\w+)\]-'            # reverse: <-[:REL]-
    r'|-\[:(\w+)\]->'            # forward: -[:REL]->
    r'|-\[:(\w+)\]-)'            # undirected: -[:REL]-
    r'\s*'
    r'\((?:\w+)?:(\w+)'          # right node, capture label
)


def _load_valid_triples(driver) -> set:
    """Query Neo4j for all (source_label, rel_type, target_label) triples."""
    with driver.session() as session:
        rows = session.run(
            "CALL db.schema.visualization() YIELD nodes, relationships "
            "UNWIND relationships AS r "
            "RETURN labels(startNode(r))[0] AS src, type(r) AS rel, "
            "       labels(endNode(r))[0] AS target"
        ).data()
    return {(row["src"], row["rel"], row["target"]) for row in rows}


def _validate_schema(cypher: str, valid_triples: set) -> Tuple[bool, str]:
    """Check that every relationship pattern in the Cypher uses valid triples.

    Returns (is_valid, error_message).
    """
    for m in _REL_PATTERN.finditer(cypher):
        left_label = m.group(1)
        right_label = m.group(6)

        if m.group(3):       # reverse: <-[:REL]-
            rel_type = m.group(3)
            # Actual direction is right→left
            triple = (right_label, rel_type, left_label)
        elif m.group(4):     # forward: -[:REL]->
            rel_type = m.group(4)
            triple = (left_label, rel_type, right_label)
        elif m.group(5):     # undirected: -[:REL]-
            rel_type = m.group(5)
            # Accept either direction
            if ((left_label, rel_type, right_label) in valid_triples
                    or (right_label, rel_type, left_label) in valid_triples):
                continue
            triple = (left_label, rel_type, right_label)
        else:
            continue

        if triple not in valid_triples:
            # Build helpful error: show what IS valid for this rel_type
            valid_for_rel = [
                f"(:{s})-[:{r}]->(:{t})"
                for s, r, t in valid_triples if r == rel_type
            ]
            hint = (f" Valid: {', '.join(valid_for_rel)}."
                    if valid_for_rel
                    else f" Relationship type {rel_type} does not exist in the graph.")
            if rel_type == "HAS_CHUNK":
                hint += (" To bridge Chunk <--> Document, use property match: "
                         "MATCH (d:Document {source_file: c.source_file})")
            return False, (
                f"Invalid relationship: (:{triple[0]})-[:{triple[1]}]->(:{triple[2]}) "
                f"does not exist.{hint}"
            )

    return True, ""


def _ensure_limit(cypher: str, max_rows: int) -> str:
    """Append a LIMIT clause if the query doesn't already have one.

    Stage 2c rows become tabular text in the answer-LLM context. 
    Without a LIMIT, a query that accidentally produces many rows (e.g. an unfiltered
    list of every entity) could blow past the context budget. 
    If the LLM already included a LIMIT, we trust its choice and leave it alone.
    """
    if re.search(r"\bLIMIT\b\s+\d+", cypher, re.IGNORECASE):
        return cypher
    return cypher.rstrip().rstrip(";") + f" LIMIT {max_rows}"


# ── LLM → Cypher ──────────────────────────────────────────────────────────
def _parse_llm_response(raw: str) -> Tuple[str, str]:
    """Parse the LLM response. Returns (cypher, reason).

    cypher is "" when the LLM declined (reason carries the explanation).
    Raises ValueError on invalid JSON.
    """
    raw = raw.strip()
    # Found that the LLM sometimes wraps its JSON response in ``` fences despite the system instruction.
    # Strip the fence and the optional "json" hint so the downstream json.loads still works.
    if "```" in raw:
        parts = raw.split("```")
        if len(parts) >= 2:
            raw = parts[1]
            if raw.lower().startswith("json"):
                raw = raw[4:]
    result = json.loads(raw.strip())
    cypher = result.get("cypher")
    reason = result.get("reason", "")
    if cypher is None:
        return "", reason or "LLM declined (no reason given)"
    if not isinstance(cypher, str):
        raise ValueError(f"'cypher' must be a string or null, got {type(cypher).__name__}")
    # Normalize whitespace (single-line expected, but tolerate multiline)
    cypher = " ".join(cypher.split())
    return cypher, reason


def _generate_cypher(
    question: str,
    llm,
    error_feedback: str = "",
    doc_filter: Optional[List[str]] = None,
    confidence: str = "n/a",
    schema_description: Optional[str] = None,
) -> Tuple[str, str]:
    """Call the LLM to generate a Cypher query.

    'question' is the user's natural-language question. 
    'llm' is the model adapter used to invoke the prompt.
    'error_feedback', when non-empty, is appended to the prompt as retry context (lets
    us tell the LLM why a previous attempt failed). 
    'doc_filter' and 'confidence' come from Stage 1 and, if provided, hint the LLM to prioritise 
    a specific set of documents in the generated query.

    Returns (cypher, reason). 
    cypher is "" when the LLM declined or parsing failed. 
    The reason carries the explanation in either case.
    """
    if doc_filter:
        files = ", ".join(f"'{f}'" for f in doc_filter)
        doc_filter_block = (
            f"\nDOCUMENT HINT ({confidence} confidence):\n"
            f"An earlier retrieval stage identified these documents as likely relevant:\n"
            f"  {files}\n"
            f"Prioritize these documents in your query (e.g. WHERE d.source_file IN [...]).\n"
            f"You may include other documents if the question warrants broader search.\n"
        )
    else:
        doc_filter_block = ""

    prompt = CYPHER_GEN_PROMPT.format(
        schema=schema_description if schema_description is not None else SCHEMA_DESCRIPTION,
        examples=FEW_SHOT_EXAMPLES,
        question=question,
        doc_filter_block=doc_filter_block,
    )
    if error_feedback:
        prompt += (
            f"\n\nPREVIOUS ATTEMPT FAILED with this error:\n  {error_feedback}\n"
            "Generate a corrected query.\n"
        )

    raw = llm.invoke(prompt, system_msg=CYPHER_SYSTEM_MSG)
    return _parse_llm_response(raw)


# ── Execution ─────────────────────────────────────────────────────────────
def _run_with_timeout(fn, timeout_s: float):
    """Run a zero-arg callable with a wall-clock timeout.

    Python has no built-in wall-clock timeout for arbitrary callables, so we use a single-worker 
    thread pool that runs 'fn' once and is shut down immediately after. 
    'future.result(timeout=...)' is what enforces the deadline.
    """
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn)
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeoutError:
            future.cancel()
            raise TimeoutError(f"Cypher query exceeded {timeout_s}s")


def _execute_cypher(driver, cypher: str, timeout_s: float) -> List[Dict]:
    """Execute a read-only Cypher query with a timeout. Returns list of row dicts."""
    def _run():
        with driver.session() as session:
            return session.execute_read(lambda tx: [dict(r) for r in tx.run(cypher)])
    return _run_with_timeout(_run, timeout_s)


# ── Result formatting ─────────────────────────────────────────────────────
def _format_value(val) -> str:
    """Render a single cell value for the text table."""
    if val is None:
        return ""
    if isinstance(val, (list, tuple)):
        return "[" + ", ".join(_format_value(x) for x in val) + "]"
    if isinstance(val, dict):
        return "{" + ", ".join(f"{key}: {_format_value(val)}" for key, val in val.items()) + "}"
    return str(val)


def _format_rows(cypher: str, rows: List[Dict]) -> str:
    """Render query + rows as a readable text block for the answer LLM."""
    if not rows:
        return ""

    columns = list(rows[0].keys())
    # For each column, the rendered width is the longest cell in that column (header included), 
    # bounded above by 60 so one wide cell does not blow up the table layout.
    widths = {}
    for column in columns:
        cell_widths = [len(_format_value(row.get(column))) for row in rows]
        longest = max([len(column)] + cell_widths)
        widths[column] = min(60, longest)

    def _pad(s: str, w: int) -> str:
        s = s if len(s) <= w else s[: w - 1] + "…"
        return s.ljust(w)

    # Header row: each column name padded to the column's width, joined by " | ".
    # They are padded so columns line up under the separator below.
    header_cells = []
    for column in columns:
        header_cells.append(_pad(column, widths[column]))
    header = " | ".join(header_cells)

    # Visual rule between header and body so the LLM can see the table shape.
    sep = "-+-".join("-" * widths[column] for column in columns)

    # Same padding and column widths as the header so cells line up beneath it.
    body_lines = []
    for row in rows:
        row_cells = []
        for column in columns:
            cell_value = _format_value(row.get(column))
            row_cells.append(_pad(cell_value, widths[column]))
        body_lines.append(" | ".join(row_cells))

    return (
        f"Structural query (generated from your question):\n"
        f"  {cypher}\n\n"
        f"Results ({len(rows)} row{'s' if len(rows) != 1 else ''}):\n"
        f"  {header}\n"
        f"  {sep}\n"
        + "\n".join(f"  {line}" for line in body_lines)
    )


# ── Public API ────────────────────────────────────────────────────────────
def run_cypher_retrieval(
    question: str,
    driver,
    llm,
    cfg=None,
    doc_filter: Optional[List[str]] = None,
    confidence: str = "n/a",
) -> Tuple[str, Dict]:
    """Generate and execute a Cypher query for the question.

    Returns (context_string, metadata). 
    context_string is "" when the stage is skipped (LLM declined, empty result, unsafe, or error). 
    In that case metadata carries the reason/error for observability.
    """
    if cfg is None:
        cfg = default_config

    max_rows = getattr(cfg, "CYPHER_MAX_ROWS", 100)
    timeout_s = getattr(cfg, "CYPHER_TIMEOUT_S", 10)

    meta: Dict = {"query": None, "n_rows": 0, "error": None, "reason": None}

    # High-level structure of this function:
    #   Generate -> Safety check -> Schema validate (with one retry) -> Execute (with one retry) -> Format
    # Each retry feeds the previous error back to the LLM as extra context.

    # Load valid triples from Neo4j schema
    try:
        valid_triples = _load_valid_triples(driver)
    except Exception:
        valid_triples = set()  # degrade gracefully and skip validation

    # Build (and cache) the schema description text used in the Cypher LLM prompt.
    # In schema-free mode this queries the live graph for entity types and rel labels.
    try:
        schema_description = _get_schema_description(cfg, driver)
    except Exception:
        schema_description = SCHEMA_DESCRIPTION  # fall back to the static description

    # ── Generate ───────────────────────────────────────────────────────
    try:
        cypher, reason = _generate_cypher(
            question, llm, doc_filter=doc_filter, confidence=confidence,
            schema_description=schema_description,
        )
    except (json.JSONDecodeError, ValueError) as e:
        meta["error"] = f"LLM response not valid JSON: {e}"
        return "", meta

    if not cypher:
        meta["reason"] = reason or "LLM declined"
        return "", meta

    # ── Safety ─────────────────────────────────────────────────────────
    is_safe, unsafe_reason = _is_safe_query(cypher)
    if not is_safe:
        meta["error"] = f"Unsafe query rejected: {unsafe_reason}"
        meta["query"] = cypher
        return "", meta

    # ── Schema validation ─────────────────────────────────────────────
    if valid_triples:
        is_valid, schema_err = _validate_schema(cypher, valid_triples)
        if not is_valid:
            # Retry with the schema error as feedback
            try:
                retry_cypher, retry_reason = _generate_cypher(
                    question, llm, error_feedback=schema_err,
                    doc_filter=doc_filter, confidence=confidence,
                    schema_description=schema_description,
                )
            except (json.JSONDecodeError, ValueError) as e2:
                meta["error"] = f"schema validation failed ({schema_err}); retry parse error: {e2}"
                meta["query"] = cypher
                return "", meta

            if not retry_cypher:
                meta["error"] = f"schema validation failed ({schema_err}); LLM declined on retry"
                meta["query"] = cypher
                meta["reason"] = retry_reason
                return "", meta

            is_safe, unsafe_reason = _is_safe_query(retry_cypher)
            if not is_safe:
                meta["error"] = f"schema validation failed; retry unsafe: {unsafe_reason}"
                meta["query"] = retry_cypher
                return "", meta

            is_valid, schema_err2 = _validate_schema(retry_cypher, valid_triples)
            if not is_valid:
                meta["error"] = f"schema validation failed twice: {schema_err2}"
                meta["query"] = retry_cypher
                return "", meta

            cypher = retry_cypher

    cypher = _ensure_limit(cypher, max_rows)
    meta["query"] = cypher

    # ── Execute (first attempt) ────────────────────────────────────────
    try:
        rows = _execute_cypher(driver, cypher, timeout_s)
    except Exception as e:
        # One retry: feed the error back to the LLM and try again
        err_msg = str(e)
        try:
            retry_cypher, retry_reason = _generate_cypher(
                question, llm, error_feedback=err_msg,
                doc_filter=doc_filter, confidence=confidence,
                schema_description=schema_description,
            )
        except (json.JSONDecodeError, ValueError) as e2:
            meta["error"] = f"first attempt failed ({err_msg}); retry parse error: {e2}"
            return "", meta

        if not retry_cypher:
            meta["error"] = f"first attempt failed ({err_msg}); LLM declined on retry"
            meta["reason"] = retry_reason
            return "", meta

        is_safe, unsafe_reason = _is_safe_query(retry_cypher)
        if not is_safe:
            meta["error"] = f"first attempt failed ({err_msg}); retry unsafe: {unsafe_reason}"
            meta["query"] = retry_cypher
            return "", meta

        retry_cypher = _ensure_limit(retry_cypher, max_rows)
        meta["query"] = retry_cypher

        try:
            rows = _execute_cypher(driver, retry_cypher, timeout_s)
        except Exception as e2:
            meta["error"] = f"first attempt failed ({err_msg}); retry failed ({e2})"
            return "", meta

        cypher = retry_cypher

    # ── Format ─────────────────────────────────────────────────────────
    meta["n_rows"] = len(rows)
    if not rows:
        return "", meta

    # Check if this is a retrieval query (returns chunk_ids)
    if "chunk_id" in rows[0]:
        meta["chunk_ids"] = [r["chunk_id"] for r in rows if r.get("chunk_id")]
        # Don't format as a table. Chunks will be fetched by the pipeline
        return "", meta

    return _format_rows(cypher, rows), meta
