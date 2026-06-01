"""LLM based entity and relationship extraction from a single chunk.

This module is the LLM call site of the entity stage. 
The orchestrator calls extract_entities_from_chunk once per chunk.
This builds the extraction prompt from the chunk text, the document context, and the schema, sends
it to the LLM, and parses the JSON response into raw entity and relationship lists.
The lists then go through filters.py and finally to the entity writer.

Large tables are split into row group batches so each LLM call stays within
EXTRACTION_MAX_CHUNK_CHARS. The per-batch results are merged at the end with
a small dedup pass since the same entity often appears in adjacent rows.
"""

import json
import time
from typing import List, Dict, Tuple

from graphrag import config as default_config
from graphrag.chunking import chunk_to_text
from graphrag.entities.schemas import (
    CORE_SCHEMA,
    RELATIONSHIPS,
    detect_document_type,
    format_schema_for_prompt,
    format_relationships_for_prompt,
)


# The RULES section below is what shapes extraction quality. 
# Edits there directly move the precision/recall trade-off, and careful tuning is therefore needed.
ENTITY_EXTRACTION_PROMPT = """\
Extract named entities and relationships from the document chunk below.

Document context:
  Type   : {doc_type}
  Section: {section_path}
  File   : {source_file}
{abbreviations_block}

Entity schema (use as guidance for naming entity_class and entity_type):
{schema_text}

Relationship schema (valid relationship types between entities):
{relationship_text}

Chunk ({chunk_type}):
{chunk_text}

RULES — read carefully:

IMPORTANT: Only extract entities that appear IN THE CHUNK TEXT above.
Do NOT extract names from these instructions, rules, or examples.
All examples below are for illustration only — they show what KIND of text to
extract or skip. Do NOT output any example name as an entity.

1. Extract SPECIFIC, NAMED entities only — things with a proper name, identifier,
   or well-known domain term that appears in the chunk text.
   GOOD: company names, rig names, equipment with brand/model, person names,
         contract identifiers, location names
   BAD:  generic descriptions, section titles, document structural labels

2. entity_class vs entity_type: the schema shows the hierarchy. Use the GROUP
   name (e.g. "BasicInformation", "Personnel") as entity_class, and the SPECIFIC
   type (e.g. "Supplier", "ContactPerson") as entity_type.

3. Equipment — must be identified by BRAND + MODEL, a part/tag number, or a
   recognized product name. A longer description does NOT make it more specific.
   The test: does it name a MANUFACTURER or contain a MODEL/TAG NUMBER?
   YES → extract.  NO → skip.
   BAD (no brand, no model, no tag — do NOT extract these kinds of names):
     "Blind Flanges To Gumbo Dump Line", "Filter In And Filter Out",
     "Remote Operated Panel", "Isolation Valve For Winch", "Sheaves",
     "Mud Pump", "control cabinet", "NDT Report"

4. Reference — a specific, identifiable cross-reference with a name, number,
   or code. Must be something you could use to look up a specific document,
   contract, project, or classification.
   BAD (generic labels that appear in many documents — do NOT extract):
     "Scope of Work", "Technical Purchase Specification", "CTR Sheet",
     "Cuttings handling", "option 2", "Appendix A", "Figure 3.2", "table 2.1"
   When a company name modifies "contract"/"campaign"/"project", extract the
   FULL phrase as Reference, not the company as Organization.

5. AnalysisMethod — a NAMED, ESTABLISHED method, theory, or formula you could
   find in a textbook or engineering standard.
   BAD (parameter values, solver settings, generic terms — do NOT extract):
     "10deg end heading", "explicit solver", "dynamic analysis", "modelling",
     "Marine", "Environment variables", "DoF", "rig coordinate system"

6. StandardReference — must include an alphanumeric code (e.g. "NORSOK D-010").
   An organization name alone ("API", "ISO", "DNV") without a code is NOT a standard.

7. Disambiguation:
   - Organization = COMPANY (signs contracts, employs people). Location = PLACE.
   - ContactPerson = HUMAN NAME. Short uppercase abbreviations (ESD, BOP, ROV)
     are technical terms, not person initials — only treat as person in
     signature blocks or distribution lists.
   - "Deepsea ___" or "West ___" + place name → RigInstallation, never Organization.

8. Provide the exact source phrase as evidence for each entity.

RELATIONSHIP RULES:
- Extract relationships ONLY between entities you extracted above.
- The relationship "type" must match one from the relationship schema.
- Source and target entity types must match the schema constraints.
- There must be textual evidence — co-occurrence alone is not enough.

Return ONLY valid JSON — no explanation, no markdown:
{{"entities": [{{"name": "...", "entity_class": "...", "entity_type": "...", "evidence": "..."}}], "relationships": [{{"source": "...", "target": "...", "type": "..."}}]}}

If no qualifying entities or relationships found: {{"entities": [], "relationships": []}}
"""

ENTITY_EXTRACTION_SYSTEM_MSG = (
    "You are a precise named entity and relationship extractor for offshore drilling documents. "
    "Extract the specific named entities and relationships that the chunk text clearly supports. "
    "Use the provided conceptual schema as your priority guide; do not invent entities or "
    "force-fit types that the text does not clearly support."
)


# Schema-free variant of the prompt above.
# Used when cfg.USE_CONCEPTUAL_SCHEMA is False — the LLM picks entity_type, entity_class, and
# relationship labels on its own with no curated vocabulary.
# Mirrors ENTITY_EXTRACTION_PROMPT one-for-one, minus the two schema blocks.
# The general quality rules (proper-noun guidance, no generic terms, evidence required) are kept
# since they are about what counts as a real entity, not about which type to assign.
ENTITY_EXTRACTION_PROMPT_SCHEMA_FREE = """\
Extract named entities and relationships from the document chunk below.

Document context:
  Type   : {doc_type}
  Section: {section_path}
  File   : {source_file}
{abbreviations_block}

Chunk ({chunk_type}):
{chunk_text}

RULES — read carefully:

IMPORTANT: Only extract entities that appear IN THE CHUNK TEXT above.
Do NOT extract names from these instructions, rules, or examples.
All examples below are for illustration only — they show what KIND of text to
extract or skip. Do NOT output any example name as an entity.

1. Extract SPECIFIC, NAMED entities only — things with a proper name, identifier,
   or well-known domain term that appears in the chunk text.
   GOOD: company names, rig names, equipment with brand/model, person names,
         contract identifiers, location names
   BAD:  generic descriptions, section titles, document structural labels

2. For each entity, pick a short entity_type that captures what kind of thing it is
   (e.g. "Company", "Rig", "Person", "Equipment", "Standard", "Contract")
   and an entity_class that groups related types (e.g. "Organization", "Place",
   "Personnel", "Scope"). Use Capitalized single words for both. Be consistent
   within the same chunk.

3. Equipment-like entities should be identified by BRAND + MODEL, a part/tag number,
   or a recognized product name. A longer description does NOT make it more specific.
   BAD (no brand, no model, no tag — do NOT extract these kinds of names):
     "Blind Flanges To Gumbo Dump Line", "Filter In And Filter Out",
     "Remote Operated Panel", "Isolation Valve For Winch", "Sheaves",
     "Mud Pump", "control cabinet", "NDT Report"

4. Cross-references — only extract them when they are something you could use to look
   up a specific document, contract, project, or classification (a name, number, or code).
   BAD (generic labels that appear in many documents — do NOT extract):
     "Scope of Work", "Technical Purchase Specification", "CTR Sheet",
     "Cuttings handling", "option 2", "Appendix A", "Figure 3.2", "table 2.1"
   When a company name modifies "contract"/"campaign"/"project", extract the
   FULL phrase, not the company alone.

5. Methods — extract only NAMED, ESTABLISHED methods, theories, or formulas
   (e.g. JONSWAP spectrum, Morison equation, FMEA, HAZOP).
   BAD (parameter values, solver settings, generic terms — do NOT extract):
     "10deg end heading", "explicit solver", "dynamic analysis", "modelling",
     "Marine", "Environment variables", "DoF", "rig coordinate system"

6. Industry standards must include an alphanumeric code (e.g. "NORSOK D-010").
   An organization name alone ("API", "ISO", "DNV") without a code is NOT a standard.

7. Disambiguation:
   - A company is a COMPANY (signs contracts, employs people). A place is a PLACE.
   - A person is a HUMAN NAME. Short uppercase abbreviations (ESD, BOP, ROV)
     are technical terms, not person initials — only treat as person in
     signature blocks or distribution lists.
   - "Deepsea ___" or "West ___" + place name → rig, never company.

8. Provide the exact source phrase as evidence for each entity.

RELATIONSHIP RULES:
- Extract relationships ONLY between entities you extracted above.
- Pick a short ALL_CAPS_WITH_UNDERSCORES label for each relationship type
  (e.g. SUPPLIES, OWNS, LOCATED_AT, WORKS_FOR, COMPLIES_WITH).
- There must be textual evidence — co-occurrence alone is not enough.
- Be consistent: if you label one supplier→rig edge as SUPPLIES, label every
  similar edge in this chunk the same way.

Return ONLY valid JSON — no explanation, no markdown:
{{"entities": [{{"name": "...", "entity_class": "...", "entity_type": "...", "evidence": "..."}}], "relationships": [{{"source": "...", "target": "...", "type": "..."}}]}}

If no qualifying entities or relationships found: {{"entities": [], "relationships": []}}
"""

ENTITY_EXTRACTION_SYSTEM_MSG_SCHEMA_FREE = (
    "You are a precise named entity and relationship extractor for offshore drilling documents. "
    "Extract the specific named entities and relationships that the chunk text clearly supports. "
    "Pick your own short labels for entity_type, entity_class, and relationship type — do not invent "
    "entities or force-fit types that the text does not clearly support."
)


def _table_to_extraction_batches(chunk: Dict) -> List[str]:
    """Convert a table chunk into one or more text batches for entity extraction.

    Each row is formatted as "Col1: val | Col2: val" using the header row for column names. 
    Rows are grouped into batches that fit within the per-call character cap so the LLM processes manageable pieces. 
    Smaller tables produce a single batch.
    """
    max_chars = getattr(default_config, "EXTRACTION_MAX_CHUNK_CHARS", 2800)

    rows = chunk["content"].get("rows", [])
    if not rows:
        return []

    # Use first row as column headers
    headers = [str(c).strip() for c in rows[0]]
    data_rows = rows[1:] if len(rows) > 1 else rows

    # Format each row as "Header1: val | Header2: val"
    formatted_lines = []
    for row in data_rows:
        cells = [str(c).strip() for c in row]
        parts = []
        for i, cell in enumerate(cells):
            # Some rows have more cells than the header has columns.
            # Fall back to a generic "Col1", "Col2", ... label for those extra cells.
            if i < len(headers):
                col_name = headers[i]
            else:
                col_name = f"Col{i+1}"
            if cell:
                parts.append(f"{col_name}: {cell}")
        if parts:
            formatted_lines.append(" | ".join(parts))

    if not formatted_lines:
        return []

    # Group lines into batches that fit within the per-call character cap
    batches = []
    current_lines = []
    current_len = 0

    for line in formatted_lines:
        line_len = len(line) + 1  # +1 for newline
        if current_lines and current_len + line_len > max_chars:
            batches.append("\n".join(current_lines))
            current_lines = []
            current_len = 0
        current_lines.append(line)
        current_len += line_len

    if current_lines:
        batches.append("\n".join(current_lines))

    return batches


def _prepare_extraction_texts(chunk: Dict) -> List[str]:
    """Convert any chunk into a list of text batches for entity extraction.

    Tables may produce multiple batches (one per row group).
    Text and list chunks always produce a single batch.
    """
    if chunk["chunk_type"] == "table":
        return _table_to_extraction_batches(chunk)
    max_chars = getattr(default_config, "EXTRACTION_MAX_CHUNK_CHARS", 2800)
    text = chunk_to_text(chunk)[:max_chars]
    return [text] if text.strip() else []


def _extract_single(prompt: str, llm, chunk_id_prefix: str) -> Tuple[List[Dict], List[Dict]]:
    """Run a single LLM extraction call and parse the result.

    Retries exactly once and only when the LLM returns an empty body or a partial JSON ("Expecting value"). 
    Any other exception fails fast with a [WARN] log and an empty result, so one broken chunk
    does not stall the full extraction run.

    Returns (entities, relationships).
    """
    ctx_size = getattr(default_config, "EXTRACTION_CTX_SIZE", 4096)
    system_msg = (
        ENTITY_EXTRACTION_SYSTEM_MSG
        if getattr(default_config, "USE_CONCEPTUAL_SCHEMA", True)
        else ENTITY_EXTRACTION_SYSTEM_MSG_SCHEMA_FREE
    )

    for attempt in range(2):
        try:
            t0 = time.time()
            raw = llm.invoke(prompt, system_msg=system_msg, num_ctx=ctx_size)
            elapsed = time.time() - t0
            raw = raw.strip()
            if not raw:
                if attempt == 0:
                    print(f"    [RETRY] chunk {chunk_id_prefix}: empty response ({elapsed:.1f}s), retrying...")
                    continue
                print(f"    [WARN] chunk {chunk_id_prefix}: empty response after retry ({elapsed:.1f}s)")
                return [], []
            
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            result = json.loads(raw)
            entities = result.get("entities", [])
            relationships = result.get("relationships", [])

            if elapsed > 10:
                print(f"    [SLOW] chunk {chunk_id_prefix}: {elapsed:.1f}s, {len(entities)} entities, {len(relationships)} rels")
            return entities, relationships
        except Exception as e:
            elapsed = time.time() - t0
            if attempt == 0 and "Expecting value" in str(e):
                print(f"    [RETRY] chunk {chunk_id_prefix}: {e} ({elapsed:.1f}s), retrying...")
                continue
            print(f"    [WARN] chunk {chunk_id_prefix}: {e} ({elapsed:.1f}s)")
            return [], []
    return [], []


def extract_entities_from_chunk(chunk: Dict, doc: Dict, llm) -> Tuple[List[Dict], List[Dict]]:
    """
    Extract entities and relationships from a single chunk using the LLM.

    Large tables are split into batches and processed with separate LLM calls, then merged and deduplicated.
    All entities are linked to the same chunk_id.

    Returns (entities, relationships).
    """
    batches = _prepare_extraction_texts(chunk)
    if not batches:
        return [], []

    chunk_id_prefix = chunk["chunk_id"][:8]

    if len(batches) > 1:
        print(f"    [BATCH] chunk {chunk_id_prefix}: table split into {len(batches)} batches")

    use_schema = getattr(default_config, "USE_CONCEPTUAL_SCHEMA", True)

    all_entities = []
    all_relationships = []
    for batch_idx, batch_text in enumerate(batches):
        batch_label = (f"{chunk_id_prefix}[{batch_idx+1}/{len(batches)}]" if len(batches) > 1 else chunk_id_prefix)

        abbrevs = doc.get("abbreviations") or {}
        if abbrevs:
            abbrev_lines = "  Abbreviations defined in this document:\n" + "\n".join(
                f"    {key}: {val}" for key, val in sorted(abbrevs.items())
            )
        else:
            abbrev_lines = ""

        # Build the full prompt and run the LLM call.
        # The schema-free variant drops the entity/relationship schema blocks so the LLM picks
        # its own type and label vocabulary.
        if use_schema:
            prompt = ENTITY_EXTRACTION_PROMPT.format(
                doc_type=detect_document_type(doc),
                section_path=chunk.get("section_path", chunk["section_title"]),
                source_file=chunk["source_file"],
                abbreviations_block=abbrev_lines,
                schema_text=format_schema_for_prompt(CORE_SCHEMA),
                relationship_text=format_relationships_for_prompt(RELATIONSHIPS),
                chunk_type=chunk["chunk_type"],
                chunk_text=batch_text,
            )
        else:
            prompt = ENTITY_EXTRACTION_PROMPT_SCHEMA_FREE.format(
                doc_type=detect_document_type(doc),
                section_path=chunk.get("section_path", chunk["section_title"]),
                source_file=chunk["source_file"],
                abbreviations_block=abbrev_lines,
                chunk_type=chunk["chunk_type"],
                chunk_text=batch_text,
            )
        entities, relationships = _extract_single(prompt, llm, batch_label)
        all_entities.extend(entities)
        all_relationships.extend(relationships)

    # Dedup only when a table was split into multiple batches. 
    # Single batch chunks have no cross-batch duplicates to merge, so we skip the work.
    # Adjacent table rows often mention the same entity, hence the dedup pass.
    if len(batches) > 1:
        seen_ent = set()
        unique_ent = []
        for e in all_entities:
            key = (e.get("name", "").lower().strip(), e.get("entity_type", "").lower().strip())
            if key not in seen_ent:
                seen_ent.add(key)
                unique_ent.append(e)
        all_entities = unique_ent

        seen_rel = set()
        unique_rel = []
        for r in all_relationships:
            key = (r.get("source", "").lower().strip(),
                   r.get("target", "").lower().strip(),
                   r.get("type", "").upper().strip())
            if key not in seen_rel:
                seen_rel.add(key)
                unique_rel.append(r)
        all_relationships = unique_rel

    return all_entities, all_relationships
