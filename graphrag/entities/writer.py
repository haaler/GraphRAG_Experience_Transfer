"""Neo4j write side of the entity stage.

Called by the entity orchestrator once per chunk that produced any entities, after extraction and filtering.
The two public functions write:

  - Entity nodes, keyed by `merge_key` (lowercased and whitespace collapsed name).
    One Entity node per logical entity, no matter how many chunks mention it.
    The contextual role (entity_type, entity_class) lives on the HAS_ENTITY edge so the same entity 
    can play different roles in different chunks.
  - HAS_ENTITY edges from the source Chunk to each Entity, carrying the role and the evidence string.
  - Lightweight EntityType nodes plus IS_A edges so a reader can browse the type hierarchy in Neo4j.
  - Typed edges between Entity nodes (one Cypher template per relationship label declared in schemas.py).
"""

import re
from typing import List, Dict, Optional

from graphrag.entities.schemas import ALL_RELATIONSHIP_TYPES


def write_entities_to_graph(tx, chunk_id: str, entities: List[Dict]) -> None:
    """
    Upsert Entity nodes and wire HAS_ENTITY edges to their source Chunk.

    Entity identity key: merge_key (lowercased, whitespace-normalized name).
    "Semco" as Supplier and "Semco" as Client -> ONE node, TWO edges.
    The contextual role (entity_type, entity_class) lives on the edge.
    The display-quality name is stored in e.name (last writer wins, but
    canonicalize_entity_name already produces consistent forms).
    """
    for entity in entities:
        name = entity["name"]
        merge_key = " ".join(name.lower().split())  # lowercase + collapse whitespace
        entity_type = entity.get("entity_type", "Unknown")
        entity_class = entity.get("entity_class", "")
        evidence = entity.get("evidence", "")

        # Upsert Entity node + HAS_ENTITY edge to source Chunk
        tx.run(
            "MERGE (e:Entity {merge_key: $merge_key}) "
            "SET e.name          = $name, "
                "e.entity_class  = $entity_class, "
                "e.entity_type   = $entity_type "
            "WITH e "
            "MATCH (c:Chunk {chunk_id: $chunk_id}) "
            "MERGE (c)-[r:HAS_ENTITY]->(e) "
            "SET r.entity_type  = $entity_type, "
                "r.entity_class = $entity_class, "
                "r.evidence     = $evidence",
            merge_key=merge_key,
            name=name,
            entity_type=entity_type,
            entity_class=entity_class,
            chunk_id=chunk_id,
            evidence=evidence,
        )

        # Create EntityType node + IS_A edge
        if entity_type and entity_type != "Unknown":
            tx.run(
                "MERGE (et:EntityType {name: $etype}) "
                "WITH et "
                "MATCH (e:Entity {merge_key: $merge_key}) "
                "MERGE (e)-[:IS_A]->(et)",
                etype=entity_type, merge_key=merge_key,
            )


# ── Pre-built Cypher for each known relationship type ────────────────────
# Cypher does not let us parameterise the edge label, so the label has to be baked into the query string. 
# We have a fixed list of edge types in ALL_RELATIONSHIP_TYPES (SUPPLIER_FOR, CLIENT_OF, LOCATED_AT, ...), 
# so for each one we generate a Cypher query string that has that label baked in.
# The result is a dict shaped like:
#   {
#     "SUPPLIER_FOR": "MATCH ... MERGE (s)-[r:SUPPLIER_FOR]->(t) ...",
#     "CLIENT_OF":    "MATCH ... MERGE (s)-[r:CLIENT_OF]->(t) ...",
#     "LOCATED_AT":   "MATCH ... MERGE (s)-[r:LOCATED_AT]->(t) ...",
#     ...
#   }
# We do this once at module load instead of rebuilding the string inside
# write_relationships_to_graph for every relationship. At write time the loop
# just looks up the template by type and runs it.
_REL_CYPHER = {}
for rel_type in ALL_RELATIONSHIP_TYPES:
    _REL_CYPHER[rel_type] = (
        f"MATCH (s:Entity {{merge_key: $source_key}}), "
        f"(t:Entity {{merge_key: $target_key}}) "
        f"MERGE (s)-[r:{rel_type}]->(t) "
        f"SET r.source_chunk = $chunk_id"
    )


# Cypher relationship labels must start with a letter and contain only letters, digits, and
# underscores. We sanitise here because in schema-free mode the label comes straight from the
# LLM and gets embedded into the Cypher string (Cypher cannot parameterise edge labels).
_SAFE_LABEL_CHARS = re.compile(r"[^A-Z0-9_]")

def _safe_rel_label(label: str) -> Optional[str]:
    """Normalise an LLM-supplied relationship label into a Cypher-safe form.

    Returns None when the label is empty, doesn't start with a letter after cleaning, or has
    no surviving characters. The single caller skips relationships that come back as None.
    """
    if not label:
        return None
    cleaned = label.strip().upper().replace(" ", "_").replace("-", "_")
    cleaned = _SAFE_LABEL_CHARS.sub("", cleaned)
    if not cleaned or not cleaned[0].isalpha():
        return None
    return cleaned


def write_relationships_to_graph(tx, chunk_id: str, relationships: List[Dict]) -> None:
    """Write typed relationship edges between Entity nodes.

    Each relationship creates a typed Neo4j edge (e.g. -[:SUPPLIER_FOR]->)
    with a source_chunk property for traceability. Entities are matched by
    merge_key (lowercased name) for consistency with write_entities_to_graph.

    Schema-on path: the rel type is already in ALL_RELATIONSHIP_TYPES so the pre-built template
    in _REL_CYPHER is used directly.
    Schema-off path: the LLM picked the label, so we sanitise it and build the Cypher on the spot.
    """
    for rel in relationships:
        rel_type = rel.get("type", "")
        cypher = _REL_CYPHER.get(rel_type)

        if cypher is None:
            safe_label = _safe_rel_label(rel_type)
            if safe_label is None:
                continue
            cypher = (
                f"MATCH (s:Entity {{merge_key: $source_key}}), "
                f"(t:Entity {{merge_key: $target_key}}) "
                f"MERGE (s)-[r:{safe_label}]->(t) "
                f"SET r.source_chunk = $chunk_id"
            )

        tx.run(
            cypher,
            source_key=" ".join(rel["source"].lower().split()),
            target_key=" ".join(rel["target"].lower().split()),
            chunk_id=chunk_id,
        )
