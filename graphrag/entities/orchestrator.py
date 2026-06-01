"""Entity stage orchestrator.

Called once per indexing run, after chunking is done. The single public function 'extract_and_write_all_entities'
walks every chunk of every document, runs the per-chunk pipeline (extract -> filter -> canonicalise -> write)
on a thread pool, and prints a normalisation report at the end.

This file owns the shared state for the run.
The actual extraction call lives in extraction.py, the filters and 
canonicalisation in filters.py, and the Neo4j writes in writer.py.
"""

import time
import threading
from typing import List, Dict, Set
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from graphrag import config as default_config
from graphrag.entities.schemas import RELATIONSHIPS, create_entity_indexes
from graphrag.entities.extraction import extract_entities_from_chunk
from graphrag.entities.filters import (
    canonicalize_entity_name, filter_entities, filter_relationships,
)
from graphrag.entities.writer import write_entities_to_graph, write_relationships_to_graph


def _process_chunk(
        chunk, 
        doc, 
        llm, 
        driver, 
        skip_lists, 
        skip_sections, 
        name_variants=None, 
        name_variants_lock=None
    ):
    """Process a single chunk: extract → filter → canonicalize → write.

    Returns (status, entity_count, relationship_count).
    """
    section_lower = chunk.get("section_title", "").lower()
    chunk_type = chunk["chunk_type"]

    if any(pat in section_lower for pat in skip_sections):
        return "skipped", 0, 0
    if skip_lists and chunk_type == "list":
        return "skipped", 0, 0

    try:
        entities, relationships = extract_entities_from_chunk(chunk, doc, llm)
        entities = filter_entities(entities, chunk_type=chunk_type, source_file=chunk["source_file"])

        # Canonicalize entity names. 
        # The raw name (what the LLM extracted) is kept in _raw_name so the relationship canonicalisation loop
        # below can match a raw relationship endpoint back to its entity even after
        # the entity name has been renamed (e.g. "BP" -> "AkerBP").
        for e in entities:
            e["_raw_name"] = e.get("name", "")
            e["name"] = canonicalize_entity_name(e.get("name", ""), e.get("entity_type", ""))

        # Track every distinct surface form that maps to the same merge_key.
        # The normalisation report at the end of the run uses this to show
        # which name variants got collapsed into the same Entity node.
        if name_variants is not None and name_variants_lock is not None:
            with name_variants_lock:
                for e in entities:
                    key = " ".join(e["name"].lower().split())
                    name_variants[key].add(e["name"])

        # Build maps for relationship filtering
        valid_names = {e["name"] for e in entities}
        type_map = {e["name"]: e["entity_type"] for e in entities}

        # Canonicalize source/target names in relationships so they match the canonical entity names from the loop above. 
        # For each relationship endpoint we look up the matching entity (by either raw or canonical name) 
        # to pull its entity_type, then run canonicalize_entity_name on the endpoint with that type.
        # Without this, filter_relationships would drop any relationship whose endpoints were renamed by canonicalisation.
        for r in relationships:
            src_type = ""
            tgt_type = ""
            # Try to find the entity_type for source/target from raw extractions
            for e in entities:
                raw = e.get("_raw_name", e["name"])
                if raw == r.get("source") or e["name"] == r.get("source"):
                    src_type = e.get("entity_type", "")
                if raw == r.get("target") or e["name"] == r.get("target"):
                    tgt_type = e.get("entity_type", "")
            r["source"] = canonicalize_entity_name(r.get("source", ""), src_type)
            r["target"] = canonicalize_entity_name(r.get("target", ""), tgt_type)

        # Filter relationships against schema and surviving entities. In schema-free mode the
        # schema list is empty, so filter_relationships keeps any non-self-loop with both endpoints present.
        use_schema = getattr(default_config, "USE_CONCEPTUAL_SCHEMA", True)
        active_rel_schema = RELATIONSHIPS if use_schema else []
        relationships = filter_relationships(relationships, valid_names, active_rel_schema, type_map)

        # Write to Neo4j
        if entities or relationships:
            with driver.session() as session:
                if entities:
                    session.execute_write(write_entities_to_graph, chunk["chunk_id"], entities)
                if relationships:
                    session.execute_write(write_relationships_to_graph, chunk["chunk_id"], relationships)
        return "processed", len(entities), len(relationships)
    except Exception as e:
        print(f"    [ERROR] chunk {chunk['chunk_id'][:8]}: {e}")
        return "error", 0, 0


def extract_and_write_all_entities(
        documents: List[Dict], 
        driver, 
        llm,
        cfg=None,
    ) -> Dict:
    """Extract entities from every chunk of every document and write them to Neo4j.

    'documents' is the list produced by the chunking stage (each dict has a 'chunks' list).
    'driver' is an open Neo4j driver, 'llm' is the model used by every per-chunk extraction call.
    'cfg' is the config module (defaults to graphrag.config) and
    provides EXTRACTION_WORKERS, SKIP_LIST_CHUNKS and DEFAULT_SKIP_SECTIONS.

    Chunks within each document run on a thread pool of size cfg.EXTRACTION_WORKERS (default 4),
    so the order of writes inside a document is not deterministic.

    Returns a stats dict with chunks_processed, chunks_skipped, entities_extracted, entities_written,
    relationships_extracted, relationships_written, entities_unique, and entities_normalized.
    """
    if cfg is None:
        cfg = default_config

    # Section titles to skip are read once per run from config. Lowercased here
    # so the per-chunk substring check is case insensitive.
    default_skip = getattr(cfg, "DEFAULT_SKIP_SECTIONS", set())
    skip_sections = {s.lower() for s in default_skip}

    skip_lists = getattr(cfg, "SKIP_LIST_CHUNKS", False)
    workers = getattr(cfg, "EXTRACTION_WORKERS", 4)

    # Thread-safe tracker: merge_key → {name_variant_1, name_variant_2, ...}
    name_variants: Dict[str, Set[str]] = defaultdict(set)
    name_variants_lock = threading.Lock()

    # Run once per pipeline run so every later write can rely on the unique
    # constraint on Entity.merge_key being enforced.
    with driver.session() as session:
        session.execute_write(create_entity_indexes)

    stats = {
        "chunks_processed": 0,
        "chunks_skipped":   0,
        "entities_extracted": 0,
        "entities_written":   0,
        "relationships_extracted": 0,
        "relationships_written":   0,
    }

    pipeline_start = time.time()

    for doc_idx, doc in enumerate(documents):
        source_file = doc["source_file"]
        chunks = doc.get("chunks", [])
        doc_ent = 0
        doc_rel = 0
        doc_start = time.time()
        print(f"\n[{doc_idx+1}/{len(documents)}] {source_file}  ({len(chunks)} chunks, {workers} workers)")

        # Fan out chunks to the thread pool. 
        # Each worker runs the per-chunk pipeline (extract -> filter -> canonicalise -> write).
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_process_chunk, chunk, doc, llm, driver, skip_lists, 
                            skip_sections, name_variants, name_variants_lock): chunk
                for chunk in chunks
            }

            for future in as_completed(futures):
                status, ent_count, rel_count = future.result()
                if status == "skipped":
                    stats["chunks_skipped"] += 1
                elif status == "processed":
                    stats["chunks_processed"] += 1
                    stats["entities_extracted"] += ent_count
                    stats["entities_written"] += ent_count
                    stats["relationships_extracted"] += rel_count
                    stats["relationships_written"] += rel_count
                    doc_ent += ent_count
                    doc_rel += rel_count
                else:  # error
                    stats["chunks_processed"] += 1

        doc_elapsed = time.time() - doc_start
        print(f"  -> {doc_ent} entities, {doc_rel} relationships written  ({doc_elapsed:.1f}s)")

    total_elapsed = time.time() - pipeline_start

    # ── Normalization report ─────────────────────────────────────────────
    with driver.session() as session:
        unique_count = session.run("MATCH (e:Entity) RETURN count(e) AS n").single()["n"]

    total_mentions = stats['entities_extracted']
    print(f"\nEntity extraction complete in {total_elapsed:.0f}s "
          f"({stats['chunks_processed']} chunks, "
          f"{total_mentions} mentions → {unique_count} unique entities, "
          f"{stats['relationships_written']} relationships)")

    # Show which name variants were collapsed into the same merge_key
    collapsed = {key: val for key, val in name_variants.items() if len(val) > 1}
    if collapsed:
        print(f"\nNormalization merged {len(collapsed)} entities with variant names:")
        for key_2, variants in sorted(collapsed.items()):
            print(f"  '{key_2}' ← {sorted(variants)}")
    else:
        print("\nNo name variants detected (all names already consistent).")

    stats["entities_unique"] = unique_count
    stats["entities_normalized"] = len(collapsed)
    return stats
