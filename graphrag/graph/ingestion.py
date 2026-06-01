"""
Graph ingestion orchestrator.

This module sits between the parsing/chunking stage and the entity extraction stage of the pipeline.
It takes the list of parsed document dictionaries and walks through them one by one, 
asking Neo4j whether each one is a fresh revision worth keeping, then writing the document and all of its surrounding
nodes (project, document type, discipline, sections, chunks, describes edges) in the right order.
"""

from typing import List, Dict

from graphrag.graph.schema import create_constraints
from graphrag.graph.writers import (
    check_supersession,
    write_document_node,
    write_project_node,
    write_document_type_node,
    write_discipline_node,
    write_sections_and_chunks,
    write_describes_edges,
    supersede_document,
)

def ingest_all_documents(documents: List[Dict], driver) -> Dict:
    """Write every parsed document to Neo4j with revision-aware supersession.

    'documents' is the list produced by the parsing and chunking stage.
    Each dict carries the metadata fields (source_file, base_doc_id, revision, project, document_type, discipline, ...)
    plus the section and chunk payload that the writers expect. 
    'driver' is an open Neo4j driver; the function opens its own session and runs every write inside that session.

    For each document we first ask Neo4j whether something with the same base_doc_id is already there:
      1. New revision is higher        -> supersede the existing one, ingest new
      2. New revision is same or lower -> skip
      3. No existing document found    -> ingest freely

    Returns a bookkeeping dict with three lists:
      "ingested"   source_file values that were written this run (this also
                   covers documents that triggered a supersession).
      "superseded" tuples of (old_source_file, new_source_file) for every pair
                   where the new revision replaced the old one.
      "skipped"    source_file values that were skipped because a current or
                   newer revision is already in the graph.
    """
    results = {"ingested": [], "superseded": [], "skipped": []}

    with driver.session() as session:

        # Constraints are created once per run so every later write can rely
        # on uniqueness (Document.source_file, Chunk.id, etc.) being enforced.
        session.execute_write(create_constraints)

        for doc in documents:
            source_file = doc["source_file"]

            # Decide the outcome for this document before any write touches the graph.
            # supersedes is either None or the source_file of an older revision that this one replaces.
            should_ingest, supersedes = session.execute_read(check_supersession, doc)

            if not should_ingest:
                print(f"  SKIP      {source_file}")
                results["skipped"].append(source_file)
                continue

            # One document expands into a small subgraph which consists of the document node plus its project, 
            # document type and discipline, then all of its sections and chunks, 
            # then the describes edges that link chunks to the entities mentioned earlier in the metadata.
            session.execute_write(write_document_node, doc)
            session.execute_write(write_project_node, doc)
            session.execute_write(write_document_type_node, doc)
            session.execute_write(write_discipline_node, doc)
            session.execute_write(write_sections_and_chunks, doc)
            session.execute_write(write_describes_edges, doc)

            if supersedes:
                session.execute_write(supersede_document, supersedes, source_file)

                print(f"  SUPERSEDE {supersedes}  ->  {source_file}")
                results["superseded"].append((supersedes, source_file))
            else:
                print(f"  INGEST    {source_file}")
            results["ingested"].append(source_file)

    return results
