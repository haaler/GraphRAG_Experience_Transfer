"""Graph-primary chunk retrieval for the GraphRAG pipeline.

Chunks are reached through the entity graph where query terms match Entity nodes, traversal over typed relationships and 
2-hop paths yields related entities, and those entities' HAS_ENTITY edges point back to chunks. 
Each retrieval path carries a decaying score.

A chunk reached by several paths keeps the *best* score across them, so a chunk reachable both by a direct 
entity match and by a 2-hop walk lands at 1.0, not 0.4. 
The optional doc_filter argument, when set, restricts the chunk search universe to 
that set of source files; it is a gate, not a score input.
Returned chunks are sorted descending by score and capped at GRAPH_MAX_CHUNKS.
"""

import re
from typing import List, Optional, Tuple

from graphrag import config as default_config
from graphrag.entities.schemas import ALL_RELATIONSHIP_TYPES


def _extract_query_terms(question: str) -> List[str]:
    """Extract candidate match terms from the question.

    Returns single tokens of length >= 4 plus all two-word bigrams.
    The minimum length stops short common words ("or", "on", "the") from matching
    entity names as substrings via Cypher's CONTAINS. 
    Bigrams catch compound entity names like "Deepsea Atlantic" that would split into two short tokens otherwise.
    """
    # Allowed token characters: letters, digits, &, /, apostrophe, hyphen.
    # Leading and trailing apostrophes/backslashes/hyphens are stripped after
    # the match because regex word boundaries don't catch them cleanly.
    tokens = re.findall(r"[A-Za-z0-9&/'\-]{2,}", question)
    terms = [token.strip("'\\-").lower() for token in tokens if len(token.strip("'\\-")) >= 4]

    words = question.split()
    for i in range(len(words) - 1):
        bigram = (words[i] + " " + words[i + 1]).lower()
        terms.append(bigram)

    # Order-preserving dedup. 
    # dict.fromkeys keeps first occurrence and Python retain insertion order, 
    # so this returns unique terms in the order they first appeared.
    return list(dict.fromkeys(terms))


# ── Typed relationship filter for Cypher ──────────────────────────────────
# Schema-on (default) uses an allowlist: every rel type declared in our schema.
# Schema-off uses a denylist of structural (non-entity-to-entity) rel types so any LLM-chosen
# entity-to-entity label flows through regardless of name.
_STRUCTURAL_REL_TYPES = (
    "HAS_ENTITY", "IS_A", "MEMBER_OF", "SIMILAR_TO",
    "HAS_SECTION", "HAS_CHUNK", "PARENT_SECTION",
    "SUPERSEDED_BY", "HAS_DOCUMENT", "HAS_TYPE",
    "HAS_DISCIPLINE", "DESCRIBES",
)


def _typed_rel_filter_clauses(cfg) -> Tuple[str, str]:
    """Build the Cypher filter snippets for entity-to-entity relationship traversal.

    Returns (one_rel_clause, two_rel_clause).
      one_rel_clause looks like "type(r) IN [...]" or "NOT type(r) IN [...]"
      two_rel_clause looks like "type(r1) IN [...] AND type(r2) IN [...]" or the negated form
    """
    use_schema = getattr(cfg, "USE_CONCEPTUAL_SCHEMA", True)
    if use_schema:
        list_str = "[" + ", ".join(f"'{rel_type}'" for rel_type in ALL_RELATIONSHIP_TYPES) + "]"
        return (
            f"type(r) IN {list_str}",
            f"type(r1) IN {list_str} AND type(r2) IN {list_str}",
        )
    list_str = "[" + ", ".join(f"'{rel_type}'" for rel_type in _STRUCTURAL_REL_TYPES) + "]"
    return (
        f"NOT type(r) IN {list_str}",
        f"NOT type(r1) IN {list_str} AND NOT type(r2) IN {list_str}",
    )


def retrieve_by_graph(
    question: str,
    driver,
    doc_filter: Optional[List[str]] = None,
    cfg=None,
) -> List[Tuple[str, float]]:
    """Graph-primary retrieval using typed relationships.

    Multi-hop traversal with decaying scores:
      - Direct entity match:          GRAPH_DIRECT_SCORE (1.0)
      - 1-hop typed relationship:     GRAPH_TYPED_REL_SCORE (0.8)
      - 2-hop (any combination):      GRAPH_2HOP_SCORE (0.4)
    """
    if cfg is None:
        cfg = default_config

    direct_score = getattr(cfg, "GRAPH_DIRECT_SCORE", 1.0)
    typed_score = getattr(cfg, "GRAPH_TYPED_REL_SCORE", 0.8)
    hop2_score = getattr(cfg, "GRAPH_2HOP_SCORE", 0.4)

    terms = _extract_query_terms(question)
    if not terms:
        return []

    one_rel_clause, two_rel_clause = _typed_rel_filter_clauses(cfg)

    scores: dict = {}

    with driver.session() as session:
        for term in terms:
            # When doc_filter is set, restrict the chunk search universe to those source files. 
            # The same WHERE clause is appended to all three Cypher queries below.
            doc_clause = "WHERE c.source_file IN $doc_filter " if doc_filter else ""
            params: dict = {"term": term}
            if doc_filter:
                params["doc_filter"] = doc_filter

            # Direct entity match → direct_score
            cypher_direct = (
                "MATCH (e:Entity) "
                "WHERE toLower(e.name) CONTAINS $term "
                "MATCH (e)<-[:HAS_ENTITY]-(c:Chunk) "
                + doc_clause +
                "RETURN DISTINCT c.chunk_id AS chunk_id"
            )
            # Keep the best score per chunk.
            # A chunk reachable by both a direct match and a 2-hop walk lands at 1.0, not 0.4.
            for row in session.run(cypher_direct, **params):
                cid = row["chunk_id"]
                scores[cid] = max(scores.get(cid, 0.0), direct_score)

            # 1-hop typed relationships → typed_score
            cypher_typed = (
                "MATCH (e:Entity) "
                "WHERE toLower(e.name) CONTAINS $term "
                "MATCH (e)-[r]-(related:Entity) "
                f"WHERE {one_rel_clause} "
                "MATCH (related)<-[:HAS_ENTITY]-(c:Chunk) "
                + doc_clause +
                "RETURN DISTINCT c.chunk_id AS chunk_id"
            )
            for row in session.run(cypher_typed, **params):
                cid = row["chunk_id"]
                scores[cid] = max(scores.get(cid, 0.0), typed_score)

            # 2-hop via typed relationships only → hop2_score
            cypher_2hop = (
                "MATCH (e:Entity) "
                "WHERE toLower(e.name) CONTAINS $term "
                "MATCH (e)-[r1]-(mid:Entity)-[r2]-(far:Entity) "
                f"WHERE {two_rel_clause} "
                "AND far <> e "
                "MATCH (far)<-[:HAS_ENTITY]-(c:Chunk) "
                + doc_clause +
                "RETURN DISTINCT c.chunk_id AS chunk_id"
            )
            for row in session.run(cypher_2hop, **params):
                cid = row["chunk_id"]
                scores[cid] = max(scores.get(cid, 0.0), hop2_score)

    max_chunks = getattr(cfg, "GRAPH_MAX_CHUNKS", 30)
    min_score = getattr(cfg, "GRAPH_MIN_SCORE", 0.5)

    # Keep only chunks whose best score reached the threshold. 
    # Each surviving entry is a (chunk_id, score) tuple.
    results = []
    for cid, score in scores.items():
        if score >= min_score:
            results.append((cid, score))

    # Sort by score, highest first. 
    # The key function pulls the score (the second slot) out of each (chunk_id, score) 
    # tuple so .sort() knows what value to compare on.
    results.sort(key=lambda chunk_score: chunk_score[1], reverse=True)
    return results[:max_chunks]
