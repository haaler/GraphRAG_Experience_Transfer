"""Community detection on the Entity graph.

Runs after entity extraction and before community summarisation.
Leiden (via Neo4j GDS) assigns one community_id per Entity. 
It clusters entities so that most typed relationships stay inside a community and few cross between communities. 
After Leiden, any community smaller than COMMUNITY_MIN_SIZE has its community_id stripped so the summarisation step
is not asked to write an LLM summary for a singleton or near-singleton cluster (entity count < COMMUNITY_MIN_SIZE).

Neo4j GDS (Graph Data Science) is the Neo4j plugin that provides graph algorithms like Leiden plus 
the in-memory graph projection mechanism they run on. 
It must be installed on the current Neo4j instance to work.

The second function in this file, get_community_members, is what summarisation.py uses to load each
community's members and their internal relationships.
"""

from typing import Dict

from graphrag import config as default_config
from graphrag.entities.schemas import ALL_RELATIONSHIP_TYPES


# All relationship types projected into the community graph (undirected).
# Used as the default when the conceptual schema is on.
_PROJECTED_TYPES = ALL_RELATIONSHIP_TYPES + ["SIMILAR_TO"]


def detect_communities(driver) -> Dict:
    """Run Leiden community detection on the entity graph via Neo4j GDS.

    Two projection strategies depending on whether the conceptual schema is on.

    Schema on: typed native projection over the curated _PROJECTED_TYPES list,
    intersected with the types actually in the graph. The vocabulary is small
    and fixed, so per-type bookkeeping fits in memory.

    Schema off: the LLM picks its own relationship labels and can produce thousands
    of distinct types. Native typed projection runs out of heap on per-type
    bookkeeping long before edge count becomes the problem (the dict of per-type
    config keys grows to thousands, each allocating its own adjacency arrays and
    importer state). We instead use the gds.graph.project Cypher subquery form
    (GDS 2.4+) to collapse every entity-to-entity edge into one synthetic type,
    weighted by the count of parallel edges between the two nodes, and flagged
    undirected via undirectedRelationshipTypes (which Leiden requires). Leiden
    does not use edge labels in its modularity objective, so the partition is
    the same as the typed version would have produced.

    After projection, Leiden writes community_id onto each Entity, communities below
    COMMUNITY_MIN_SIZE are stripped, and the in-memory projection is dropped.
    """
    graph_name = "entity-community-graph"

    with driver.session() as session:
        # Drop any stale projection from a prior run. 
        # GDS keeps named projections in memory between sessions, so re-running this function
        # without the drop would fail with "graph already exists".
        try:
            session.run(f"CALL gds.graph.drop('{graph_name}', false)")
        except Exception:
            pass

        use_schema = getattr(default_config, "USE_CONCEPTUAL_SCHEMA", True)

        if use_schema:
            # Intersect the curated list with what's actually in the graph.
            # GDS rejects the projection if any requested type is missing.
            existing_rel_types = {r["relationshipType"] for r in session.run("CALL db.relationshipTypes()")}
            present_types = [rel_type for rel_type in _PROJECTED_TYPES if rel_type in existing_rel_types]

            if not present_types:
                print("  [SKIP] No typed relationships found, community detection skipped")
                return {"community_count": 0, "modularity": 0.0, "sizes": {}}

            print(f"  Projecting {len(present_types)} relationship types: {', '.join(present_types)}")

            rel_projection = {rel_type: {"orientation": "UNDIRECTED"} for rel_type in present_types}
            session.run(
                "CALL gds.graph.project($name, 'Entity', $rels)",
                name=graph_name, rels=rel_projection,
            )

            leiden_query = (
                "CALL gds.leiden.write($name, {"
                "  writeProperty: 'community_id',"
                "  includeIntermediateCommunities: false"
                "}) YIELD communityCount, modularity"
            )
        else:
            # Schema off path. Collapse all entity-to-entity edges into one synthetic
            # type so projection memory does not blow up on thousands of LLM-invented
            # labels. The undirectedRelationshipTypes flag is what makes Leiden accept
            # the projection. Without it Leiden refuses to run.
            edge_count = session.run(
                "MATCH (:Entity)-[r]-(:Entity) RETURN count(r) AS n"
            ).single()["n"]

            if edge_count == 0:
                print("  [SKIP] No entity-to-entity edges found, community detection skipped")
                return {"community_count": 0, "modularity": 0.0, "sizes": {}}

            print(f"  Projecting {edge_count} entity-to-entity edges as a single collapsed type")

            session.run(
                "MATCH (s:Entity)-[r]-(t:Entity) "
                "WHERE id(s) < id(t) "
                "WITH s, t, count(r) AS w "
                "WITH gds.graph.project("
                "  $name, "
                "  s, t, "
                "  { relationshipType: 'REL', relationshipProperties: { weight: w } }, "
                "  { undirectedRelationshipTypes: ['REL'] }"
                ") AS g "
                "RETURN g.graphName AS graphName, g.relationshipCount AS rels",
                name=graph_name,
            )

            # Weight tells Leiden that parallel edges between the same pair should
            # count as a stronger connection rather than a single edge.
            leiden_query = (
                "CALL gds.leiden.write($name, {"
                "  writeProperty: 'community_id',"
                "  includeIntermediateCommunities: false,"
                "  relationshipWeightProperty: 'weight'"
                "}) YIELD communityCount, modularity"
            )

        print("  GDS graph projected")

        result = session.run(leiden_query, name=graph_name).single()

        community_count = result["communityCount"]
        modularity = result["modularity"]
        print(f"  Leiden: {community_count} communities, modularity={modularity:.4f}")

        # Strip sub-minimum communities. Leiden frequently returns many
        # singleton communities (one entity per community) that carry no semantic value. 
        # Loading them into summarization would waste DB work and pollute the logs.
        min_size = getattr(default_config, "COMMUNITY_MIN_SIZE", 2)
        cleared = session.run(
            "MATCH (e:Entity) WHERE e.community_id IS NOT NULL "
            "WITH e.community_id AS cid, collect(e) AS members "
            "WHERE size(members) < $min_size "
            "UNWIND members AS m "
            "REMOVE m.community_id "
            "RETURN count(*) AS cleared",
            min_size=min_size,
        ).single()["cleared"]
        if cleared:
            print(f"  Cleared community_id from {cleared} entities "
                  f"in communities below size {min_size}")

        # Get sizes of the retained communities
        sizes = {}
        for rec in session.run(
            "MATCH (e:Entity) WHERE e.community_id IS NOT NULL "
            "RETURN e.community_id AS cid, count(*) AS size "
            "ORDER BY size DESC"
        ):
            sizes[rec["cid"]] = rec["size"]

        # Drop the projection
        session.run(f"CALL gds.graph.drop('{graph_name}', false)")

    return {"community_count": len(sizes), "modularity": modularity, "sizes": sizes}


def get_community_members(driver) -> Dict[int, list]:
    """Retrieve all community members and their internal relationships.

    Returns {community_id: [{"name": ..., "entity_type": ..., "relationships": [...]}]}
    """
    with driver.session() as session:
        # Two passes (members first, then internal relationships) instead of one big join. 
        # A single query joining members to all their internal relationships would multiply 
        # rows and inflate memory for large communities.

        # Pass 1: members per community
        members_by_community: Dict[int, dict] = {}
        for rec in session.run(
            "MATCH (e:Entity) WHERE e.community_id IS NOT NULL "
            "RETURN e.community_id AS cid, e.name AS name, "
            "       e.entity_type AS entity_type "
            "ORDER BY e.community_id"
        ):
            cid = rec["cid"]
            if cid not in members_by_community:
                members_by_community[cid] = {}
            members_by_community[cid][rec["name"]] = {
                "name": rec["name"],
                "entity_type": rec["entity_type"] or "Unknown",
                "relationships": [],
            }

        # Pass 2: internal relationships per community.
        # SIMILAR_TO and IS_A are excluded here because they are graph-structural edges 
        # (semantic similarity and type hierarchy), not the entity-to-entity domain relationships that the
        # community summary should describe.
        for rec in session.run(
            "MATCH (e:Entity)-[r]->(other:Entity) "
            "WHERE e.community_id IS NOT NULL "
            "  AND e.community_id = other.community_id "
            "  AND NOT type(r) IN ['SIMILAR_TO', 'IS_A'] "
            "RETURN e.community_id AS cid, e.name AS src, "
            "       type(r) AS rel_type, other.name AS tgt"
        ):
            cid = rec["cid"]
            src = rec["src"]
            if cid in members_by_community and src in members_by_community[cid]:
                members_by_community[cid][src]["relationships"].append({
                    "type": rec["rel_type"],
                    "target": rec["tgt"],
                })

    # Convert to list format
    return {
        cid: list(members.values())
        for cid, members in members_by_community.items()
    }
