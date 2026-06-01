"""LLM based community summarisation and embedding.

Runs after detection.py has assigned a community_id to every Entity, and before retrieval. 
For each community that survived the size cutoff, this module asks the LLM for
a 2 to 3 sentence summary, embeds the summary, and writes to Neo4j:

  - One Community node per community, carrying the summary text, the
    member count, and a vector embedding of the summary.
  - One MEMBER_OF edge per member Entity pointing at its Community node,
    so retrieval can walk from a community back to its entities.

The embedding is stored on the Community node so retrieval can find semantically related communities by 
cosine similarity to the user's question, even when the question shares no entity names with the cluster.
"""

import time

from graphrag import config as default_config
from graphrag.communities.detection import get_community_members

COMMUNITY_SUMMARY_SYSTEM_MSG = (
    "You are a knowledge graph analyst for the offshore drilling industry. "
    "Summarize clusters of related entities concisely."
)

COMMUNITY_SUMMARY_PROMPT = """\
Below is a cluster of entities and their relationships extracted from offshore drilling documents.

Members:
{members_text}

Internal relationships:
{relationships_text}

Summarize this cluster in 2-3 sentences. Focus on:
- What connects these entities (shared project, rig, supplier relationship, etc.)
- The key domain context (what kind of work, which rig/field, etc.)

Be specific and factual. Do not speculate beyond what the entities and relationships show.
"""

def _format_community_for_prompt(members):
    """Format community members and relationships for the LLM prompt."""
    members_lines = []
    rel_lines = []

    for m in members:
        members_lines.append(f"  - {m['name']} ({m['entity_type']})")
        for r in m.get("relationships", []):
            rel_lines.append(
                f"  ({m['name']}) -[{r['type']}]-> ({r['target']})")

    members_text = "\n".join(members_lines) if members_lines else "  (none)"
    relationships_text = "\n".join(rel_lines) if rel_lines else "  (none)"

    return members_text, relationships_text


def summarize_communities(driver, llm, emb_model) -> dict:
    """Generate LLM summaries and embeddings for all communities.

    For each community with >= COMMUNITY_MIN_SIZE members:
    1. Format members + relationships as text
    2. Ask the LLM for a 2-3 sentence summary
    3. Embed the summary
    4. Write Community node with summary + embedding to Neo4j
    5. Link member entities via MEMBER_OF edges

    Returns stats dict.
    """
    min_size = getattr(default_config, "COMMUNITY_MIN_SIZE", 2)
    communities = get_community_members(driver)
    print(f"Summarizing {len(communities)} communities (>= {min_size} members)...")

    stats = {"summarized": 0, "skipped": 0}
    t0 = time.time()

    # Singleton communities are stripped upstream in detection.py, so every
    # community reaching this point already has >= COMMUNITY_MIN_SIZE members.
    for cid, members in communities.items():
        members_text, relationships_text = _format_community_for_prompt(members)

        prompt = COMMUNITY_SUMMARY_PROMPT.format(
            members_text=members_text,
            relationships_text=relationships_text,
        )

        try:
            summary = llm.invoke(prompt, system_msg=COMMUNITY_SUMMARY_SYSTEM_MSG)
            summary = summary.strip()
            # An empty LLM body (rare, e.g. a refusal or a blank response) is
            # treated the same as an exception: skip and keep the loop moving.
            if not summary:
                stats["skipped"] += 1
                continue

            # Embed the summary. 
            # The vector is what retrieval uses for the semantic community lookup at query time.
            embedding = emb_model.embed_query(summary)

            # Write Community node
            with driver.session() as session:
                session.run(
                    "MERGE (comm:Community {community_id: $cid}) "
                    "SET comm.summary = $summary, "
                    "    comm.member_count = $count, "
                    "    comm.embedding = $embedding",
                    cid=cid,
                    summary=summary,
                    count=len(members),
                    embedding=embedding,
                )

                # Link each member entity to the Community node so retrieval
                # can walk from a relevant community back to its entities.
                for m in members:
                    session.run(
                        "MATCH (e:Entity {name: $name}) "
                        "MATCH (comm:Community {community_id: $cid}) "
                        "MERGE (e)-[:MEMBER_OF]->(comm)",
                        name=m["name"], cid=cid,
                    )

            stats["summarized"] += 1
        except Exception as e:
            # Swallow per community failures so one bad summary does not stall the rest.
            # The community is logged and skipped.
            print(f"  [WARN] community {cid}: {e}")
            stats["skipped"] += 1

    elapsed = time.time() - t0
    print(f"  Summarized {stats['summarized']} communities, "
          f"skipped {stats['skipped']} ({elapsed:.1f}s)")

    return stats
