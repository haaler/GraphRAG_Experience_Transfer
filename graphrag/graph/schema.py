"""Neo4j constraints, indexes, and vector indexes for the graph schema.

'create_constraints' runs once at the start of the indexing pipeline.
'create_community_constraints' runs once after community detection.
'create_vector_indexes' is invoked manually from the notebook after community summaries have been written.
"""


def create_constraints(tx) -> None:
    """Create the uniqueness constraints and lookup indexes for the graph."""
    # Uniqueness constraints
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.source_file IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (s:Section) REQUIRE s.section_id IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Project) REQUIRE p.project_number IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (et:EntityType) REQUIRE et.name IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (dt:DocumentType) REQUIRE dt.code IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (di:Discipline) REQUIRE di.code IS UNIQUE")

    # Indexes for common query patterns
    tx.run("CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.base_doc_id)")
    tx.run("CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.status)")
    tx.run("CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.rig)")
    tx.run("CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.supplier)")
    tx.run("CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.client)")
    tx.run("CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.discipline)")
    tx.run("CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.source_file)")
    tx.run("CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.chunk_type)")

    print("Constraints and indexes ready")


def create_community_constraints(tx) -> None:
    """Create the uniqueness constraint on Community nodes."""
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (comm:Community) REQUIRE comm.community_id IS UNIQUE")
    print("Community constraints ready")


def create_vector_indexes(driver, embedding_dim: int) -> None:
    """Create the community embedding vector index.

    Only communities carry vector embeddings. Chunks are reached through the entity graph, not by vector similarity.
    """
    name = "community_embedding_index"
    label = "Community"
    prop = "embedding"

    with driver.session() as session:
        try:
            session.run(
                f"CALL db.index.vector.createNodeIndex("
                f"'{name}', '{label}', '{prop}', $dim, 'cosine')",
                dim=embedding_dim,
            )
            print(f"  Vector index '{name}' created")
        except Exception as e:
            # Neo4j raises a generic error when the index already exists, only distinguishable by message text.
            if "already exists" in str(e).lower():
                print(f"  Vector index '{name}' already exists")
            else:
                raise
