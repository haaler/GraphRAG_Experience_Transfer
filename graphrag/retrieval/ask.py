"""GraphRAG question-answering entry point.

The single public function 'ask' orchestrates the query pipeline end to end:

  Stage 1   Document routing: narrow the corpus to candidate documents.
  Stage 2   Graph chunk retrieval: score chunks reached through the entity graph.
  Stage 2a  Document-level inclusion for high-confidence documents.
  Stage 2b  Entity and community context (graph traversal).
  Stage 2c  Text-to-Cypher structural retrieval.
  Stage 3   Answer generation: LLM call with the assembled context.

'ask' returns (answer, metadata). 
The metadata dict carries everything the caller might need for follow-up turns (doc_filter, confidence, chunk ids,
Cypher diagnostics) and is what chat.py uses to inherit a document filter into the next turn.
"""

from typing import List, Dict, Tuple, Optional

from graphrag import config as default_config
from graphrag.retrieval.document_ranking import identify_relevant_documents
from graphrag.retrieval.chunk_retrieval import retrieve_by_graph, _extract_query_terms
from graphrag.retrieval.context import (
    fetch_chunks, gather_entity_context, gather_community_context,
    assemble_context,
)
from graphrag.retrieval.cypher_retrieval import run_cypher_retrieval


ANSWER_PROMPT = """\
You are an expert assistant for offshore drilling engineering documents.
Use the provided context as your knowledge base to answer the question.

You may:
- Synthesize and summarize information across multiple chunks and documents
- Identify patterns, commonalities, and structures in the documents
- Reason about the context to answer analytical or comparative questions
- Generate templates, summaries, or overviews based on what the documents contain
- Reference documents by their filename (shown in [Document: ...] headers) when
  presenting information, so the user can trace each fact back to its source

Some sources may be from older (superseded) document revisions, marked with
"[SUPERSEDED]" in the context. When newer and older revisions conflict,
prioritize the newer revision. However, older revisions often contain detailed
content that was not repeated in the update — include this information unless
it is directly contradicted by a newer revision.

You must NOT:
- Invent specific facts (names, numbers, dates, costs) that are not in the context
- Claim a document says something it does not
- Conflate entities of different types. Each entity has a type in parentheses
  — e.g. (RigInstallation), (Reference), (ContactPerson), (Supplier). A rig is
  NOT a project. A Reference (contract/project code) is NOT a Document. A Role
  is NOT a Person. Treat each type as a distinct concept.
- Present tangentially related content as if it answers the question. A chunk
  mentioning "budget" is NOT an answer to "what was the 2020 budget" if the
  chunk is from a different year or project. Matching keywords are not matching answers.
- Override typed entity relationships from the knowledge graph with your own
  interpretation of raw text. When the knowledge graph context identifies an
  entity as (Supplier) with a SUPPLIER_FOR relationship, that is authoritative
  — do not re-derive the supplier from "To:", "From:", or other document fields.
  The same applies to all typed roles: Client, ContactPerson, RigInstallation,
  etc. Trust the graph types over text inference.
- IMPORTANT: If you are given a previous conversation history, do not re-list, re-state, 
  or summarize information from those questions or answers. Answer ONLY the new question. 
  For example, if the prior answer listed document numbers and the new question 
  asks about creation dates, provide only the dates — do not list the document 
  numbers again.

Answer the question directly. Do not pad the answer with adjacent information
about unrelated entities unless it is explicitly asked for. If the question asks
for projects, list only projects — not the rigs, suppliers, or documents that
happen to appear alongside them in the context.

When the question references a specific document section (e.g. "deliverables",
"pricing", "schedule", "scope of work"), pay close attention to the [section_title]
tags on each chunk. Prefer content from chunks whose section title matches the
concept in the question. Content under a different section title (e.g. "Scope of
Work" when asked about "Deliverables") may describe related but distinct information
— do not treat it as equivalent.

Unless explicitly told not to, always explain your reasoning and how you used the context to arrive at your answer.

When answering, cite the source document for each piece of information using the
filename from the [Document: ...] headers. For example: "According to
96090-K-RA-0008-01-1_.DOCX, ..."

The context has up to four parts (some may be absent), ordered from raw text
to authoritative structured data:
1. Chunk extracts — actual text from relevant document sections.
2. Community context — summaries of entity clusters that share domain connections.
3. Structural query results — a Cypher query generated from your question and its
   tabular results. When present, these are AUTHORITATIVE for the question they answer
   (lists, counts, aggregations over the graph). Use them directly.
4. Knowledge graph context — entities, their typed relationships, and document
   associations. This is the MOST RELIABLE source for identifying entity roles
   (Supplier, Client, ContactPerson, RigInstallation, etc.) and their connections.
   When a relationship like SUPPLIER_FOR or CLIENT_FOR is present, it is
   authoritative — use it directly rather than inferring roles from chunk text.

{history_block}

CRITICAL — before answering, check whether the context actually addresses the
SPECIFIC question. DO NOT ASSUME. If the question asks about a particular year, project, document
type, or entity and the context only contains tangentially related content (e.g.
the word "budget" appears but from a different year/project), do NOT present that
content as an answer. Instead:
  - Say what you DID find and why it does not match the question
  - If the context is ambiguous and no history is inherited (e.g. "the project" but multiple projects exist"),
    say you found nothing because the question is ambiguous - ask the user to clarify what they mean
  - If no context matches, say so directly — do not stretch unrelated content to fit the question

Question:
{question}

{context}

Answer:
"""

ANSWER_SYSTEM_MSG = (
    "You are a knowledgeable assistant for offshore drilling engineering documents. "
    "Use the provided context as your knowledge base — analyze, synthesize, and reason about it. "
    "Leverage relationship paths between entities (e.g., supplier→rig, finding→equipment) "
    "to connect information across documents. "
    "Do not invent specific facts that are not in the context, but you may draw conclusions, "
    "identify patterns, and create structured outputs (templates, summaries, comparisons) "
    "based on what the documents contain."
)


def _format_history(history: List[Dict]) -> str:
    """Format conversation history for inclusion in the prompt."""
    if not history:
        return ""
    lines = []
    for turn in history:
        lines.append(f"User: {turn['question']}")
        answer = turn["answer"]
        # 200-char cap on prior answers: history is a hint, not the full
        # record, and a long answer would balloon the prompt for follow-ups.
        if len(answer) > 200:
            answer = answer[:200] + "..."
        lines.append(f"Assistant: {answer}")
    return "\n".join(lines)


def ask(
    question: str,
    driver,
    embed_fn,
    llm,
    cfg=None,
    use_entity_context: bool = True,
    history: Optional[List[Dict]] = None,
    verbose: bool = True,
    doc_filter: Optional[List[str]] = None,
    retrieval_query: Optional[str] = None,
) -> Tuple[str, Dict]:
    """
    Full GraphRAG query pipeline.

    Stage 1 — Document identification (metadata filter)
    Stage 2 — Graph chunk retrieval (typed relationships + 2-hop)
    Stage 2b — Entity + community context (graph traversal, optional)
    Stage 2c — Text-to-Cypher structural retrieval (toggle via ENABLE_CYPHER_RETRIEVAL)
    Stage 3 — Answer generation (LLM with grounded context)

    All retrieval uses score thresholds from config (no hardcoded top_k).

    Args:
        question: The user's question
        driver: Neo4j driver
        embed_fn: Embedding function (e.g., emb_model.embed_query) — used for semantic community retrieval
        llm: LLM for answer generation
        cfg: Config module (defaults to graphrag.config)
        use_entity_context: If False, skips graph entity + community context
        history: Optional conversation history for multi-turn chat
        verbose: Print stage-by-stage progress
        doc_filter: Pre-determined document filter (e.g. inherited from a prior conversation turn).
            When provided, Stage 1 is skipped.
        retrieval_query: Optional alternate query used only for the retrieval stages
            (document routing, graph retrieval, entity/community context, Cypher
            generation, section-title boost). When None, retrieval uses 'question'.
            The prompt's <question> block always shows 'question'. chat_turn uses
            this to feed retrieval a keyword-rich merged query without showing
            the merge to the LLM as the thing to answer.

    Returns (answer, metadata).
    """
    if cfg is None:
        cfg = default_config

    # retrieval_q drives Stage 1/2/2b/2c and the section-title term boost.
    # The prompt's <question> block uses the original 'question' so the LLM
    # is never shown a merged query as if it were one big question.
    retrieval_q = retrieval_query if retrieval_query is not None else question

    # ── Stage 1: Document identification ───────────────────────────────
    high_conf_docs: list = []

    if doc_filter is not None:
        # Inherited from a prior conversation turn, so we treat every doc as trusted.
        # Applying a 1.0 score means an inherited filter (which have no real ranking),
        # doesn't break downstream code that reads doc_scores.
        doc_scores = {source_file: 1.0 for source_file in doc_filter}
        high_conf_docs = list(doc_filter)
        # "inherited" / "high" / "low" are the contract with run_cypher_retrieval,
        # which uses the label to phrase its doc-filter hint to the LLM.
        confidence = "inherited"
        if verbose:
            print(f"[Stage 1] Inherited {len(doc_filter)} document(s) from prior turn:")
            for source_file in doc_filter[:10]:
                print(f"  -> {source_file}")
            if len(doc_filter) > 10:
                print(f"  ... and {len(doc_filter) - 10} more")
    else:
        doc_scores, ranking_meta = identify_relevant_documents(retrieval_q, driver, cfg=cfg)

        high_conf_ratio = getattr(cfg, "DOC_HIGH_CONF_RATIO", 0.5)
        max_score = max(doc_scores.values()) if doc_scores else 0.0
        high_conf_cutoff = max_score * high_conf_ratio

        if doc_scores:
            doc_filter = list(doc_scores.keys())
            high_conf_docs = [
                source_file for source_file, score in doc_scores.items()
                if score >= high_conf_cutoff
            ]
        else:
            doc_filter = None
            high_conf_docs = []

        confidence = "high" if high_conf_docs else "low"

        if verbose and doc_filter:
            kg_n = ranking_meta["kg_count"]
            meta_n = ranking_meta["meta_count"]
            fam_n = ranking_meta["family_added"]
            skipped = ranking_meta.get("skipped_terms", [])
            print(f"[Stage 1] Document identification:")
            print(f"          KG: {kg_n} doc(s), Metadata: {meta_n} doc(s), "
                  f"Family: +{fam_n} sibling(s), Total: {len(doc_filter)}")
            if skipped:
                print(f"          Skipped generic terms: "
                      f"{', '.join(skipped[:5])}"
                      + (f" (+{len(skipped)-5} more)" if len(skipped) > 5 else ""))
            # Show high-confidence docs with scores
            sorted_high = sorted(high_conf_docs, key=lambda s: doc_scores.get(s, 0), reverse=True)
            print(f"          High confidence ({len(high_conf_docs)}):")
            for source_file in sorted_high[:10]:
                print(f"            {source_file} ({doc_scores[source_file]:.2f})")
            if len(sorted_high) > 10:
                print(f"            ... and {len(sorted_high) - 10} more")
            n_low = len(doc_filter) - len(high_conf_docs)
            if n_low > 0:
                print(f"          Included ({n_low} more at lower confidence)")
        elif verbose:
            print("[Stage 1] No documents matched")

    # ── Stage 2: Graph chunk retrieval (scored, threshold-filtered) ────
    scored_chunks = retrieve_by_graph(retrieval_q, driver, doc_filter=doc_filter, cfg=cfg)

    chunk_ids = [cid for cid, _ in scored_chunks]
    chunks = fetch_chunks(chunk_ids, driver)

    # Attach graph retrieval scores to chunk dicts
    score_map = {cid: s for cid, s in scored_chunks}
    for c in chunks:
        c["_score"] = score_map.get(c["chunk_id"], 0.0)

    n_docs_retrieved = len(set(c["source_file"] for c in chunks))
    if verbose:
        print(f"[Stage 2] Retrieved {len(chunks)} chunks from {n_docs_retrieved} document(s)")
        if scored_chunks:
            top_score = scored_chunks[0][1]
            bot_score = scored_chunks[-1][1]
            print(f"          Scores: {top_score:.2f} (best) → {bot_score:.2f} (worst)")

    # ── Stage 2a: Document-level chunk inclusion ─────────────────────
    # These chunks are not the result of graph retrieval.
    # They are all pulled in from high-confidence documents so no relevant chunk in those docs is missed, 
    # at a deliberately low score so the higher-confidence chunks still rank above them.
    if high_conf_docs:
        stage2a_baseline_score = getattr(cfg, "STAGE2A_BASELINE_SCORE", 0.2)
        with driver.session() as session:
            doc_chunk_rows = session.run(
                "MATCH (c:Chunk) "
                "WHERE c.source_file IN $files "
                "RETURN c.chunk_id AS chunk_id",
                files=high_conf_docs
            ).data()
        all_doc_chunk_ids = [r["chunk_id"] for r in doc_chunk_rows]

        existing_ids = set(chunk_ids)
        new_ids = [cid for cid in all_doc_chunk_ids if cid not in existing_ids]

        if new_ids:
            baseline_chunks = fetch_chunks(new_ids, driver)
            for c in baseline_chunks:
                c["_score"] = stage2a_baseline_score
            chunks.extend(baseline_chunks)
            chunk_ids.extend(new_ids)

        if verbose:
            print(f"[Stage 2a] Document-level: {len(new_ids)} additional chunk(s) from {len(high_conf_docs)} high-confidence document(s)")

    # ── Stage 2b: Entity + community context (graph traversal, optional) ─
    entity_ctx = ""
    community_ctx = ""
    if use_entity_context:
        entity_ctx = gather_entity_context(retrieval_q, driver, doc_filter=doc_filter)
        community_ctx = gather_community_context(retrieval_q, driver, embed_fn)
        if verbose and entity_ctx:
            n_lines = len(entity_ctx.strip().split("\n")) - 1
            print(f"[Stage 2b] Knowledge graph context: {n_lines} entity entries")
        if verbose and community_ctx:
            n_comm = len(community_ctx.strip().split("\n")) - 1
            print(f"[Stage 2b] Community context: {n_comm} community summaries")
    elif verbose:
        print("[Stage 2b] Entity context disabled")

    # ── Stage 2c: Text-to-Cypher structural retrieval ──────────────────
    cypher_ctx = ""
    cypher_meta: Dict = {"query": None, "n_rows": 0, "error": None, "reason": None}
    if getattr(cfg, "ENABLE_CYPHER_RETRIEVAL", True):
        cypher_ctx, cypher_meta = run_cypher_retrieval(
            question, driver, llm, cfg=cfg, doc_filter=doc_filter, confidence=confidence,
        )
        if verbose:
            query = cypher_meta.get("query")
            n_rows = cypher_meta.get("n_rows", 0)
            error = cypher_meta.get("error")
            reason = cypher_meta.get("reason")

            if error:
                print(f"[Stage 2c] Cypher error: {error}")
                if query:
                    print(f"          Query: {query}")
            elif query:
                print(f"[Stage 2c] Cypher: {n_rows} row(s)")
                print(f"          Query: {query}")
            elif reason:
                print(f"[Stage 2c] Cypher skipped: {reason}")
            else:
                print(f"[Stage 2c] Cypher: 0 rows (no query generated)")
    elif verbose:
        print("[Stage 2c] Cypher retrieval disabled")

    # ── Merge Cypher-retrieved chunks ─────────────────────────────────
    cypher_chunk_ids = cypher_meta.get("chunk_ids", [])
    if cypher_chunk_ids:
        max_cypher_chunks = getattr(cfg, "GRAPH_MAX_CHUNKS", 30)
        cypher_chunk_score = getattr(cfg, "CYPHER_CHUNK_SCORE", 0.6)
        existing_ids = set(chunk_ids)
        new_ids = [cid for cid in cypher_chunk_ids if cid not in existing_ids]
        new_ids = new_ids[:max_cypher_chunks]
        if new_ids:
            cypher_chunks = fetch_chunks(new_ids, driver)
            for c in cypher_chunks:
                c["_score"] = cypher_chunk_score
            chunks.extend(cypher_chunks)
            chunk_ids.extend(new_ids)
            if verbose:
                n_cypher_docs = len(set(c["source_file"] for c in cypher_chunks))
                print(f"[Stage 2c] Merged {len(new_ids)} chunk(s) from {n_cypher_docs} document(s) into retrieval")
        n_docs_retrieved = len(set(c["source_file"] for c in chunks))

    # ── Section-title keyword boost ──────────────────────────────────────
    #    Boost chunks whose section_title matches query terms so they survive
    #    the CONTEXT_MAX_CHARS selection in assemble_context.
    #    Applied after all retrieval stages, before context assembly.
    section_boost = getattr(cfg, "SECTION_TITLE_BOOST", 0.6)
    q_terms = _extract_query_terms(question)
    boosted_count = 0
    for c in chunks:
        section = (c.get("section_title") or "").lower()
        if any(term in section for term in q_terms):
            if c["_score"] < section_boost:
                c["_score"] = section_boost
                boosted_count += 1
    if verbose and boosted_count:
        print(f"[Boost] Section-title match: {boosted_count} chunk(s) boosted to {section_boost}")

    # ── Stage 3: Answer generation ──────────────────────────────────────
    chunk_ctx, used_docs = assemble_context(chunks, cfg=cfg)

    context_parts = []
    context_parts.append(f"Chunk extracts ({len(used_docs)} document(s), {len(chunks)} chunks):\n{chunk_ctx}")
    if community_ctx:
        context_parts.append(community_ctx)
    if cypher_ctx:
        context_parts.append(cypher_ctx)
    if entity_ctx:
        context_parts.append(entity_ctx)
    full_context = "\n\n---\n\n".join(context_parts)

    history_text = _format_history(history or [])
    history_block = ""
    if history_text:
        history_block = (
            "\nConversation history (for follow-up context. DO NOT ANSWER THIS QUESTION):\n"
            + history_text + "\n"
        )

    prompt = ANSWER_PROMPT.format(
        question=question,
        context=full_context,
        history_block=history_block,
    )

    answer = llm.invoke(prompt, system_msg=ANSWER_SYSTEM_MSG)

    metadata = {
        "identified_documents": used_docs,
        "doc_filter":           doc_filter or [],
        "confidence":           confidence,
        "n_chunks":             len(chunks),
        "chunk_ids":            chunk_ids,
        "chunk_scores":         [score for _, score in scored_chunks],
        "entity_context":       entity_ctx,
        "cypher_query":         cypher_meta.get("query"),
        "cypher_rows":          cypher_meta.get("n_rows", 0),
        "cypher_error":         cypher_meta.get("error"),
        "cypher_reason":        cypher_meta.get("reason"),
    }
    return answer, metadata
