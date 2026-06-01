"""Document-level identification and ranking for question routing.

Two-channel candidate generation with per-document scoring:
  1. KG channel: entity graph lookup with rarity weighting. 
     Each query term contributes 1/entity_count per document it reaches, so specific terms 
     (matching few entities) carry more weight than generic ones. 
     This is the same idea as inverse document frequency (IDF), applied here to entities instead of documents:
     a term that pinpoints one entity weighs ~1.0, a term that hits 50 entities weighs ~0.02.
  2. Metadata channel: keyword scoring on Document properties (title,
     project_number, rig, supplier, client, document_type, discipline).

Each channel is normalised to [0, 1] independently, then summed.
Documents above DOC_INCLUSION_RATIO of the best combined score survive,
and revision siblings are pulled in afterwards with score inheritance.
"""

import re
from typing import Dict, List, Tuple

from graphrag import config as default_config
from graphrag.retrieval.chunk_retrieval import _extract_query_terms
from graphrag.entities.filters import RIG_CANONICAL


def _word_in_text(word: str, text: str) -> bool:
    """True if 'word' appears in 'text' as a whole word, not as a substring.

    "DSS" matches "DSS rig" but not "process" (which contains "ss").
    Word boundaries on both sides keep it from matching short abbreviations inside unrelated longer words.
    """
    pattern = rf'\b{re.escape(word)}\b'
    return re.search(pattern, text) is not None


# ── Rig term expansion ───────────────────────────────────────────────────────
def _expand_rig_terms(terms: List[str], question: str) -> List[str]:
    """Add rig abbreviation/full name counterparts to query terms.

    Scans the original question (not just extracted terms) for known rig abbreviations and full names from RIG_CANONICAL.
    This handles short abbreviations like "DSS" (3 chars, below the 4-char _extract_query_terms
    threshold) and ensures both forms are available for KG and metadata lookup.
    """
    expanded = list(terms)
    q_lower = question.lower()

    for abbrev, full_name in RIG_CANONICAL.items():
        full_lower = full_name.lower()
        # Abbreviation appears in question -> add full name
        if _word_in_text(abbrev, q_lower):
            if full_lower not in expanded:
                expanded.append(full_lower)
            if abbrev not in expanded:
                expanded.append(abbrev)
        # Full name appears in question -> add abbreviation
        if full_lower in q_lower:
            if abbrev not in expanded:
                expanded.append(abbrev)
            if full_lower not in expanded:
                expanded.append(full_lower)

    return expanded


# ── KG channel ───────────────────────────────────────────────────────────────
def _kg_lookup(question: str, session, cfg) -> Tuple[Dict[str, float], List[str]]:
    """Entity graph lookup with rarity weighting (the IDF idea).

    For each query term, counts how many entities match.
    Terms matching too many entities (> KG_MAX_ENTITY_MATCHES) are skipped as too generic.
    Remaining terms contribute 1/entity_count per document they reach,
    so specific terms (matching few entities) carry more weight than common ones.

    Returns ({source_file: score}, [skipped_terms]).
    """
    terms = _extract_query_terms(question)
    terms = _expand_rig_terms(terms, question)
    max_matches = getattr(cfg, "KG_MAX_ENTITY_MATCHES", 50)
    doc_scores: Dict[str, float] = {}
    skipped: List[str] = []

    for term in terms:
        count = session.run(
            "MATCH (e:Entity) "
            "WHERE toLower(e.name) CONTAINS $term "
            "RETURN count(e) AS c",
            term=term,
        ).single()["c"]

        if count == 0:
            continue
        if count > max_matches:
            skipped.append(term)
            continue

        # Inverse-document-frequency shape: 
        # rare entity matches contribute more per document than common ones, so a term that pinpoints one
        # entity weighs ~1.0 and a term that hits 50 entities weighs ~0.02.
        weight = 1.0 / count

        rows = session.run(
            "MATCH (e:Entity) "
            "WHERE toLower(e.name) CONTAINS $term "
            "MATCH (e)<-[:HAS_ENTITY]-(c:Chunk) "
            "RETURN DISTINCT c.source_file AS sf",
            term=term,
        ).data()

        for r in rows:
            doc_scores[r["sf"]] = doc_scores.get(r["sf"], 0.0) + weight

    return doc_scores, skipped


# ── Metadata channel ─────────────────────────────────────────────────────────
def _metadata_scoring(question: str, session, cfg) -> List[Tuple[float, str]]:
    """Score all Document nodes by metadata keyword matching.

    Returns list of (score, source_file) sorted descending.
    Family expansion handles revisions, so no supersession penalty is applied here.
    Expands question with rig abbreviation counterparts before matching.
    """
    q_lower = question.lower()

    # Expand with rig abbreviation counterparts
    for abbrev, full_name in RIG_CANONICAL.items():
        full_lower = full_name.lower()
        if _word_in_text(abbrev, q_lower):
            q_lower += " " + full_lower
        if full_lower in q_lower:
            q_lower += " " + abbrev

    rows = session.run(
        "MATCH (d:Document) "
        "OPTIONAL MATCH (d)-[:HAS_DISCIPLINE]->(disc:Discipline) "
        "OPTIONAL MATCH (d)-[:HAS_TYPE]->(dt:DocumentType) "
        "RETURN d.source_file     AS source_file, "
        "       d.project_number  AS project_number, "
        "       d.rig             AS rig, "
        "       d.supplier        AS supplier, "
        "       d.client          AS client, "
        "       d.document_type   AS document_type, "
        "       d.status          AS status, "
        "       d.title           AS title, "
        "       disc.name         AS discipline_name, "
        "       dt.name           AS document_type_name"
    ).data()

    scored = []
    for row in rows:
        kw_bonus = 0.0

        # Large confidence bumps for cases when the user names the source file or project.
        # So a question that mentions a source filename or project number routes to that
        # document with very high confidence.
        sf = (row.get("source_file") or "").lower().strip()
        if sf and sf in q_lower:
            kw_bonus += 10.0

        pn = (row.get("project_number") or "").lower().strip()
        if pn and pn in q_lower:
            kw_bonus += 3.0

        # Metadata field token overlap. Each tuple is (field name, per-token weight). 
        # The per-token bonus added is 'weight * 0.5', so a token has to overlap on multiple high-weight fields
        # to add up to a strong signal on its own.
        for field, weight in [("title", 2.0), ("rig", 2.0),
                              ("supplier", 1.5), ("client", 1.5),
                              ("document_type", 1.0),
                              ("document_type_name", 1.5),
                              ("discipline_name", 1.5)]:
            val = (row.get(field) or "").lower().strip()
            if not val:
                continue
            for tok in re.split(r"[\s\-_/]+", val):
                if len(tok) >= 3 and tok in q_lower:
                    kw_bonus += weight * 0.5

        scored.append((kw_bonus, row["source_file"]))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


# ── Family expansion ─────────────────────────────────────────────────────────
def _expand_families(doc_scores: Dict[str, float], session) -> Dict[str, float]:
    """Expand to include all revision siblings with score inheritance.

    Family identity: project_number + supplier + client + discipline + document_type + document_sequence
    (everything except revision, running_number, and attachment).

    New family members inherit the score of the triggering document.
    """
    if not doc_scores:
        return doc_scores

    family_rows = session.run(
        "MATCH (d:Document) WHERE d.source_file IN $files "
        "WITH d.source_file AS trigger_sf, "
        "  COALESCE(d.project_number,'') AS pn, "
        "  COALESCE(d.supplier,'')       AS sup, "
        "  COALESCE(d.client,'')         AS cl, "
        "  COALESCE(d.discipline,'')     AS disc, "
        "  COALESCE(d.document_type,'')  AS dt, "
        "  COALESCE(d.document_sequence,'') AS seq "
        "MATCH (fam:Document) "
        "WHERE COALESCE(fam.project_number,'') = pn "
        "  AND COALESCE(fam.supplier,'')       = sup "
        "  AND COALESCE(fam.client,'')         = cl "
        "  AND COALESCE(fam.discipline,'')     = disc "
        "  AND COALESCE(fam.document_type,'')  = dt "
        "  AND COALESCE(fam.document_sequence,'') = seq "
        "RETURN trigger_sf, fam.source_file AS sibling_sf",
        files=list(doc_scores.keys()),
    ).data()

    expanded = dict(doc_scores)
    for row in family_rows:
        sibling = row["sibling_sf"]
        trigger_score = doc_scores.get(row["trigger_sf"], 0.0)
        # Use max so a sibling that scored high on its own is not downgraded by inheriting a weaker sibling's score.
        expanded[sibling] = max(expanded.get(sibling, 0.0), trigger_score)

    return expanded


# ── Main entry point ─────────────────────────────────────────────────────────

def identify_relevant_documents(question: str, driver, cfg=None) -> Tuple[Dict[str, float], dict]:
    """
    Identify relevant documents via two-channel candidate generation with per-document scoring.

    Channel 1 (KG): IDF-weighted entity graph lookup. Specific terms (matching few entities) contribute more per document.
    Channel 2 (Metadata): Keyword scoring on Document properties, expanded with rig abbreviation counterparts.

    Scores are normalized to [0, 1] per channel, summed, threshold-filtered,
    then expanded to include revision families (with score inheritance).

    Returns ({source_file: score}, result_meta).
    """
    if cfg is None:
        cfg = default_config

    inclusion_ratio = getattr(cfg, "DOC_INCLUSION_RATIO", 0.1)

    with driver.session() as session:
        # ── Channel 1: KG lookup (rarity-weighted) ──────────────────
        kg_scores, skipped_terms = _kg_lookup(question, session, cfg)

        # ── Channel 2: Metadata scoring ──────────────────────────────
        scored = _metadata_scoring(question, session, cfg)

        # ── Normalize each channel to [0, 1] ─────────────────────────
        kg_max = max(kg_scores.values()) if kg_scores else 1.0
        kg_norm = {source_file: score / kg_max for source_file, score in kg_scores.items()}

        meta_max = scored[0][0] if scored and scored[0][0] > 0 else 1.0
        meta_norm = {source_file: score / meta_max for score, source_file in scored if score > 0}

        # ── Combine: sum of normalized scores ────────────────────────
        all_docs = set(kg_norm) | set(meta_norm)
        combined = {
            source_file: kg_norm.get(source_file, 0.0) + meta_norm.get(source_file, 0.0)
            for source_file in all_docs
        }

        # ── Threshold filtering ──────────────────────────────────────
        if combined:
            max_combined = max(combined.values())
            inclusion_cutoff = max_combined * inclusion_ratio
            doc_scores = {
                source_file: score for source_file, score in combined.items()
                if score >= inclusion_cutoff
            }
        else:
            doc_scores = {}

        # ── Family expansion with score inheritance ──────────────────
        pre_expansion = len(doc_scores)
        doc_scores = _expand_families(doc_scores, session)

    # ── Stats ─────────────────────────────────────────────────────────
    result_meta = {
        "kg_count": len(kg_scores),
        "meta_count": len(meta_norm),
        "family_added": len(doc_scores) - pre_expansion,
        "skipped_terms": skipped_terms,
    }

    return doc_scores, result_meta
