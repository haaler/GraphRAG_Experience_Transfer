"""Entity ontology and schema helpers.

This module is the single source of truth for what an entity is in the graph.
It defines the entity classes and their types (CORE_SCHEMA), the human readable
descriptions sent to the LLM (TYPE_DESCRIPTIONS), the typed edges between
entities (RELATIONSHIPS), and the lookup tables that turn the short codes in
filenames into full document type and discipline names.

The rest of the pipeline depends on this file. Extraction reads the schema and
the descriptions to build its prompt. The entity writer uses ALL_RELATIONSHIP_TYPES
to know which edge labels exist. Cypher retrieval and community detection read
the same constants to filter relationships.
"""

from typing import Dict, List, Tuple

# ── Universal entity schema ────────────────────────────────────────────────
# A single schema used for all document types.
# Discipline and DocumentType are not extracted as entities.
# They are instead parsed from the filename and stored as Document metadata and as linked graph nodes.
CORE_SCHEMA = {
    "BasicInformation": [
        "Supplier",          # bidding / providing company
        "Client",            # contracting company
        "Organization",      # any other named organization
    ],
    "CrossReference": [
        "Reference",         # a specific named reference: contract, project number, campaign, or document code
    ],
    "Place": [
        "RigInstallation",   # rig name
        "Field",             # oil field / installation
        "Location",          # named place: yard, port, base, city (NOT a company)
    ],
    "Personnel": [
        "ContactPerson",     # named person
        "Role",              # job role in TEXT chunks only
    ],
    "Scope": [
        "Equipment",         # specific named equipment (model number, product name, tag)
        "StandardReference", # named industry standard with alphanumeric code
        "AnalysisMethod",    # named, established method/theory/formula
    ],
}

# ── Entity type descriptions (used in LLM prompt) ────────────────────────
TYPE_DESCRIPTIONS = {
    "Supplier":          "a company providing goods or services",
    "Client":            "a company contracting or purchasing goods/services",
    "Organization":      "a company or registered business entity — not a person, not a rig, not a contract/project name",
    "Reference":         "a specific, named cross-reference: contract name, project number, campaign, or document code — must be identifiable (a name, number, or code), not a generic structural label like 'Appendix A', 'Figure 3.2', 'Scope of Work', or 'Technical Purchase Specification'",
    "RigInstallation":   "a named rig or drilling unit",
    "Field":             "a named oil/gas field or offshore installation",
    "Location":          "a named place: yard, port, base, city — not a company",
    "ContactPerson":     "a named person",
    "Role":              "a job title or role (text chunks only)",
    "Equipment":         "a specific named equipment or product — must have a model number, part number, tag, or recognized product name",
    "StandardReference": "a named industry standard or specification, cited with its alphanumeric code (e.g. NORSOK M-630, API 4F, ISO 13534)",
    "AnalysisMethod":    "a named, established method, theory, or formula used in engineering analysis — must be a recognized name you could find in a textbook or standard (e.g. JONSWAP spectrum, Morison equation, FMEA, HAZOP). NOT: parameter values, solver settings, coordinate systems, generic terms like 'dynamic analysis', or discipline names",
}

# ── Relationship schema ───────────────────────────────────────────────────
# Typed edges between entities. Each tuple is (source_type, relationship_type, target_type, description).
# The description is the short hint shown to the LLM so it picks the right edge label.
RELATIONSHIPS: List[Tuple[str, str, str, str]] = [
    # Actors
    ("Supplier",          "SUPPLIER_FOR",    "RigInstallation",  "supplies services/equipment to a rig"),
    ("Supplier",          "SUPPLIER_FOR",    "Client",           "supplies services/equipment to a client"),
    ("Client",            "CLIENT_OF",       "Supplier",         "contracts services from supplier"),
    ("Organization",      "LOCATED_AT",      "Location",         "organization based at a place"),
    ("RigInstallation",   "LOCATED_AT",      "Field",            "rig operates at a field"),
    ("RigInstallation",   "LOCATED_AT",      "Location",         "rig located at a place"),
    ("ContactPerson",     "WORKS_FOR",       "Organization",     "person works for organization"),
    ("ContactPerson",     "WORKS_FOR",       "Supplier",         "person works for supplier"),
    ("ContactPerson",     "WORKS_FOR",       "Client",           "person works for client"),
    # References
    ("Reference",         "ASSOCIATED_WITH", "RigInstallation",  "reference linked to rig"),
    ("Reference",         "ASSOCIATED_WITH", "Equipment",         "reference linked to equipment"),
    # Scope
    ("Supplier",          "PROVIDES",        "Equipment",        "supplier provides named equipment"),
    ("Organization",      "PROVIDES",        "Equipment",        "organization provides named equipment"),
    ("Equipment",         "COMPLIES_WITH",   "StandardReference","equipment certified to a standard"),
    ("RigInstallation",   "COMPLIES_WITH",   "StandardReference","rig certified to a standard"),
]

# All typed relationship labels used in the pipeline (for Cypher queries).
# Each tuple in RELATIONSHIPS is (source_type, rel_type, target_type, description).
# We only care about the second slot here, so the underscores skip the rest.
# Result is a sorted list of distinct rel_type labels like ["ASSOCIATED_WITH", "CLIENT_OF", "COMPLIES_WITH", ...].
ALL_RELATIONSHIP_TYPES = sorted({rel_type for _, rel_type, _, _ in RELATIONSHIPS})


# ── Document type lookup ──────────────────────────────────────────────────
# Maps the two-letter filename code to a human-readable document type name.
# Users add or remove codes as needed.
DOCUMENT_TYPE_CODES: Dict[str, str] = {
    "AL": "Action Log",
    "BE": "Bid Evaluation",
    "BP": "Bid/Proposal",
    "CL": "Checklist",
    "CR": "Cost Report",
    "EM": "Email",
    "ER": "Experience Report",
    "FR": "CFR - Company Findings Report",
    "LA": "Lists/Registers",
    "LE": "Letter",
    "ME": "Memo",
    "MM": "Minutes of Meeting",
    "OC": "Organisation Chart",
    "PA": "Purchase Orders",
    "PB": "Blanket Order/Frame Agreement",
    "PD": "Contract",
    "PL": "Plan",
    "PR": "Project Administration",
    "PT": "Presentation",
    "QA": "Quality Plan",
    "RA": "Reports",
    "RF": "RFI - Request for Information",
    "RQ": "RFQ - Request for Quotation",
    "RR": "Risk Register",
    "TE": "Estimates",
    "VR": "Variation Order/Variation Order Request",
}

# Maps the single-letter discipline code to a full discipline name.
# Users fill in the values for their corpus.
DISCIPLINE_CODES: Dict[str, str] = {
    "A": "Administration",
    "B": "Procurement",
    "C": "Civil/Architect",
    "D": "Drilling",
    "E": "Electrical",
    "F": "Project Control/Planning/DCC/LCI",
    "G": "Geology",
    "H": "HVAC",
    "I": "Instrumentation/Metering",
    "J": "Marine Operations",
    "K": "Inspection",
    "L": "Piping",
    "M": "Material Tech",
    "N": "Structural",
    "O": "Operation/Construction",
    "P": "Process",
    "Q": "Quality Management",
    "R": "Mechanical",
    "S": "Health, Safety, Environment (HSE)",
    "T": "Telecommunication",
    "U": "Subsea",
    "V": "Commissioning",
    "W": "Weight Control",
    "X": "Technical Safety",
    "Y": "Maintenance",
    "Z": "Multidiscipline",
}

def detect_document_type(doc: Dict) -> str:
    """Return the human-readable document type for a document."""
    doctype = (doc.get("document_type") or "").upper()
    return DOCUMENT_TYPE_CODES.get(doctype, "Typeless")

def detect_discipline(doc: Dict) -> str:
    """Return the full discipline name for a document (empty string if not set)."""
    code = (doc.get("discipline") or "").upper()
    return DISCIPLINE_CODES.get(code, code)  # fall back to the raw code


# ── Prompt formatting ─────────────────────────────────────────────────────
def format_schema_for_prompt(schema: Dict) -> str:
    """Format schema so the LLM clearly sees the entity_class -> entity_type hierarchy.

    Types are nested under their entity class so the LLM groups its output the
    same way the rest of the pipeline expects (class first, then type).
    """
    lines = []
    for entityclass, types in schema.items():
        lines.append(f'  entity_class "{entityclass}":')
        for t in types:
            desc = TYPE_DESCRIPTIONS.get(t, "")
            lines.append(f'    - {t}: {desc}' if desc else f'    - {t}')
    return "\n".join(lines)

def format_relationships_for_prompt(rel_schema: List[Tuple[str, str, str, str]]) -> str:
    """Format relationship schema for the LLM prompt.

    Duplicate (source_type, relationship_type, target_type) triples are
    collapsed so the LLM sees each edge shape once even if the schema lists
    several variants with different descriptions.
    """
    seen = set()
    lines = []
    for src_type, rel_type, tgt_type, desc in rel_schema:
        key = (src_type, rel_type, tgt_type)
        if key in seen:
            continue
        seen.add(key)
        # Example output line:
        #   (Supplier) -[SUPPLIER_FOR]-> (RigInstallation): supplies services/equipment to a rig
        lines.append(f"  ({src_type}) -[{rel_type}]-> ({tgt_type}): {desc}")
    return "\n".join(lines)


# ── Neo4j indexes ─────────────────────────────────────────────────────────
def create_entity_indexes(tx) -> None:
    """Create entity-specific constraints and indexes.

    Run once at the start of an entity ingestion pass. 
    The unique constraint on 'merge_key' is what makes the deduplication writer idempotent: writing
    the same entity twice merges into the existing node instead of creating a duplicate.
    """
    tx.run("CREATE CONSTRAINT entity_merge_key_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.merge_key IS UNIQUE")
    tx.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)")
    tx.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)")
    tx.run("CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.entity_class)")
    print("Entity indexes ready")
