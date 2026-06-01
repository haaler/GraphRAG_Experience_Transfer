"""Entity name canonicalisation and filtering.

This module sits between LLM extraction and the Neo4j writers in the entity stage.
It does three jobs on the raw entity and relationship lists the LLM returns for each chunk:

  1. Canonicalise entity names. Type scoped alias lookups for known rigs
     and known organisations, with title casing as a fallback for other
     proper noun types.
  2. Drop noisy or hallucinated entity extractions: too short, all numeric,
     a short all caps signature, a name that is just its own type or class
     label, a slash joined generic concept, a document revision reference,
     or a self reference to the source file.
  3. Drop relationships whose shape is not in the schema or whose endpoints
     no longer exist after the entity filter.
"""

import re
from typing import List, Dict, Tuple

from graphrag import config as default_config
from graphrag.entities.schemas import CORE_SCHEMA


# ── Canonical lookup tables ───────────────────────────────────────────────
# Applied only when entity_type == "RigInstallation"
RIG_CANONICAL = {
    "alv":   "Deepsea Alvheim",
    "and":   "Deepsea Andrew",
    "asl":   "Askeladden",
    "asp":   "Askepott",
    "beq":   "Buzzard",
    "bra":   "Brage",
    "brae":  "East Brae",
    "bru":   "Serica Bruce",
    "braa":  "Brae Alpha",
    "buc":   "Buchan",
    "byf":   "Byford Dolphin",
    "cla":   "Clair",
    "clm":   "Claymore",
    "clr":   "Clair Ridge",
    "cly":   "Clyde",
    "cora":  "Cormorant Alpha",
    "corn":  "North Cormorant",
    "dab":   "Deepsea Aberdeen",
    "dal":   "Dalian Developer",
    "dbg":   "Deepsea Bergen",
    "dbo":   "Deepsea Bollsta",
    "dmi":   "Deepsea Mira",
    "dp6":   "DP6 CENTRAL",
    "dp8":   "DP8 CENTRAL",
    "dsa":   "Deepsea Atlantic",
    "dsh":   "Deepsea Hercules",
    "dsn":   "Deepsea Nordkapp",
    "dss":   "Deepsea Stavanger",
    "dsw":   "Dan Swift",
    "dsy":   "Deepsea Yantai",
    "egr":   "Edvard Grieg",
    "eid":   "Eider",
    "eko":   "Ekofisk Diverse",
    "ekoc":  "Ekofisk C",
    "ekok":  "Ekofisk K",
    "ekox":  "Ekofisk X",
    "elda":  "Eldfisk A",
    "eldb":  "Eldfisk B",
    "fasp":  "Forties Alpha Satellite Platform",
    "ftr":   "Floatell Triumph",
    "ful":   "Fulmar",
    "gfa":   "Gullfaks A",
    "gfb":   "Gullfaks B",
    "gfc":   "Gullfaks C",
    "gol":   "Goliat",
    "gra":   "Grane",
    "had":   "Harding",
    "hd":    "Heidrun",
    "hdb":   "Heidrun B",
    "hea":   "Heimdal",
    "inn":   "Island Innovator",
    "iva":   "Ivar Aasen",
    "jsv":   "Johan Sverdrup",
    "kdc":   "Kuwait Drilling Company",
    "lin":   "Linus",
    "mag":   "BP Magnus",
    "mar":   "Mariner",
    "mon":   "Montrose",
    "nel":   "Nelson",
    "njoa":  "Njord A",
    "njob":  "Njord B",
    "noc":   "Northern Ocean",
    "pip":   "Piper Bravo",
    "rho":   "Ringhorne",
    "sca8":  "Scarabeo 8",
    "sfa":   "Statfjord A",
    "sfb":   "Statfjord B",
    "sfc":   "Statfjord C",
    "ska":   "Skarv",
    "sla":   "Sleipner A",
    "sna":   "Snorre A",
    "snb":   "Snorre B",
    "tar":   "Tartan",
    "tera":  "Tern Alpha",
    "try":   "Songa Trym",
    "val":   "Valhall",
    "vfr":   "Veslefrikk A/B",
    "vip":   "Valhall IP",
    "vis":   "Visund",
    "wea":   "West Elara",
    "ymi":   "Yme Inspirer",
}

# Applied only when entity_type in {"Organization", "Supplier", "Client"}.
# Entries are kept here only when they actually transform the name (statoil -> Equinor) 
# and acronym expansions (odt -> Odfjell Drilling Technology).
# Plain case fixes are left to the title casing fallback in canonicalize_entity_name.
ORG_CANONICAL = {
    "odt":              "Odfjell Drilling Technology",
    "bp":               "AkerBP",
    "statoil":          "Equinor",
    "oe":               "Odfjell Energy",
    "oe bergen":        "Odfjell Energy Bergen",
    "kdc":              "Kuwait Drilling Company",
    "oeid":             "Odfjell Technology Innovation and Development",
    "oem1":             "Odfjell Technology Marine",
    "oem2":             "Odfjell Technology Subsea",
    "oem3":             "Odfjell Technology Maintenance",
    "oepe":             "Odfjell Technology P&E",
    "oes":              "Odfjell Technology Green Technology",
    "oris":             "Odfjell Rig Inspection Services",
    "total":            "TOTAL E&P South Africa (TEPSA)",
}


# Built once at module load. Maps each entity type to its parent entity_class from CORE_SCHEMA.
# Used by the type recovery branch in filter_entities so a rewritten entity_type stays consistent with its entity_class.
# CORE_SCHEMA groups several entity types under each entity class, so we walk the structure once and invert it.
_TYPE_TO_CLASS = {}
for entityclass, entitytypes in CORE_SCHEMA.items():
    for entitytype in entitytypes:
        _TYPE_TO_CLASS[entitytype] = entityclass


# ── Entity type whitelists ────────────────────────────────────────────────
# Restricts which entity types may be extracted from table / list chunks.
# Tables/Lists rarely contain non-specific information which is valuable to extract as free-form entities. 
# Therefore we restrict the extraction to the entity types from CORE_SCHEMA 
# (whereas extraction from text chunks is only guided by CORE_SCHEMA). 
# Keep in sync with CORE_SCHEMA in schemas.py.
TABLE_ENTITY_WHITELIST = {
    "Supplier", "Client", "Organization", "RigInstallation", "Field", "Location", "Reference",
    "ContactPerson", "Equipment", "StandardReference", "AnalysisMethod",
}

LIST_ENTITY_WHITELIST = {
    "Supplier", "Client", "Organization", "RigInstallation", "Field", "Location", "Reference",
    "ContactPerson", "Equipment", "StandardReference", "AnalysisMethod",
}

# Entity types whose names should run through the title casing fallback in canonicalize_entity_name.
# Some entity types are deliberately absent as their names carry specific casing conventions 
# (e.g. NORSOK D-010) that title casing would destroy.
PROPER_NOUN_TYPES = {
    "RigInstallation", "Organization", "Supplier", "Client", "Location","ContactPerson", 
    "Equipment", "Field", "Role",
}


def canonicalize_entity_name(name: str, entity_type: str) -> str:
    """Normalise an entity name.

    First a type scoped alias lookup (rig dict for RigInstallation, org dict for Organization/Supplier/Client).
    If no alias hits, fall back to title casing for proper noun types. Names of any other type are returned as is.
    """
    lower = name.lower()

    # 1. Type-scoped lookup
    if entity_type == "RigInstallation" and lower in RIG_CANONICAL:
        return RIG_CANONICAL[lower]
    if entity_type in ("Organization", "Supplier", "Client") and lower in ORG_CANONICAL:
        return ORG_CANONICAL[lower]

    # 2. Title-case proper nouns. Short uppercase tokens (acronyms like API, ISO, BP) are kept 
    # as is instead of being .capitalize()'d, which would turn API into Api and lose recognisability.
    if entity_type in PROPER_NOUN_TYPES:
        return " ".join(w if (w.isupper() and len(w) <= 5) else w.capitalize() for w in name.split())
    return name

def filter_entities(entities: List[Dict], chunk_type: str = "text", source_file: str = "") -> List[Dict]:
    """Remove noisy and hallucinated extractions and apply chunk-type whitelisting.

    Table and list chunks have a tighter type whitelist than text chunks, since those structured layouts
    rarely produce free form mentions like ContactPerson roles or AnalysisMethod descriptions.
    Both the type-recovery branch and the chunk-type whitelist are skipped when
    cfg.USE_CONCEPTUAL_SCHEMA is False, since the LLM picks free-form types in that mode.
    """
    use_schema = getattr(default_config, "USE_CONCEPTUAL_SCHEMA", True)
    clean = []
    for e in entities:
        name = (e.get("name") or "").strip()

        # ── Basic noise ────────────────────────────────────────────────────
        if len(name) < 2:
            continue
        # Drops names made up entirely of digits, whitespace, dots, and commas
        # (e.g. "12.34", "1, 2, 3", "  42  ") as they are noise and not entities.
        if re.fullmatch(r'[\d\s.,]+', name):
            continue

        # ── Short all-caps: likely initials or signatures ──────────────────
        # Drops 2 or 3 character uppercase names unless we recognise them.
        # The LLM often picks up document signatures, header initials,
        # and stray abbreviations that look like entities but are not,
        # so the default for short uppercase tokens is to drop them unless they are recognised.
        # A name is recognised when it appears as a key in the alias dict for its own entity type
        # (RIG_CANONICAL for rigs, ORG_CANONICAL for orgs/suppliers/clients).
        #
        # Type recovery: if the name is missing from its own dict but appears in the *other* one,
        # the LLM labeled it wrong. We rewrite entity_type and entity_class to match the dict it
        # was actually found in, so canonicalize_entity_name can rename it correctly downstream.
        # Example: ("BP", "RigInstallation") -> ("BP", "Organization") -> "AkerBP".
        if len(name) <= 3 and name.isupper():
            short_lower = name.lower()
            etype = e.get("entity_type", "")
            in_rig = short_lower in RIG_CANONICAL
            in_org = short_lower in ORG_CANONICAL

            # Type recovery rewrites the entity_type / entity_class to match the alias dict
            # the name was found in. That only makes sense when there's a fixed schema to recover
            # against. With the schema off, we still drop unrecognised short uppercase tokens but
            # leave any recognised one's type alone.
            if not use_schema:
                if not (in_rig or in_org):
                    continue
            else:
                if etype == "RigInstallation" and in_rig:
                    pass
                elif etype in ("Organization", "Supplier", "Client") and in_org:
                    pass
                elif in_rig:
                    print(f"    [TYPE-RECOVER] {name}: {etype} -> RigInstallation")
                    e["entity_type"]  = "RigInstallation"
                    e["entity_class"] = _TYPE_TO_CLASS["RigInstallation"]
                elif in_org:
                    print(f"    [TYPE-RECOVER] {name}: {etype} -> Organization")
                    e["entity_type"]  = "Organization"
                    e["entity_class"] = _TYPE_TO_CLASS["Organization"]
                else:
                    continue

        # ── Hallucination: name ≈ type or class ────────────────────────────
        # Drop entities whose name is essentially the type or class label itself 
        # (e.g. an entity named "Organization" with entity_type "Organization").
        # Both directions of substring match are checked.
        name_lower  = name.lower()
        type_lower  = (e.get("entity_type")  or "").lower().replace("_", " ")
        class_lower = (e.get("entity_class") or "").lower().replace("_", " ")
        if type_lower  and (type_lower  in name_lower or name_lower in type_lower):
            continue
        if class_lower and (class_lower in name_lower or name_lower in class_lower):
            continue

        # ── Hallucination: slash-joined generic concepts ────────────────────
        # Drops slash joined names that have no uppercase letters anywhere
        # (e.g. "pump/valve/handle", "drilling/completion"). 
        # A real entity with a slash usually has at least one capital ("OE/Bergen"),
        # so the all-lowercase form is treated as a generic phrase rather than a name.
        if "/" in name and not any(char.isupper() for char in name):
            continue

        # ── Document revision references ───────────────────────────────────
        # Drops random document revision references like "23 - Rev A", "12-Rev01B", "7 — RevC".
        # The pattern is: digits, optional space, a dash (regular, en-dash, or em-dash), optional space,
        # "Rev", optional space, then a revision token (letters/digits). Case-insensitive.
        if re.match(r'\d+\s*[-–—]\s*Rev\s*[\w]+', name, re.IGNORECASE):
            continue

        # ── Self-reference filter ──────────────────────────────────────────
        if source_file:
            source_stem = source_file.rsplit(".", 1)[0]
            if name == source_stem or name == source_file:
                continue

        # ── Table whitelist ────────────────────────────────────────────────
        # Whitelists pin which entity types may come out of structured chunks. Without a fixed
        # type vocabulary the list is arbitrary, so we skip both whitelists in schema-free mode.
        if use_schema:
            if chunk_type == "table" and e.get("entity_type") not in TABLE_ENTITY_WHITELIST:
                continue
            if chunk_type == "list" and e.get("entity_type") not in LIST_ENTITY_WHITELIST:
                continue

        clean.append(e)
    return clean


def filter_relationships(
    relationships: List[Dict],
    valid_entity_names: set,
    rel_schema: List[Tuple[str, str, str, str]],
    entity_type_map: Dict[str, str],
) -> List[Dict]:
    """Filter extracted relationships against the schema and surviving entities.

    A relationship survives only if its (source_type, relationship_type, target_type) shape is
    declared in the schema and both endpoints are in the entity set that came back from filter_entities.

    When rel_schema is empty, the schema-shape check is skipped: any non-self-loop relationship
    with a non-empty type and both endpoints in the entity set survives. This is the schema-free
    branch — callers pass [] when cfg.USE_CONCEPTUAL_SCHEMA is False.
    """
    schema_free = not rel_schema

    # Build an index of the valid (source_type, target_type) pairs per relationship type.
    # Example:
    #   valid_pairs["SUPPLIER_FOR"] = {("Supplier", "RigInstallation"),("Supplier", "Client")}
    valid_pairs: Dict[str, set] = {}

    # Only need the first three slots from the relationship tuple (source_type, rel_type, target_type, description)
    for src_type, rel_type, tgt_type, _ in rel_schema:
        valid_pairs.setdefault(rel_type, set()).add((src_type, tgt_type))

    valid_rel_types = set(valid_pairs.keys())
    clean = []

    for r in relationships:
        source  = (r.get("source") or "").strip()
        target  = (r.get("target") or "").strip()
        rel_type = (r.get("type")  or "").strip().upper()

        if source not in valid_entity_names or target not in valid_entity_names:
            continue
        if source.lower() == target.lower():
            continue
        if not rel_type:
            continue

        if not schema_free:
            if rel_type not in valid_rel_types:
                continue
            src_etype = entity_type_map.get(source, "")
            tgt_etype = entity_type_map.get(target, "")
            if (src_etype, tgt_etype) not in valid_pairs[rel_type]:
                continue

        r["type"] = rel_type
        clean.append(r)

    return clean
