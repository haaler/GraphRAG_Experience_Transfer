"""
Configuration for the GraphRAG pipeline.

API endpoint and key are read from the environment file (.env) for security
"""

from os import getenv
from dotenv import load_dotenv

load_dotenv()


# ── Feature flags ─────────────────────────────────────────────────────────
# Which document formats the indexing pipeline should attempt to load.
DOCUMENT_FORMATS = {"docx"}    # Allowed value: {"docx"}. In the future: {"pdf"}, {"docx", "pdf"}


# ── Neo4j connection ───────────────────────────
NEO4J_URI = getenv("NEO4J_URI")
NEO4J_USER = getenv("NEO4J_USER")
NEO4J_PASSWORD = getenv("NEO4J_PASSWORD")


# ── LLM backend ───────────────────────────────────────────────────────────
LLM_BACKEND = "azure"            # "ollama" or "azure"

EMBEDDING_MODEL = "qwen3-embedding:8b"
LLM_MODEL = "qwen3:8b"  # Used when LLM_BACKEND = "ollama"
LLM_TEMPERATURE = 0.1
LLM_CONTEXT_SIZE = 8192
LLM_TIMEOUT = 180                # Seconds before a stuck LLM call is retried
LLM_MAX_RETRIES = 3              # Attempts before giving up on a transient failure
LLM_RETRY_DELAY = 5.0            # Base seconds between retries (multiplied by attempt #)


# ── Azure OpenAI (used when LLM_BACKEND = "azure") ────────────────────────
AZURE_ENDPOINT = getenv("AZURE_ENDPOINT", "")
AZURE_API_KEY = getenv("AZURE_API_KEY", "")
AZURE_DEPLOYMENT = getenv("AZURE_DEPLOYMENT", "")


# ── Data folders ──────────────────────────────────────────────────────────
DATA_FOLDER = "data_dump/"                 # Source DOCX files
CHUNK_DUMP_FOLDER = "chunk_dumps_new/"     # Per-document chunk JSON snapshots


# ── Document processing ───────────────────────────────────────────────────
# DOCX paragraph styles whose content should be ignored during loading.
# Used to drop table-of-contents entries before chunking.
TOC_STYLES = {"toc heading", "toc 1", "toc 2", "toc 3", "toc 4"}


# ── Front-page metadata extraction ────────────────────────────────────────
# How much of the start of the document the LLM gets to read when looking for title / date / rig / reference_persons.
# Larger values catch more layouts but cost more tokens.
FRONTPAGE_BLOCK_LIMIT = 20         # Number of leading blocks to scan for signatures
FRONTPAGE_MAX_CONTEXT_CHARS = 2500 # Hard cap on characters fed to the LLM


# ── Chunking ──────────────────────────────────────────────────────────────
TEXT_CHUNK_SIZE = 1200       # Target characters per text chunk before splitting
TEXT_CHUNK_OVERLAP = 150     # Character overlap between consecutive chunks
CHUNK_BREAK_SEARCH_FRAC = 0.7  # When splitting a long paragraph, search this fraction
                               # of the chunk size backward for a sentence/whitespace
                               # break before falling back to a hard cut.


# ── Entity extraction ─────────────────────────────────────────────────────
# When False, the entity stage runs without the conceptual schema in
# graphrag/entities/schemas.py (CORE_SCHEMA, TYPE_DESCRIPTIONS, RELATIONSHIPS).
# The LLM picks its own entity_type / entity_class / relationship labels,
# and downstream stages discover the rel-type vocabulary from the live graph.
USE_CONCEPTUAL_SCHEMA = True

EXTRACTION_WORKERS = 4              # Parallel LLM calls during entity extraction
EXTRACTION_MAX_CHUNK_CHARS = 2800   # Hard cap on chunk text fed to one extraction call.
                                    # Larger -> fewer batches per chunk, higher token cost per call.
EXTRACTION_CTX_SIZE = 4096          # LLM context window for extraction calls (num_ctx).
                                    # Smaller than LLM_CONTEXT_SIZE because the extraction
                                    # prompt plus JSON output fits comfortably in 4k tokens.
SKIP_LIST_CHUNKS = False            # When True, list chunks are skipped during entity extraction.

# Section-title substrings that are always skipped during entity extraction.
# Case-insensitive substring match against chunk['section_title'].
DEFAULT_SKIP_SECTIONS = {
    "document history",
}


# ── Communities ───────────────────────────────────────────────────────────
COMMUNITY_MIN_SIZE = 2          # Communities with fewer entities than this are stripped
                                # after Leiden detection. Lower values keep more noise,
                                # higher values risk dropping small but real clusters.
COMMUNITY_SUMMARIES_MAX = 5     # Max community summaries included in the answer prompt.
COMMUNITY_SEMANTIC_TOP_K = 5    # Number of candidates the community embedding vector
                                # index returns for the semantic fallback (broad questions).


# ── Retrieval — Stage 1: document ranking ─────────────────────────────────
# Stage 1 scores every Document by combining a knowledge-graph entity match channel with a metadata keyword channel,
# then keeps documents whose score is within DOC_INCLUSION_RATIO of the top score.
# Documents within DOC_HIGH_CONF_RATIO of the top score are additionally treated as
# "high confidence" and trigger Stage 2a (full-document chunk inclusion).
KG_MAX_ENTITY_MATCHES = 50        # Skip KG query terms matching more than this many entities
DOC_INCLUSION_RATIO = 0.1         # Include docs scoring >= 10% of best score
DOC_HIGH_CONF_RATIO = 0.5         # Stage 2a only for docs scoring >= 50% of best


# ── Retrieval — Stage 2: graph chunk retrieval ────────────────────────────
# Each chunk reached by graph traversal carries a score that decays with the path length.
# Only chunks scoring >= GRAPH_MIN_SCORE survive, and the result is capped at GRAPH_MAX_CHUNKS.
GRAPH_DIRECT_SCORE = 1.0           # Score for a direct entity match
GRAPH_TYPED_REL_SCORE = 0.8        # Score for a 1-hop typed relationship
GRAPH_2HOP_SCORE = 0.4             # Score for a 2-hop traversal
GRAPH_MAX_CHUNKS = 50              # Maximum chunks returned by graph retrieval
GRAPH_MIN_SCORE = 0.39             # Minimum score to include a chunk
STAGE2A_BASELINE_SCORE = 0.2       # Score for document-level baseline chunks added by Stage 2a
                                   # (full-document inclusion for high-confidence docs that did
                                   # not already surface chunks via graph traversal).
CYPHER_CHUNK_SCORE = 0.6           # Score for chunks merged in from Stage 2c's Cypher chunk_id
                                   # results. Sits between the typed-relationship score (0.8) and
                                   # the 2-hop score (0.4) so Cypher hits rank in between.


# ── Retrieval — context assembly & boosts ─────────────────────────────────
CONTEXT_MAX_CHARS = 100000         # Safety cap on characters sent to the LLM
SECTION_TITLE_BOOST = 0.6          # Score floor for chunks whose section title matches a query term


# ── Retrieval — Stage 2c: text-to-Cypher ──────────────────────────────────
ENABLE_CYPHER_RETRIEVAL = True     # Toggle the text-to-Cypher stage
CYPHER_MAX_ROWS = 100              # Hard cap on result rows
CYPHER_TIMEOUT_S = 10              # Query timeout in seconds





# ── Chat ──────────────────────────────────────────────────────────────────
CHAT_MAX_HISTORY_TURNS = 6   # Number of prior turns the terminal chatbot keeps.
                             # Older turns are dropped from the in-memory log so
                             # the prompt stays bounded across long sessions.


# ── PDF processing (optional path. Tested and deemed not fit for final solution) ────────────────────────────────────────
# Only consulted when "pdf" is in DOCUMENT_FORMATS. 
# PDF support is kept in the codebase for showcase purposes; the production pipeline runs on DOCX only.
PDF_DATA_FOLDER = "bid_temp_pdf_storage/"  # Source PDF files (optional path)
ENABLE_PDF_TEXT_REPAIR = False           # LLM-based text repair for scrambled PDFs
PDF_HEADER_FOOTER_THRESHOLD = 0.4        # Fraction of pages text must appear on to count as header/footer
PDF_LINE_THRESHOLD = 5                   # Y-pixel proximity for line grouping
PDF_PARAGRAPH_THRESHOLD = 12             # Y-pixel gap for paragraph splitting
PDF_LIST_INDENT_THRESHOLD = 8            # X-pixel unit for list nesting inference
PDF_TABLE_BBOX_MARGIN = 5                # Pixel margin for table region text filtering
PDF_REPAIR_MIN_WORDS = 5                 # Min words before LLM text repair kicks in
PDF_REPAIR_BATCH_MAX_CHARS = 3000        # Hard cap on characters of text-block content fed
                                         # to one PDF text-repair LLM call. Larger means fewer
                                         # calls per document but bigger prompts per call.