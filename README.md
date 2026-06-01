# GraphRAG for Experience Transfer

A graph-based retrieval-augmented generation pipeline for engineering documents. Builds a Neo4j knowledge graph from a corpus of Word documents, enriches it with entity extraction and community summarisation, and answers natural-language questions over the graph through a terminal chat interface and a notebook-driven evaluation harness.

## 1. Requirements

- Python 3.10+
- Neo4j 5.x with the **Graph Data Science (GDS)** plugin installed (used for community detection)
- Ollama, running locally, for embeddings (and optionally the local LLM backend)
- An Azure OpenAI deployment if you intend to use the Azure LLM backend

Tested on Windows 11.

## 2. Setup

### 2.1 Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2.2 Start Neo4j with the GDS plugin

Install Neo4j Desktop, create a local database, and install the **Graph Data Science** plugin from the plugin marketplace. Start the database and note the Bolt URI (default `bolt://localhost:7687`) and the password you set.

### 2.3 Install Ollama and pull the embedding model

```bash
ollama pull qwen3-embedding:8b
```

If you want to run the LLM locally rather than through Azure, also pull the generation model named in `graphrag/config.py:LLM_MODEL`.

### 2.4 Create a `.env` file

Add a `.env` file in the project root containing:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=<your password>

# Only needed when LLM_BACKEND = "azure" in graphrag/config.py
AZURE_ENDPOINT=<your-resource>
AZURE_API_KEY=<your key>
AZURE_DEPLOYMENT=<your deployment name>
```

### 2.5 Place documents in the data folder

Put the DOCX files you want to index in `data_dump/`. The system relies on the filename naming convention
`<project>-<supplier>-<client>-<doctype>-<seq>-<rev>-<attachment>.docx` to extract metadata; documents that do not follow it will still be ingested but will have empty `project_number`, `supplier`, etc.

If you are using documents with other naming conventions or document which behaves differently, additional reconfigurations to how the code behaves will be needed.

## 3. Run the indexing pipeline

Open `GraphRAG main.ipynb` and execute every cell in order. This runs the three indexing-and-enrichment stages:

1. Document ingestion and chunking
2. Graph writing into Neo4j
3. Entity extraction, community detection, community summarisation

Indexing a corpus of ≈500 documents takes roughly 1–2 hours depending on LLM backend and machine.

## 4. Ask questions in the terminal

```bash
python graphrag/chat.py
```

Commands inside the REPL:

- `quit` / `exit` — leave the chat
- `clear` — reset the conversation history
- `history` — print the conversation so far

Each turn classifies the new question as a follow-up or new topic. For follow-ups it inherits the document filter from the prior turn so the search stays focused on the same documents.

## 5. Run the evaluation

In order to run the evaluation, a ground truth file in `graphrag/evaluation/ground_truth.json` needs to be defined and filled out. This is not present in this repository due to classified information.

The notebook functions by submitting the ground-truth questions defined in through the same pipeline `chat.py` uses, scores each answer with an LLM judge (correctness, 1–5) and measures whether the answer cites the expected source documents (doc_recall, 0–1). Results are aggregated overall and per category.

## 6. Configuration

All tuning parameters live in [`graphrag/config.py`](graphrag/config.py).
Notable switches:

- `LLM_BACKEND` — `"ollama"` for local, `"azure"` for Azure OpenAI
- `USE_CONCEPTUAL_SCHEMA` — `True` uses the curated entity schema in `graphrag/entities/schemas.py`; `False` lets the extraction LLM pick its own labels
- `DOCUMENT_FORMATS` — `{"docx"}` for the production path; add `"pdf"` to also ingest PDFs (uncomment the `gmft` line in `requirements.txt` first)
- `ENABLE_CYPHER_RETRIEVAL` — toggles the text-to-Cypher retrieval stage
- `CHAT_MAX_HISTORY_TURNS` — caps the number of prior turns the chat REPL keeps in memory

## 7. Repository layout

```
graphrag/
  ingestion/         DOCX parser, DocumentData interchange shape
  pdf_ingestion/     PDF parser (kept for showcase, not in active pipeline)
  chunking.py        Block → chunk conversion
  graph/             Neo4j schema, supersession check, node/edge writers
  entities/          Entity and relationship extraction
  communities/       Community detection (Leiden via Neo4j GDS) + summarisation
  retrieval/         Document routing, graph chunk retrieval, Cypher retrieval, context assembly
  evaluation/        Ground-truth runner, LLM judge, metric helpers
  chat.py            Terminal REPL entry point
  connections.py     Single place that builds the Neo4j driver, embedding model, and LLM
  config.py          All tuning parameters
GraphRAG main.ipynb        Run the indexing and enrichment pipeline
Evaluation.ipynb           Run the evaluation harness
```

## 8. Troubleshooting

- **"Neo4j driver: connection refused"** — Neo4j isn't running, or the Bolt URI in `.env` doesn't match the one shown in Neo4j Desktop.
- **"Procedure `gds.graph.project` not found"** — the GDS plugin isn't installed on the active database. Install it from the Neo4j Desktop plugin marketplace and restart the database.
- **"Connection refused: localhost:11434"** — Ollama isn't running. Start the Ollama service.
