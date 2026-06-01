"""Initialize the three runtime services used across the pipeline.

A single call to 'init_connections' returns the Neo4j driver, an embedding model,
and an LLM backend, all configured from the active config module.
Both the indexing pipeline and the interactive chat session use this entry point 
so the rest of the codebase never has to construct these objects directly.
"""

from neo4j import GraphDatabase
from langchain_ollama import OllamaEmbeddings

from graphrag import config
from graphrag.llm import OllamaLLM, OpenAILLM


def init_connections(cfg=None):
    """Build the Neo4j driver, embedding model, and LLM from config.

    Embeddings always go through Ollama ('OllamaEmbeddings') regardless of which LLM backend is selected.
    Azure-hosted embeddings are not wired up.

    Args:
        cfg: Config module (defaults to 'graphrag.config'). Any object exposing the same attribute names 
             will work, which lets tests and notebooks pass in overrides.

    Returns:
        Tuple '(driver, embedding_model, llm)'.
    """
    if cfg is None:
        cfg = config

    # ── Neo4j driver ──────────────────────────────────────────────────
    driver = GraphDatabase.driver(
        cfg.NEO4J_URI,
        auth=(cfg.NEO4J_USER, cfg.NEO4J_PASSWORD),
    )

    # ── Embedding model ───────────────────────────────────────────────
    embedding_model = OllamaEmbeddings(model=cfg.EMBEDDING_MODEL)

    # ── LLM backend ───────────────────────────────────────────────────
    # getattr() with a default lets alternative cfg objects omit these attributes without crashing.
    backend = getattr(cfg, "LLM_BACKEND", "ollama")
    timeout = getattr(cfg, "LLM_TIMEOUT", 180)
    max_retries = getattr(cfg, "LLM_MAX_RETRIES", 3)
    retry_delay = getattr(cfg, "LLM_RETRY_DELAY", 5.0)

    if backend == "azure":
        llm = OpenAILLM(
            base_url=cfg.AZURE_ENDPOINT,
            api_key=cfg.AZURE_API_KEY,
            model=cfg.AZURE_DEPLOYMENT,
            temperature=cfg.LLM_TEMPERATURE,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        llm_label = f"{cfg.AZURE_DEPLOYMENT} (Azure)"
    else:
        llm = OllamaLLM(
            model=cfg.LLM_MODEL,
            temperature=cfg.LLM_TEMPERATURE,
            num_ctx=cfg.LLM_CONTEXT_SIZE,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        llm_label = cfg.LLM_MODEL

    # ── Confirm ───────────────────────────────────────────────────────
    print(f"Neo4j driver initialized ({cfg.NEO4J_URI})")
    print(f"Embedding model: {cfg.EMBEDDING_MODEL}")
    print(f"LLM: {llm_label}")

    return driver, embedding_model, llm
