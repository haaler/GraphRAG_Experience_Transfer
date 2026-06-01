"""Terminal chatbot for the GraphRAG pipeline with conversation memory.

Run from a shell with 'python graphrag/chat.py' after the index has been populated by the notebook.
Each prompt is first classified by the LLM as either a follow-up to a prior turn or a fresh topic. 
Follow-ups inherit the doc_filter from the referenced turn so the search stays focused on
the same documents, while fresh topics re-run Stage 1 routing inside ask().

The last CHAT_MAX_HISTORY_TURNS turns of question/answer pairs (plus their cited documents) 
are kept in memory and fed back into the next turn so the LLM can resolve pronouns and continuations.
"""

import sys
from pathlib import Path

# Put the project root on sys.path so this file can be run as a script
# (`python graphrag/chat.py`) without the package being installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import graphrag.config as config
from graphrag.connections import init_connections
from graphrag.retrieval.ask import ask

CLASSIFY_PROMPT = """\
You are a conversation analyst. Given the conversation history and a new question,
determine if the new question is a follow-up to a previous question or a new topic.

Conversation history:
{history}

New question: {question}

Respond with ONLY valid JSON, no explanation:
{{"is_followup": true/false, "reference_index": <1-based index of the question it follows up on, or null>}}
"""


def _classify_followup(question, history, llm):
    """Determine if the question is a follow-up and which prior question it refers to.

    Returns the 0-based index into history, or None if standalone.
    """
    if not history:
        return None

    history_text = "\n".join(
        f"[{i+1}] Q: {turn['question']}" for i, turn in enumerate(history)
    )

    raw = llm.invoke(
        CLASSIFY_PROMPT.format(history=history_text, question=question),
        system_msg="You classify questions as follow-ups or new topics. Respond with JSON only.",
    )

    try:
        raw = raw.strip()
        # The LLM sometimes wraps its JSON response in ``` fences despite the system instruction. 
        # Strip that and the optional "json" hint so the downstream json.loads still works.
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw)

        if result.get("is_followup") and result.get("reference_index") is not None:
            idx = int(result["reference_index"]) - 1  # convert to 0-based
            if 0 <= idx < len(history):
                return idx
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    return None


def chat_turn(question, history, driver, embed_fn, llm, cfg=None, verbose=True):
    """Execute one chat turn: retrieve context, generate answer with history.

    Returns (answer, metadata) so callers can store doc_filter in history.
    Set verbose=False to silence the turn-level prints (the eval runner uses this).
    """

    # Classify whether this is a follow-up or a new question
    ref_idx = _classify_followup(question, history, llm)
    search_query = question
    prior_doc_filter = None

    if ref_idx is not None:
        ref_question = history[ref_idx]["question"]
        search_query = f"{question} {ref_question}"
        prior_doc_filter = history[ref_idx].get("doc_filter")
        if verbose:
            print(f"   (follow-up to: {ref_question[:60]}{'...' if len(ref_question) > 60 else ''})")

    answer, metadata = ask(
        question=question,
        driver=driver,
        embed_fn=embed_fn,
        llm=llm,
        cfg=cfg,
        history=history,
        verbose=verbose,
        doc_filter=prior_doc_filter,
        retrieval_query=search_query,
    )

    if verbose:
        n_chunks = metadata.get("n_chunks", 0)
        n_docs = len(metadata.get("identified_documents", []))
        has_entity_ctx = bool(metadata.get("entity_context"))
        print(f"   {n_chunks} chunks from {n_docs} doc(s)"
              + (" + entity context" if has_entity_ctx else ""))

    return answer, metadata


def main():
    print("=" * 60)
    print("  GraphRAG Chat — Offshore Drilling Documents")
    print("=" * 60)
    print()
    print("  Commands:")
    print("    quit / exit  — stop the chatbot")
    print("    clear        — reset conversation history")
    print("    history      — show conversation so far")
    print()

    driver, emb_model, llm = init_connections(config)
    print("  Ready.\n")

    max_history_turns = getattr(config, "CHAT_MAX_HISTORY_TURNS", 6)
    history = []

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        if not question:
            continue

        cmd = question.lower()
        if cmd in ("quit", "exit", "q"):
            print("\nGoodbye!")
            break
        if cmd == "clear":
            history.clear()
            print("  Conversation history cleared.")
            continue
        if cmd == "history":
            if not history:
                print("  (no history yet)")
            else:
                for i, turn in enumerate(history, 1):
                    print(f"\n  [{i}] Q: {turn['question']}")
                    # Preview only: the full answer is too noisy in a terminal echo.
                    ans_preview = turn["answer"][:200]
                    if len(turn["answer"]) > 200:
                        ans_preview += "..."
                    print(f"      A: {ans_preview}")
            continue

        print("  Searching...")
        answer, metadata = chat_turn(question, history, driver, emb_model.embed_query, llm, cfg=config)

        print(f"\nAssistant:\n{answer}\n")

        # Match cited documents against the full doc_filter (not just chunk context docs),
        # since the LLM may cite documents found via entity context or
        # Cypher results that aren't in the chunk selection.
        all_docs = metadata.get("doc_filter", [])
        cited_docs = [source_file for source_file in all_docs if source_file in answer]
        if not cited_docs:
            cited_docs = metadata.get("identified_documents", [])

        history.append({
            "question": question,
            "answer": answer,
            "doc_filter": cited_docs,
        })
        if len(history) > max_history_turns:
            history = history[-max_history_turns:]


if __name__ == "__main__":
    main()
