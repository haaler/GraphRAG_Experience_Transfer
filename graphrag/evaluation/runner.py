"""Evaluation runner: ask each ground-truth question through the pipeline,
score the answer with an LLM judge, and report two headline numbers.

The runner groups questions by category and resets the conversation history
between categories. Within a category, follow-up questions inherit history
from previous turns (same as a real chat session).
"""

import json
import time
from collections import defaultdict, OrderedDict
from typing import List, Dict, Optional, Iterable

import pandas as pd

from graphrag.chat import chat_turn
from graphrag.evaluation.metrics import document_recall
from graphrag.evaluation.judge import judge_answer


def load_ground_truth(path: str) -> List[Dict]:
    """Read the ground-truth JSON file and return the list of test cases."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("questions", [])


def _group_by_category(test_cases: List[Dict]) -> "OrderedDict[str, List[Dict]]":
    """Group test cases by category, keeping the order in which categories
    first appear in the ground-truth file."""
    grouped: "OrderedDict[str, List[Dict]]" = OrderedDict()
    for tc in test_cases:
        category = tc.get("category", "unknown")
        grouped.setdefault(category, []).append(tc)
    return grouped


def run_evaluation(
    ground_truth_path: str,
    driver,
    embed_fn,
    llm,
    cfg=None,
    verbose: bool = True,
    history_reset_ids: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Run every ground-truth question through the pipeline and score it.

    For each question the runner records:
      doc_recall    fraction of expected source files the pipeline retrieved
      correctness   LLM-judge score (1-5) comparing system answer to ground truth

    The LLM conversation history is reset at the start of every category, so
    each category is evaluated as its own independent chat session. Within a
    category the runner uses ``chat_turn``, mirroring the chat REPL so
    follow-up questions inherit the doc_filter from the prior turn.

    ``history_reset_ids`` is an optional set of question ids whose history
    should be cleared **before** they run, on top of the category-boundary
    reset. Useful when the follow-up classifier mis-labels a question as a
    follow-up and inherits the wrong doc_filter.
    """
    test_cases = load_ground_truth(ground_truth_path)
    grouped = _group_by_category(test_cases)

    reset_ids = set(history_reset_ids or [])
    max_history_turns = getattr(cfg, "CHAT_MAX_HISTORY_TURNS", 6) if cfg else 6

    if verbose:
        print(f"Loaded {len(test_cases)} questions across {len(grouped)} category(ies)\n")

    results = []

    for category, cases in grouped.items():
        if verbose:
            print(f"── Category: {category} ({len(cases)} questions, history reset) ──")

        history: List[Dict] = []

        for tc in cases:
            qid = tc.get("id", "Q???")
            question = tc["question"]
            expected_docs = tc.get("expected_source_files", [])
            expected_answer = tc.get("expected_answer", "")

            if qid in reset_ids:
                history = []
                if verbose:
                    print(f"  (history reset before {qid})")

            if verbose:
                print(f"[{qid}] {question}")

            t0 = time.time()
            answer, metadata = chat_turn(
                question=question,
                history=history,
                driver=driver,
                embed_fn=embed_fn,
                llm=llm,
                cfg=cfg,
                verbose=False,
            )
            latency = round(time.time() - t0, 2)

            # doc_recall is computed from the actual answer text: for each
            # expected source file, check whether the answer mentions its
            # filename. End-to-end signal: did the system cite the right
            # source(s)? Not whether retrieval surfaced them internally.
            retrieved_docs = [f for f in expected_docs if f in answer]
            recall = document_recall(retrieved_docs, expected_docs)

            judge = judge_answer(
                question=question,
                reference_answer=expected_answer,
                system_answer=answer,
                llm=llm,
            )

            if verbose:
                recall_txt = "n/a" if recall is None else f"{recall:.2f}"
                print(f"  correctness={judge['correctness']}  "
                      f"doc_recall={recall_txt}  latency={latency}s\n")

            results.append({
                "question_id":     qid,
                "category":        category,
                "question":        question,
                "answer":          answer,
                "doc_recall":      recall,
                "correctness":     judge["correctness"],
                "judge_reasoning": judge["reasoning"],
                "latency_s":       latency,
                "n_chunks":        metadata.get("n_chunks", 0),
            })

            # Mirror chat.main(): store the docs the answer actually cited so the
            # next turn's follow-up classifier can inherit them as a doc_filter.
            all_docs = metadata.get("doc_filter", [])
            cited_docs = [sf for sf in all_docs if sf in answer]
            if not cited_docs:
                cited_docs = metadata.get("identified_documents", [])
            history.append({
                "question":   question,
                "answer":     answer,
                "doc_filter": cited_docs,
            })

            if len(history) > max_history_turns:
                history = history[-max_history_turns:]

    return pd.DataFrame(results)


def summarize(df: pd.DataFrame) -> Dict:
    """Build the small summary the notebook prints for the thesis.

    Overall correctness is the mean across every question. Overall doc recall
    is the mean across only the questions that had expected_source_files (so
    the 'negative' category, which has none, is excluded).
    """
    overall_correctness = float(df["correctness"].mean())
    overall_doc_recall = float(df["doc_recall"].dropna().mean()) if df["doc_recall"].notna().any() else None

    by_category: Dict[str, Dict] = {}
    for category, group in df.groupby("category"):
        recall_values = group["doc_recall"].dropna()
        by_category[category] = {
            "correctness": float(group["correctness"].mean()),
            "doc_recall":  float(recall_values.mean()) if len(recall_values) else None,
            "n":           len(group),
        }

    return {
        "overall_correctness": overall_correctness,
        "overall_doc_recall":  overall_doc_recall,
        "by_category":         by_category,
    }
