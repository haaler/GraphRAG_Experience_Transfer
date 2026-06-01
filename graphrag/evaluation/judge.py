"""LLM-as-judge: score a system answer for correctness against the ground truth."""

import re
import json
from typing import Dict


JUDGE_PROMPT = """\
You are evaluating a question-answering system for offshore drilling engineering documents.

Question:
{question}

Reference answer (ground truth):
{reference_answer}

System answer:
{system_answer}

Score the system answer for CORRECTNESS on a 1 to 5 scale:
  1 = completely wrong
  2 = mostly wrong
  3 = partially correct
  4 = mostly correct
  5 = fully correct

Notes for scoring:
- If the reference answer says the information is not available in the documents,
  a correct system answer is one that also refuses to answer or says the
  information is missing. An answer that invents facts in that case is wrong.
- Wording does not have to match the reference exactly. What matters is that the
  factual content of the system answer agrees with the reference.
- Citations, formatting, and reasoning text are fine to ignore — judge only the
  factual answer.

Return ONLY valid JSON in this exact shape:
{{"correctness": N, "reasoning": "one short sentence explaining the score"}}
"""

JUDGE_SYSTEM_MSG = (
    "You are a strict evaluator for a document question-answering system. "
    "Score honestly. Do not give high scores unless they are clearly deserved. "
    "Output ONLY valid JSON, nothing else."
)


def judge_answer(question: str, reference_answer: str, system_answer: str, llm) -> Dict:
    """Ask the LLM to score the system answer's correctness against the reference.

    Returns a dict with two keys:
      correctness: int in [1, 5]
      reasoning:   short explanation from the judge

    On any failure (no JSON in the response, JSON parse error, missing key),
    returns correctness=1 with a reasoning string that flags the failure.
    """
    prompt = JUDGE_PROMPT.format(
        question=question,
        reference_answer=reference_answer,
        system_answer=system_answer,
    )

    try:
        raw = llm.invoke(prompt, system_msg=JUDGE_SYSTEM_MSG).strip()

        # Pull the first JSON object out of the response.
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match:
            result = json.loads(match.group(0))
            score = int(result.get("correctness", 1))
            score = max(1, min(5, score))
            return {
                "correctness": score,
                "reasoning": result.get("reasoning", ""),
            }

    except Exception as exc:
        return {"correctness": 1, "reasoning": f"Judge failed: {exc}"}

    return {"correctness": 1, "reasoning": "Judge returned no JSON"}
