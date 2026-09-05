"""Two small checks on the retrieval step.

    python -m advisor.cli.evaluate_rag                                   # hit@3 on eval/rag_questions.json
    python -m advisor.cli.evaluate_rag --run eval/runs/flash-lite_v2_rag.json   # plus the agent's searches in a run

Retrieval: for each question, do the top three passages come from the expected source and
contain the expected keyword. Trajectory: how many searches the model made per case, whether
each search returned something, and whether the output names a source.
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from advisor.config import LLMSettings
from advisor.log import configure, logger
from advisor.models import GuidanceRecord
from advisor.rag import SOURCES, Index

K = 3
SOURCE_WORDS = r"guidelines|guide to healthy sleep|NIH|Department of Health"


@dataclass
class TopKHit(Evaluator[dict, list[dict], dict]):
    def evaluate(self, ctx: EvaluatorContext[dict, list[dict], dict]) -> dict[str, bool]:
        expected = ctx.inputs
        return {
            "source in top k": any(c["source"] == expected["source"] for c in ctx.output),
            "keyword in top k": any(
                expected["keyword"].lower() in c["text"].lower() for c in ctx.output
            ),
        }


def retrieval(index: Index, questions: list[dict]) -> None:
    dataset = Dataset(
        name="retrieval",
        cases=[Case(name=q["question"][:50], inputs=q) for q in questions],
        evaluators=[TopKHit()],
    )
    report = dataset.evaluate_sync(
        lambda q: [{"source": c.source, "text": c.text} for c in index.search(q["question"], K)],
        progress=False,
    )
    report.print(include_averages=True)


def trajectory(records: list[GuidanceRecord]) -> None:
    for r in records:
        searches = len(r.tool_calls)
        empty = sum(not c["results"] for c in r.tool_calls)
        names_source = bool(re.search(SOURCE_WORDS, " ".join(r.guidance.recommendations), re.I))
        logger.info(
            "case=%s searches=%d empty=%d names_source=%s queries=%s",
            r.case,
            searches,
            empty,
            names_source,
            [c["query"] for c in r.tool_calls],
        )
    n = len(records)
    print(
        f"\n{n} cases: searched in {sum(bool(r.tool_calls) for r in records)}, "
        f"mean searches {sum(len(r.tool_calls) for r in records) / n:.1f}, "
        f"named a source in {sum(bool(re.search(SOURCE_WORDS, ' '.join(r.guidance.recommendations), re.I)) for r in records)}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--questions", type=Path, default=Path("eval/rag_questions.json"))
    parser.add_argument("--run", type=Path, help="a run made with --rag, for the trajectory check")
    args = parser.parse_args(argv)
    configure()
    try:
        settings = LLMSettings()  # type: ignore[call-arg]
        index = Index(settings.api_key.get_secret_value())
    except (FileNotFoundError, ValidationError) as e:
        print(e, file=sys.stderr)
        return 1
    questions = json.loads(args.questions.read_text())
    assert all(q["source"] in SOURCES for q in questions)
    retrieval(index, questions)
    if args.run:
        trajectory([GuidanceRecord.model_validate(r) for r in json.loads(args.run.read_text())])
    return 0


if __name__ == "__main__":
    sys.exit(main())
