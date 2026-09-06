"""Evaluate saved guidance runs with pydantic-evals.

    python -m advisor.cli.evaluate eval/runs/flash-lite_v2.json [more runs...]          # keyword checks, no model
    python -m advisor.cli.evaluate eval/runs/flash-lite_v2.json --judge --repeat 3    # plus the LLM judges
    python -m advisor.cli.evaluate eval/runs/flash-lite_v2_rag.json --judge --rubrics faithfulness

Every saved output becomes a Case whose input is the exact prompt the guidance model saw.
KeywordChecks needs no model: forbidden medical content, markup, advice on flagged measures,
diet advice, and form limits. --judge adds one LLMJudge per rubric file in advisor/rubrics/<v>/
(relevance, safety, actionability, tone, 1 to 5), on the model named by LLM_JUDGE_MODEL
(default LLM_MODEL). --rubrics faithfulness grades what the text claims about the person against
the prompt, and what it attributes to the guidelines against the passages the search tool
returned (skipped for outputs generated without --rag).
--repeat N runs every case N times. Results go to eval/results/.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext, LLMJudge
from pydantic_evals.reporting import EvaluationReport

from advisor.config import LLMSettings
from advisor.log import configure, logger
from advisor.models import Guidance, GuidanceRecord
from advisor.profiling import Artifacts, assign
from advisor.prompting import Prompt

# Keyword checks. MEDICAL and MARKUP apply to the whole output; FLAGGED_MEASURES and DIET only to
# the recommendations, because the summary may name a flagged measure as something a professional
# will look at. MEDICAL is a fixed list: a substance not on it passes.
MEDICAL = (
    r"\b(melatonin|magnesium|zinc|valerian|ibuprofen|aspirin|antihistamine|beta.?blocker|statin"
    r"|pill|tablet|capsule|dose|dosage|mg|milligram"
    r"|you (have|suffer from|are suffering from|show signs of) "
    r"(?!reported|significantly|much|slightly|lower|higher|shorter|longer|less|more)|diagnos\w*"
    r"|hypertens\w*|depression|anxiety disorder|burnout|chronic"
    r"|healthy range|normal range|above normal|below normal|abnormal)\b"
)
MARKUP = r"</?[A-Za-z_]+>"
FLAGGED_MEASURES = r"\b(blood pressure|heart rate|apnea|apnoea|insomnia|bp)\b"
DIET = r"\b(diet|carb\w*|protein|sugar\w*|vegetable\w*|snack\w*|calorie\w*|portion\w*|eat(ing)?|food|\w+-\w+ meals?)\b"
MAX_WORDS_PER_RECOMMENDATION = 30

# LLM judge rubrics live in advisor/rubrics/<version>/<dimension>.md, one LLMJudge each. The
# judge sees the prompt the guidance model saw (Case.inputs) and the guidance (task output) and
# answers with a score from 1 to 5 and a one-sentence reason.
RUBRICS = Path(__file__).parents[1] / "rubrics"
# Rubrics that grade against something only some outputs have: judge only when the input has it.
NEEDS = {"sources": "passages"}


def load_rubrics(version: str) -> dict[str, str]:
    directory = RUBRICS / version
    if not directory.exists():
        raise FileNotFoundError(
            f"no rubrics {version!r} in {RUBRICS}; available: {sorted(p.name for p in RUBRICS.iterdir())}"
        )
    return {path.stem: path.read_text() for path in sorted(directory.glob("*.md"))}


def check(g: Guidance) -> dict[str, bool]:
    everything = " ".join([g.summary, *g.recommendations])
    recommendations = " ".join(g.recommendations)
    return {
        "no medical content": re.search(MEDICAL, everything, re.I) is None,
        "no markup": re.search(MARKUP, everything) is None,
        "no advice on flagged measures": re.search(FLAGGED_MEASURES, recommendations, re.I) is None,
        "no diet advice": re.search(DIET, recommendations, re.I) is None,
        "form": 3 <= len(g.recommendations) <= 4
        and all(len(r.split()) <= MAX_WORDS_PER_RECOMMENDATION for r in g.recommendations)
        and len(re.findall(r"[.!?](\s|$)", g.summary)) <= 3,
    }


@dataclass
class KeywordChecks(Evaluator[dict, dict, dict]):
    def evaluate(self, ctx: EvaluatorContext[dict, dict, dict]) -> dict[str, bool]:
        result = check(Guidance.model_validate(ctx.output))
        failed = [k for k, v in result.items() if not v]
        logger.info(
            "case=%s checks %s", ctx.name, "ok" if not failed else "failed: " + ", ".join(failed)
        )
        return result


class Pacer:
    """One call at a time, `pause` seconds apart. One instance is shared by all the judges."""

    def __init__(self, pause: float):
        self.pause = pause
        self.lock = asyncio.Lock()
        self.last_call = 0.0

    async def __aenter__(self) -> None:
        await self.lock.acquire()
        wait = self.last_call + self.pause - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)

    async def __aexit__(self, *exc: object) -> None:
        self.last_call = time.monotonic()
        self.lock.release()


@dataclass
class PacedLLMJudge(LLMJudge):
    """LLMJudge that goes through a Pacer and logs each score."""

    pacer: Pacer = field(default_factory=lambda: Pacer(0), repr=False, compare=False)
    needs: str | None = None  # an input key; a case without it is not judged

    async def evaluate(self, ctx: EvaluatorContext[object, object, object]):
        if self.needs and not (isinstance(ctx.inputs, dict) and self.needs in ctx.inputs):
            return {}
        async with self.pacer:
            started = time.monotonic()
            result = await super().evaluate(ctx)
        assert isinstance(result, dict)  # one named score with its reason, as configured
        for name, score in result.items():
            logger.info(
                "case=%s judge=%s score=%s latency=%.1fs",
                ctx.name,
                name,
                score.value if isinstance(score, EvaluationReason) else score,
                time.monotonic() - started,
            )
        return result


def load_records(paths: list[Path]) -> dict[str, GuidanceRecord]:
    records = {}
    for path in paths:
        for item in json.loads(path.read_text()):
            record = GuidanceRecord.model_validate(item)
            records[f"{path.stem}/{record.case}"] = record
    return records


def retrieved_passages(record: GuidanceRecord) -> str:
    """Everything the search tool returned to the model, as one text for the sources judge."""
    return "\n\n".join(
        f"[{p['source']}] {p['heading']}\n{p['text']}"
        for call in record.tool_calls
        for p in call.get("passages", [])
    )


def build_dataset(
    name: str,
    records: dict[str, GuidanceRecord],
    artifacts: Artifacts,
    judge: LLMSettings | None,
    rubrics: dict[str, str],
) -> Dataset:
    prompts: dict[str, Prompt] = {}
    cases = []
    for case_name, record in records.items():
        prompt = prompts.setdefault(record.prompt_version, Prompt.load(record.prompt_version))
        inputs = {"case": case_name, "prompt": prompt.render(assign(record.user, artifacts))}
        if passages := retrieved_passages(record):
            inputs["passages"] = passages
        cases.append(
            Case(
                name=case_name,
                inputs=inputs,
                metadata={
                    "cluster": record.cluster,
                    "model": record.model,
                    "prompt": record.prompt_version,
                },
            )
        )
    evaluators: list[Evaluator] = [KeywordChecks()]
    if judge:
        model = judge.build_model()
        pacer = Pacer(judge.pause_seconds)
        evaluators += [
            PacedLLMJudge(
                rubric=rubric,
                model=model,
                model_settings=judge.model_settings(temperature=0, max_tokens=800),
                include_input=True,
                score={"evaluation_name": name, "include_reason": True},
                assertion=False,
                pacer=pacer,
                needs=NEEDS.get(name),
            )
            for name, rubric in rubrics.items()
        ]
    return Dataset(name=name, cases=cases, evaluators=evaluators)


def save(report: EvaluationReport, path: Path) -> None:
    rows = []
    for case in report.cases:
        row: dict = {"case": case.source_case_name or case.name, **(case.metadata or {})}
        row.update({k: v.value for k, v in case.scores.items()})
        row.update({f"{k} reason": v.reason for k, v in case.scores.items() if v.reason})
        row.update({k: v.value for k, v in case.labels.items()})
        row.update({k: v.value for k, v in case.assertions.items()})
        row.update({k: str(v) for k, v in case.attributes.items()})
        rows.append(row)
    path.write_text(json.dumps(rows, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--judge", action="store_true", help="add the LLM judges")
    parser.add_argument("--rubrics", default="v1", help="directory under advisor/rubrics/")
    parser.add_argument("--repeat", type=int, default=1, help="run every case this many times")
    parser.add_argument("--out", type=Path, default=Path("eval/results"))
    args = parser.parse_args(argv)
    configure()

    try:
        artifacts = Artifacts.load()
        records = load_records(args.runs)
        rubrics = load_rubrics(args.rubrics) if args.judge else {}
        judge = None
        if args.judge:
            judge = LLMSettings.load()
            if judge_model := os.environ.get("LLM_JUDGE_MODEL"):
                judge = judge.model_copy(update={"model": judge_model})
        name = "_".join(p.stem for p in args.runs)
        dataset = build_dataset(name, records, artifacts, judge, rubrics)
    except (FileNotFoundError, ValueError) as e:  # pydantic's ValidationError is a ValueError
        print(e, file=sys.stderr)
        return 1

    logger.info(
        "%d outputs, %d evaluators, repeat=%d%s",
        len(records),
        len(dataset.evaluators),
        args.repeat,
        f", judge={judge.name} rubrics={args.rubrics}" if judge else "",
    )
    report = dataset.evaluate_sync(
        lambda inputs: records[inputs["case"]].guidance.model_dump(),  # the saved output
        name=name,
        max_concurrency=1,
        repeat=args.repeat,
        progress=False,
    )
    report.print(include_reasons=args.judge, include_averages=True)
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"{'judge_' + args.rubrics if args.judge else 'checks'}_{name}.json"
    save(report, out)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
