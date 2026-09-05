"""Generate wellness guidance for a user from their cluster profile and their own data.

    python -m advisor.cli.generate_guidance --all              # one typical user per cluster
    python -m advisor.cli.generate_guidance --cluster 3        # the most typical user of cluster 3
    python -m advisor.cli.generate_guidance --row 350          # a row of the dataset
    python -m advisor.cli.generate_guidance --user user.json   # a UserInput as JSON
    python -m advisor.cli.generate_guidance --testset eval/testset.json
    ... --show-prompt   prints the prompt instead of calling the model
    ... --prompt-version v2   another directory under advisor/prompts/
    ... --out file.json saves the results for the evaluation step

Needs `make fit` and the LLM_* variables in `.env`.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from pydantic import ValidationError
from pydantic_ai import Agent, ModelSettings, UnexpectedModelBehavior
from pydantic_ai.exceptions import ModelAPIError

from advisor.cli.fit import K
from advisor.config import LLMSettings
from advisor.llm import run_with_retries
from advisor.log import configure, logger
from advisor.models import Guidance, GuidanceRecord, UserInput
from advisor.profiling import Artifacts, Assignment, assign
from advisor.prompting import Prompt, available_versions

DEFAULT_PROMPT_VERSION = "v2"


def make_agent(settings: LLMSettings, prompt: Prompt) -> Agent[None, Guidance]:
    return Agent(
        settings.build_model(),
        name="guidance_agent",
        output_type=Guidance,
        instructions=prompt.system,
        retries=2,
        model_settings=ModelSettings(
            temperature=0.3,
            max_tokens=1500,
            timeout=settings.timeout_seconds,
            **settings.provider_settings(),  # type: ignore[typeddict-item]
        ),
    )


def generate(
    agent: Agent[None, Guidance],
    prompt: Prompt,
    assignment: Assignment,
    settings: LLMSettings,
    case: str,
) -> GuidanceRecord:
    logger.info(
        "case=%s cluster=%d model=%s prompt=%s",
        case,
        assignment.cluster,
        settings.name,
        prompt.version,
    )
    started = time.perf_counter()
    result = run_with_retries(agent, prompt.render(assignment), settings.max_attempts)
    latency = time.perf_counter() - started
    logger.info(
        "case=%s done latency=%.2fs in=%d out=%d referral=%s",
        case,
        latency,
        result.usage.input_tokens,
        result.usage.output_tokens,
        bool(assignment.referral_reasons),
    )
    return GuidanceRecord(
        case=case,
        cluster=assignment.cluster,
        user=assignment.user,
        referral_reasons=assignment.referral_reasons,
        guidance=result.output,
        model=settings.name,
        prompt_version=prompt.version,
        reasoning=settings.reasoning,
        latency_seconds=round(latency, 2),
        input_tokens=result.usage.input_tokens,
        output_tokens=result.usage.output_tokens,
    )


def print_record(record: GuidanceRecord) -> None:
    g = record.guidance
    print(f"\n=== {record.case} (cluster {record.cluster}) ===")
    print(g.summary)
    for line in g.recommendations:
        print(f"- {line}")
    if record.referral_reasons:
        print(f"Talk to a professional about: {'; '.join(record.referral_reasons)}")
    print(
        f"[{record.model} | prompt {record.prompt_version} | {record.latency_seconds}s | "
        f"{record.input_tokens} in / {record.output_tokens} out]"
    )


def typical_rows(raw: pd.DataFrame, artifacts: Artifacts) -> dict[int, int]:
    """For each cluster, the row closest to its centre."""
    distance = artifacts.distances_to_centres(artifacts.scale(raw))
    return {c: int(distance[:, c].argmin()) for c in range(distance.shape[1])}


def select_users(args: argparse.Namespace, artifacts: Artifacts) -> list[tuple[str, UserInput]]:
    if args.user:
        return [(Path(args.user).stem, UserInput.model_validate_json(Path(args.user).read_text()))]
    if args.testset:
        cases = json.loads(Path(args.testset).read_text())
        return [(c["id"], UserInput(**c["user"])) for c in cases]
    raw = pd.read_csv(args.data)
    if args.row is not None:
        return [(f"row-{args.row}", UserInput.from_row(raw.iloc[args.row].to_dict()))]
    typical = typical_rows(raw, artifacts)
    clusters = sorted(typical) if args.all else [args.cluster]
    return [
        (f"cluster-{c}-typical", UserInput.from_row(raw.iloc[typical[c]].to_dict()))
        for c in clusters
    ]


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    who = parser.add_mutually_exclusive_group(required=True)
    who.add_argument("--all", action="store_true", help="the most typical user of each cluster")
    who.add_argument(
        "--cluster", type=int, choices=range(K), help="the most typical user of one cluster"
    )
    who.add_argument("--row", type=int, help="a row of the dataset")
    who.add_argument("--user", help="path to a JSON file with the fields of UserInput")
    who.add_argument("--testset", help="path to eval/testset.json, every case in it")
    parser.add_argument("--data", type=Path, default=Path("data/raw/sleep_health.csv"))
    parser.add_argument(
        "--prompt-version", default=DEFAULT_PROMPT_VERSION, help=f"one of {available_versions()}"
    )
    parser.add_argument(
        "--show-prompt", action="store_true", help="print the prompt, do not call the model"
    )
    parser.add_argument("--out", type=Path, help="save the results as JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure()
    try:
        artifacts = Artifacts.load()
        prompt = Prompt.load(args.prompt_version)
        users = select_users(args, artifacts)
    except (FileNotFoundError, ValidationError) as e:
        print(e, file=sys.stderr)
        return 1
    assignments = [(label, assign(user, artifacts)) for label, user in users]

    if args.show_prompt:
        print(prompt.system)
        for label, assignment in assignments:
            print(f"# {label}\n{prompt.render(assignment)}")
        return 0

    try:
        settings = LLMSettings()  # type: ignore[call-arg]  # values come from the environment
    except ValidationError:
        print(
            "Set LLM_PROVIDER, LLM_MODEL and LLM_API_KEY in .env (see .env.example).",
            file=sys.stderr,
        )
        return 1
    agent = make_agent(settings, prompt)
    logger.info(
        "%d users, model=%s prompt=%s reasoning=%s",
        len(assignments),
        settings.name,
        prompt.version,
        settings.reasoning,
    )

    records = []
    for label, assignment in assignments:
        if records:
            time.sleep(settings.pause_seconds)
        try:
            record = generate(agent, prompt, assignment, settings, case=label)
        except (ModelAPIError, UnexpectedModelBehavior) as e:
            logger.error("case=%s failed %s: %s", label, type(e).__name__, str(e)[:200])
            continue
        records.append(record)
        print_record(record)

    if args.out and records:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps([r.model_dump(mode="json") for r in records], indent=2))
        print(f"\nsaved {len(records)} results to {args.out}")
    return 0 if len(records) == len(assignments) else 1


if __name__ == "__main__":
    sys.exit(main())
