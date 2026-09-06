"""Generate wellness guidance for a user from their cluster profile and their own data.

    python -m advisor.cli.generate_guidance --all              # one typical user per cluster
    python -m advisor.cli.generate_guidance --cluster 3        # the most typical user of cluster 3
    python -m advisor.cli.generate_guidance --row 350          # a row of the dataset
    python -m advisor.cli.generate_guidance --user user.json   # a UserInput as JSON
    python -m advisor.cli.generate_guidance --testset eval/testset.json
    ... --rag           gives the model a search tool over the reference guidelines (needs `make rag`)
    ... --show-prompt   prints the prompt instead of calling the model
    ... --prompt-version v2   another directory under advisor/prompts/
    ... --out file.json saves the results for the evaluation step

Needs `make fit` and the LLM_* variables in `.env`.
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pydantic_ai import Agent, RunContext, UnexpectedModelBehavior
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelMessage, ThinkingPart, ToolCallPart, ToolReturnPart

from advisor.cli.fit import K
from advisor.config import LLMSettings
from advisor.llm import run_with_retries
from advisor.log import configure, logger
from advisor.models import Guidance, GuidanceRecord, UserInput
from advisor.profiling import Artifacts, Assignment, assign
from advisor.prompting import Prompt, available_versions
from advisor.rag import Chunk, Index

DEFAULT_PROMPT_VERSION = "v2"
MAX_SEARCHES = 2  # per user, see Searches
PASSAGE_CHARS = 700


@dataclass
class Searches:
    """The search tool's state for one run: the index and how many searches are left.
    Passed as deps, so every user starts with a full budget on the same agent."""

    index: Index | None = None  # None when the tool is off
    left: int = MAX_SEARCHES


def passage(chunk: Chunk) -> dict[str, str]:
    return {"source": chunk.cite(), "heading": chunk.heading, "text": chunk.text[:PASSAGE_CHARS]}


def make_agent(
    settings: LLMSettings, prompt: Prompt, rag: bool = False
) -> Agent[Searches, Guidance]:
    agent: Agent[Searches, Guidance] = Agent(
        settings.build_model(),
        name="guidance_agent",
        deps_type=Searches,
        output_type=Guidance,
        instructions=prompt.system + (prompt.rag_addendum if rag else ""),
        retries=2,
        model_settings=settings.model_settings(temperature=0.3, max_tokens=1500),
    )
    if rag:

        @agent.tool
        def search_guidelines(
            ctx: RunContext[Searches], question: str
        ) -> list[dict[str, str]] | str:
            """Search the reference guidelines on physical activity and sleep. Returns the three
            passages closest to the question, each with its source and page. At most two
            searches per user."""
            searches = ctx.deps
            if searches.index is None or searches.left == 0:
                return "No searches left; write the guidance."
            searches.left -= 1
            chunks = searches.index.search(question, k=3)
            logger.info("search_guidelines(%r) -> %s", question, [c.cite() for c in chunks])
            return [passage(c) for c in chunks]

    return agent


def tool_calls(messages: list[ModelMessage]) -> list[dict]:
    """The searches the model made and the passages that came back, from the run's message
    history. `results` lists the citations, `passages` the full texts the model saw."""
    calls, returns = {}, {}
    for message in messages:
        for part in message.parts:
            if isinstance(part, ToolCallPart) and part.tool_name == "search_guidelines":
                calls[part.tool_call_id] = part.args_as_dict().get("question", "")
            elif isinstance(part, ToolReturnPart):
                content = part.content if isinstance(part.content, list) else []
                returns[part.tool_call_id] = [
                    r for r in content if isinstance(r, dict) and r.get("source")
                ]
    return [
        {
            "query": q,
            "results": [r["source"] for r in returns.get(i, [])],
            "passages": returns.get(i, []),
        }
        for i, q in calls.items()
    ]


def generate(
    agent: Agent[Searches, Guidance],
    prompt: Prompt,
    assignment: Assignment,
    settings: LLMSettings,
    case: str,
    index: Index | None = None,
) -> GuidanceRecord:
    logger.info(
        "case=%s cluster=%d model=%s prompt=%s",
        case,
        assignment.cluster,
        settings.name,
        prompt.version,
    )
    started = time.perf_counter()
    result = run_with_retries(
        agent, prompt.render(assignment), settings.max_attempts, deps=Searches(index)
    )
    latency = time.perf_counter() - started
    calls = tool_calls(result.all_messages())
    thoughts = "\n".join(
        p.content for m in result.all_messages() for p in m.parts if isinstance(p, ThinkingPart)
    )
    logger.info(
        "case=%s done latency=%.2fs in=%d out=%d referral=%s searches=%d",
        case,
        latency,
        result.usage.input_tokens,
        result.usage.output_tokens,
        bool(assignment.referral_reasons),
        len(calls),
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
        thoughts=thoughts,
        rag=index is not None,
        tool_calls=calls,
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
    for call in record.tool_calls:
        print(f"[searched: {call['query']} -> {'; '.join(call['results'])}]")
    if record.thoughts:
        print(f"[thought: {record.thoughts[:200]!r}...]")
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
        "--rag", action="store_true", help="add the search tool over the guidelines"
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
    except (FileNotFoundError, ValueError) as e:  # pydantic's ValidationError is a ValueError
        print(e, file=sys.stderr)
        return 1
    assignments = [(label, assign(user, artifacts)) for label, user in users]

    if args.show_prompt:
        print(prompt.system)
        for label, assignment in assignments:
            print(f"# {label}\n{prompt.render(assignment)}")
        return 0

    try:
        settings = LLMSettings.load()
        index = Index(settings.api_key.get_secret_value()) if args.rag else None
        agent = make_agent(settings, prompt, rag=args.rag)
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1
    logger.info(
        "%d users, model=%s prompt=%s reasoning=%s rag=%s",
        len(assignments),
        settings.name,
        prompt.version,
        settings.reasoning,
        args.rag,
    )

    records = []
    for label, assignment in assignments:
        if records:
            time.sleep(settings.pause_seconds)
        try:
            record = generate(agent, prompt, assignment, settings, case=label, index=index)
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
