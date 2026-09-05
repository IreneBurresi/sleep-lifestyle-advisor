from pydantic_ai.models.test import TestModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge

from advisor.cli.evaluate import KeywordChecks, check, load_rubrics
from advisor.models import Guidance, GuidanceRecord, UserInput


def record(summary: str, recommendations: list[str], raw) -> GuidanceRecord:
    return GuidanceRecord(
        case="t",
        cluster=0,
        user=UserInput.from_row(raw.iloc[3].to_dict()),
        referral_reasons=[],
        guidance=Guidance(summary=summary, recommendations=recommendations),
        model="test",
        prompt_version="v2",
        reasoning=False,
        latency_seconds=0,
        input_tokens=0,
        output_tokens=0,
    )


def test_checks_catch_the_known_defects(raw, artifacts):
    bad = record(
        "You have hypertension and your numbers are outside the healthy range.",
        [
            "Take 3 mg of melatonin before bed.",
            "Get your blood pressure checked once a year.",
            "Swap one carb-heavy meal for vegetables and protein.",
        ],
        raw,
    )
    out = check(bad)
    failed = {k for k, v in out.items() if v is False}
    assert failed == {"no medical content", "no advice on flagged measures", "no diet advice"}


def test_checks_pass_a_clean_output(raw, artifacts):
    good = record(
        "You have reported sleep apnea. Your steps and sleep quality sit below your group.",
        [
            "Take a ten minute walk after lunch to lift your daily steps.",
            "Turn off screens thirty minutes before bed to help your sleep quality.",
            "Take five slow breaths when stress rises during the day.",
        ],
        raw,
    )
    assert all(check(good).values())


def test_dataset_runs_keyword_checks_and_an_offline_judge(raw, artifacts):
    rec = record(
        "You are typical of your group.",
        [
            "Walk ten minutes after lunch.",
            "Screens off before bed.",
            "Breathe slowly when stressed.",
        ],
        raw,
    )
    records = {"t": rec}
    dataset = Dataset(
        name="t",
        cases=[Case(name="t", inputs={"case": "t", "prompt": "the prompt"})],
        evaluators=[
            KeywordChecks(records),
            LLMJudge(
                rubric=load_rubrics("v1")["safety"],
                model=TestModel(),
                include_input=True,
                score={"evaluation_name": "safety"},
                assertion=False,
            ),
        ],
    )
    report = dataset.evaluate_sync(
        lambda inputs: records[inputs["case"]].guidance.model_dump(), progress=False
    )
    case = report.cases[0]
    assert case.assertions["no diet advice"].value is True
    assert "safety" in case.scores


def test_rubrics_have_the_four_dimensions():
    assert set(load_rubrics("v1")) == {"relevance", "safety", "actionability", "tone"}
