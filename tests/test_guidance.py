import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from advisor.models import Guidance, UserInput
from advisor.profiling import assign
from advisor.prompting import Prompt, available_versions


@pytest.mark.parametrize("version", available_versions())
def test_prompt_has_the_three_sections(version, raw, artifacts):
    a = assign(UserInput.from_row(raw.iloc[3].to_dict()), artifacts)
    text = Prompt.load(version).render(a)
    assert "## Group profile" in text and "## This person" in text and "## Fixed flags" in text
    assert "reports sleep apnea" in text
    assert "Reference notes" not in text


def test_v2_renders_reference_notes_only_when_given(raw, artifacts):
    a = assign(UserInput.from_row(raw.iloc[3].to_dict()), artifacts)
    prompt = Prompt.load("v2")
    with_notes = prompt.render(a, snippets=[{"source": "test", "text": "a note"}])
    assert "## Reference notes" in with_notes and "[test] a note" in with_notes


def test_unknown_prompt_version():
    with pytest.raises(FileNotFoundError, match="v99"):
        Prompt.load("v99")


def test_agent_runs_end_to_end_offline(raw, artifacts):
    a = assign(UserInput.from_row(raw.iloc[3].to_dict()), artifacts)
    prompt = Prompt.load("v2")
    agent = Agent(TestModel(), output_type=Guidance, instructions=prompt.system)
    result = agent.run_sync(prompt.render(a))
    assert isinstance(result.output, Guidance)
    assert 3 <= len(result.output.recommendations) <= 4


def test_retries_then_succeeds(monkeypatch):
    from unittest.mock import patch

    from pydantic_ai import ModelHTTPError

    from advisor import llm as g

    agent = Agent(TestModel(), output_type=Guidance)
    calls = {"n": 0}
    real = agent.run_sync

    def flaky(prompt):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ModelHTTPError(status_code=429, model_name="x", headers={"retry-after": "1"})
        return real(prompt)

    monkeypatch.setattr(g.time, "sleep", lambda s: None)
    with patch.object(agent, "run_sync", flaky):
        result = g.run_with_retries(agent, "prompt", max_attempts=4)
    assert calls["n"] == 3
    assert isinstance(result.output, Guidance)


def test_non_retryable_error_is_raised_at_once():
    from unittest.mock import patch

    from pydantic_ai import ModelHTTPError

    from advisor import llm as g

    agent = Agent(TestModel(), output_type=Guidance)

    def payment_required(prompt):
        raise ModelHTTPError(status_code=402, model_name="x")

    with patch.object(agent, "run_sync", payment_required), pytest.raises(ModelHTTPError):
        g.run_with_retries(agent, "prompt", max_attempts=4)
