import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from advisor.cli.generate_guidance import make_agent, tool_calls
from advisor.config import LLMSettings
from advisor.models import Guidance
from advisor.prompting import Prompt
from advisor.rag import SOURCE_DIR, Chunk, extract_chunks


class FakeIndex:
    def search(self, question: str, k: int = 3) -> list[Chunk]:
        return [
            Chunk("healthy_sleep_guide_nih", "Caffeine", 31, "Caffeine keeps you awake. " * 40)
        ][:k]


def test_search_tool_is_registered_and_recorded(monkeypatch):
    monkeypatch.setattr(LLMSettings, "build_model", lambda self: TestModel())
    settings = LLMSettings(provider="google", model="x", api_key="k")  # type: ignore[arg-type]
    agent = make_agent(settings, Prompt.load("v2"), FakeIndex())  # ty: ignore[invalid-argument-type]
    assert isinstance(agent, Agent)
    result = agent.run_sync("prompt")
    assert isinstance(result.output, Guidance)
    calls = tool_calls(result.all_messages())
    assert calls and all(c["results"] == ["Your Guide to Healthy Sleep, p. 31"] for c in calls)


def test_prompt_addendum_only_with_index(monkeypatch):
    monkeypatch.setattr(LLMSettings, "build_model", lambda self: TestModel())
    settings = LLMSettings(provider="google", model="x", api_key="k")  # type: ignore[arg-type]
    prompt = Prompt.load("v2")
    assert "search_guidelines" in prompt.rag_addendum
    plain = make_agent(settings, prompt)
    assert "search_guidelines" not in str(plain._instructions)


def test_chunks_follow_headings():
    pdf = SOURCE_DIR / "healthy_sleep_guide_nih.pdf"
    if not pdf.exists():
        pytest.skip("run `make rag` first")
    chunks = extract_chunks(pdf, "healthy_sleep_guide_nih")
    assert 50 < len(chunks) < 200
    assert all(200 <= len(c.text) <= 1300 for c in chunks)
    assert any("Caffeine" in c.text for c in chunks)
