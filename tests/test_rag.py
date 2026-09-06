from types import SimpleNamespace

import pytest
from google import genai
from google.genai import errors
from pydantic import SecretStr
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from advisor import rag
from advisor.cli.generate_guidance import Searches, make_agent, tool_calls
from advisor.config import LLMSettings
from advisor.models import Guidance
from advisor.prompting import Prompt
from advisor.rag import SOURCE_DIR, Chunk, extract_chunks


class FakeIndex:
    def search(self, question: str, k: int = 3) -> list[Chunk]:
        return [
            Chunk("healthy_sleep_guide_nih", "Caffeine", 31, "Caffeine keeps you awake. " * 40)
        ][:k]


@pytest.fixture
def settings(monkeypatch) -> LLMSettings:
    monkeypatch.setattr(LLMSettings, "build_model", lambda self: TestModel())
    return LLMSettings(provider="google", model="x", api_key=SecretStr("k"))


@pytest.fixture
def embedder(monkeypatch) -> rag.Embedder:
    monkeypatch.setattr(rag.time, "sleep", lambda s: None)
    return rag.Embedder(genai.Client(api_key="k"))


def test_search_tool_is_registered_and_recorded(settings):
    agent = make_agent(settings, Prompt.load("v2"), rag=True)
    assert isinstance(agent, Agent)
    searches = Searches(FakeIndex())  # ty: ignore[invalid-argument-type]
    result = agent.run_sync("prompt", deps=searches)
    assert isinstance(result.output, Guidance)
    calls = tool_calls(result.all_messages())
    assert calls and all(c["results"] == ["Your Guide to Healthy Sleep, p. 31"] for c in calls)
    assert all(c["passages"][0]["heading"] == "Caffeine" for c in calls)
    assert searches.left == 2 - len(calls)


def test_search_budget_is_per_run(settings):
    agent = make_agent(settings, Prompt.load("v2"), rag=True)
    exhausted = Searches(FakeIndex(), left=0)  # ty: ignore[invalid-argument-type]
    result = agent.run_sync("prompt", deps=exhausted)
    assert all(c["results"] == [] for c in tool_calls(result.all_messages()))
    fresh = Searches(FakeIndex())  # ty: ignore[invalid-argument-type]
    agent.run_sync("prompt", deps=fresh)
    assert fresh.left < 2


def test_prompt_addendum_only_with_index(settings):
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


def test_embedding_retries_on_rate_limit(embedder, monkeypatch):
    calls = []

    def embed_content(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise errors.APIError(429, {"error": {"message": "quota"}})
        return SimpleNamespace(embeddings=[SimpleNamespace(values=[0.0] * 3)])

    monkeypatch.setattr(embedder.client.models, "embed_content", embed_content)
    assert embedder.embed(["one"], "RETRIEVAL_QUERY") == [[0.0, 0.0, 0.0]]
    assert len(calls) == 2


def test_embedding_does_not_retry_a_bad_request(embedder, monkeypatch):
    def embed_content(**kwargs):
        raise errors.APIError(400, {"error": {"message": "bad"}})

    monkeypatch.setattr(embedder.client.models, "embed_content", embed_content)
    with pytest.raises(errors.APIError):
        embedder.embed(["one"], "RETRIEVAL_QUERY")
