"""LLM settings from the environment or `.env`.

    LLM_PROVIDER=openrouter
    LLM_MODEL=minimax/minimax-m3
    LLM_API_KEY=...
    LLM_PAUSE_SECONDS=4   # optional, between consecutive calls

The generic key is copied into the provider's own variable, so pydantic-ai's model string
`provider:model` works for any provider it knows.
"""

import os
from typing import Self

from pydantic import SecretStr, ValidationError
from pydantic_ai import ModelSettings
from pydantic_ai.models import Model, infer_model
from pydantic_settings import BaseSettings, SettingsConfigDict

API_KEY_VARIABLE = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistral": "MISTRAL_API_KEY",
}


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=".env", extra="ignore")

    provider: str
    model: str
    api_key: SecretStr
    timeout_seconds: float = 60
    max_attempts: int = 4
    pause_seconds: float = 4  # between consecutive calls; free tiers cap requests per minute
    reasoning: bool = False

    @classmethod
    def load(cls) -> Self:
        """From the environment and `.env`; a missing variable is a ValueError with the fix."""
        try:
            return cls()  # type: ignore[call-arg]  # the required fields come from the environment
        except ValidationError as e:
            raise ValueError(
                "Set LLM_PROVIDER, LLM_MODEL and LLM_API_KEY in .env (see .env.example)."
            ) from e

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    def model_settings(self, temperature: float, max_tokens: int) -> ModelSettings:
        """Sampling, timeout and whether the model thinks first, in the provider's own keys."""
        settings: dict = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": self.timeout_seconds,
        }
        if self.provider == "openrouter":
            settings["openrouter_reasoning"] = {"enabled": self.reasoning}
        elif self.provider == "google" and "lite" not in self.model:  # lite rejects the config
            settings["google_thinking_config"] = (
                {"include_thoughts": True} if self.reasoning else {"thinking_budget": 0}
            )
        return ModelSettings(**settings)  # type: ignore[typeddict-item]  # provider keys

    def build_model(self) -> Model:
        variable = API_KEY_VARIABLE.get(self.provider)
        if variable is None:
            raise ValueError(
                f"LLM_PROVIDER={self.provider!r} is not in advisor/config.py. "
                f"Known: {', '.join(API_KEY_VARIABLE)}."
            )
        os.environ[variable] = self.api_key.get_secret_value()
        return infer_model(self.name)
