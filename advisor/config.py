"""LLM settings from the environment or `.env`.

    LLM_PROVIDER=openrouter
    LLM_MODEL=minimax/minimax-m3
    LLM_API_KEY=...
    LLM_PAUSE_SECONDS=4   # optional, between consecutive calls

The generic key is copied into the provider's own variable, so pydantic-ai's model string
`provider:model` works for any provider it knows.
"""

import os

from pydantic import SecretStr
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

    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"

    def provider_settings(self) -> dict:
        """Whether the model thinks before answering, in each provider's own setting."""
        if self.provider == "openrouter":
            return {"openrouter_reasoning": {"enabled": self.reasoning}}
        if self.provider == "google" and not self.reasoning and "lite" not in self.model:
            return {"google_thinking_config": {"thinking_budget": 0}}  # lite models reject it
        return {}

    def build_model(self) -> Model:
        variable = API_KEY_VARIABLE.get(self.provider)
        if variable is None:
            raise SystemExit(
                f"LLM_PROVIDER={self.provider!r} is not in advisor/config.py. "
                f"Known: {', '.join(API_KEY_VARIABLE)}."
            )
        os.environ[variable] = self.api_key.get_secret_value()
        return infer_model(self.name)
