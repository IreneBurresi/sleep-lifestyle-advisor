"""Model calls with retries, shared by the guidance and the judge."""

import random
import time

from pydantic_ai import Agent, ModelHTTPError
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.exceptions import ModelAPIError

from advisor.log import logger

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}
BACKOFF_BASE_SECONDS = 5  # free-tier 429s clear in tens of seconds: 5, 15, 45
BACKOFF_FACTOR = 3


def run_with_retries[DepsT, OutputT](
    agent: Agent[DepsT, OutputT], user_prompt: str, max_attempts: int, deps: DepsT
) -> AgentRunResult[OutputT]:
    """Retry on rate limits, server errors and network errors with exponential backoff,
    honouring Retry-After when the server sends it."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return agent.run_sync(user_prompt, deps=deps)
        except ModelHTTPError as e:
            if e.status_code not in RETRYABLE_STATUS or attempt == max_attempts:
                raise
            reason, retry_after = f"HTTP {e.status_code}", (e.headers or {}).get("retry-after")
        except ModelAPIError as e:
            if attempt == max_attempts:
                raise
            reason, retry_after = type(e).__name__, None
        wait = BACKOFF_BASE_SECONDS * BACKOFF_FACTOR ** (attempt - 1) + random.uniform(0, 1)
        if retry_after and retry_after.isdigit():
            wait = max(wait, int(retry_after))
        logger.warning("%s, retrying in %.0fs (%d/%d)", reason, wait, attempt, max_attempts)
        time.sleep(wait)
