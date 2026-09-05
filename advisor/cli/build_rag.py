"""Download the reference PDFs, chunk them on headings, embed and index them.

    python -m advisor.cli.build_rag

Writes artifacts/rag/ (a Qdrant Edge shard and chunks.json). Needs the Gemini key in .env.
"""

import sys

from pydantic import ValidationError

from advisor.config import LLMSettings
from advisor.log import configure, logger
from advisor.rag import INDEX_DIR, build


def main() -> int:
    configure()
    try:
        settings = LLMSettings()  # type: ignore[call-arg]
    except ValidationError:
        print(
            "Set LLM_PROVIDER=google and LLM_API_KEY in .env: the index uses Gemini embeddings.",
            file=sys.stderr,
        )
        return 1
    n = build(settings.api_key.get_secret_value())
    logger.info("indexed %d chunks in %s", n, INDEX_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
