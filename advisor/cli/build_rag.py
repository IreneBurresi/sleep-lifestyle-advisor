"""Download the reference PDFs, chunk them on headings, embed and index them.

    python -m advisor.cli.build_rag

Writes rag/index/ (a Qdrant Edge shard and chunks.json), which is committed so that --rag works
without rebuilding. Needs the Gemini key in .env.
"""

import sys

from advisor.config import LLMSettings
from advisor.log import configure, logger
from advisor.rag import INDEX_DIR, build


def main() -> int:
    configure()
    try:
        settings = LLMSettings.load()
        n = build(settings.api_key.get_secret_value())
    except ValueError as e:  # missing settings, or a key the Gemini client rejects
        print(f"{e} The index uses Gemini embeddings.", file=sys.stderr)
        return 1
    logger.info("indexed %d chunks in %s", n, INDEX_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
