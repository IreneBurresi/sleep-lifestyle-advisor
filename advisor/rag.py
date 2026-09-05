"""Reference guidelines for the guidance model: two public-domain PDFs converted to Markdown,
chunked on their headings, embedded with Gemini and stored in an embedded Qdrant Edge shard."""

import json
import re
import shutil
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf4llm
from google import genai
from google.genai import types
from qdrant_edge import (
    Distance,
    EdgeConfig,
    EdgeShard,
    EdgeVectorParams,
    Point,
    Query,
    QueryRequest,
    UpdateOperation,
)

from advisor.log import logger

SOURCES = {
    "physical_activity_guidelines_2018": {
        "title": "Physical Activity Guidelines for Americans, 2nd edition",
        "publisher": "U.S. Department of Health and Human Services, 2018",
        "url": "https://health.gov/sites/default/files/2019-09/Physical_Activity_Guidelines_2nd_edition.pdf",
    },
    "healthy_sleep_guide_nih": {
        "title": "Your Guide to Healthy Sleep",
        "publisher": "National Heart, Lung, and Blood Institute (NIH), 2011",
        "url": "https://www.nhlbi.nih.gov/sites/default/files/publications/11-5271.pdf",
    },
}
SOURCE_DIR = Path("data/sources")
INDEX_DIR = Path("rag/index")  # committed, so that the guidance runs without rebuilding it
EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_SIZE = 768
VECTOR = "text"
BATCH = 20  # texts per embedding request
MAX_CHUNK_CHARS = 1200
MIN_CHUNK_CHARS = 200
# Sections that are not guidance: front matter, indexes, the case-story boxes set in spaced capitals.
SKIP_HEADINGS = r"^(contents|table of contents|notes|acknowledgments|message from|glossary|appendix|references|for more|sample sleep diary|research|clinical research|learn more|nhlbi.*|.*\*\*.*|[A-Z](?: [A-Z])+(?: -)?\s*.*)$"


@dataclass(frozen=True)
class Chunk:
    source: str
    heading: str
    page: int
    text: str

    def cite(self) -> str:
        return f"{SOURCES[self.source]['title']}, p. {self.page}"


def download(directory: Path = SOURCE_DIR) -> dict[str, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, source in SOURCES.items():
        path = directory / f"{name}.pdf"
        if not path.exists():
            logger.info("downloading %s", source["url"])
            request = urllib.request.Request(source["url"], headers={"User-Agent": "Mozilla/5.0"})
            path.write_bytes(urllib.request.urlopen(request).read())
        paths[name] = path
    return paths


def extract_chunks(pdf_path: Path, source: str) -> list[Chunk]:
    """One chunk per Markdown heading, split further when longer than MAX_CHUNK_CHARS."""
    pages = pymupdf4llm.to_markdown(
        str(pdf_path),
        page_chunks=True,
        ignore_images=True,
        ignore_graphics=True,
        show_progress=False,
    )
    chunks: list[Chunk] = []
    heading, buffer, page_of_heading = "", [], 1

    def flush() -> None:
        text = re.sub(r"\s+", " ", " ".join(buffer)).strip()
        if len(text) >= MIN_CHUNK_CHARS and not re.match(SKIP_HEADINGS, heading, re.I):
            chunks.append(Chunk(source, heading, page_of_heading, text))
        buffer.clear()

    for page in pages:
        number = page["metadata"]["page_number"]
        for line in page["text"].splitlines():
            if match := re.match(r"^#{1,6}\s+(.*)$", line):
                flush()
                heading, page_of_heading = match.group(1).strip("* ").strip(), number
            elif line.strip() and not re.search(r"(\. ){4,}|\.{5,}", line):  # contents lines
                for sentence in re.split(r"(?<=[.!?])\s+", line.strip()):
                    if sum(len(t) for t in buffer) + len(sentence) > MAX_CHUNK_CHARS:
                        flush()
                        page_of_heading = number
                    buffer.append(sentence)
    flush()
    return chunks


class Embedder:
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def embed(self, texts: list[str], task: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH):
            batch = [
                types.Content(parts=[types.Part(text=t)]) for t in texts[start : start + BATCH]
            ]
            result = self.client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch,  # ty: ignore[invalid-argument-type]
                config=types.EmbedContentConfig(
                    task_type=task, output_dimensionality=EMBEDDING_SIZE
                ),
            )
            got = [list(e.values or []) for e in result.embeddings or []]
            if len(got) != len(batch):
                raise RuntimeError(f"asked {len(batch)} embeddings, got {len(got)}")
            vectors += got
        return vectors


def build(api_key: str, index_dir: Path = INDEX_DIR, source_dir: Path = SOURCE_DIR) -> int:
    chunks = [c for name, path in download(source_dir).items() for c in extract_chunks(path, name)]
    logger.info("%d chunks from %d sources", len(chunks), len(SOURCES))
    vectors = Embedder(api_key).embed(
        [f"{c.heading}\n{c.text}" for c in chunks], "RETRIEVAL_DOCUMENT"
    )
    shard_dir = index_dir / "shard"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True)
    config = EdgeConfig(
        vectors={VECTOR: EdgeVectorParams(size=EMBEDDING_SIZE, distance=Distance.Cosine)}
    )
    shard = EdgeShard.create(str(shard_dir), config)
    points = [
        Point(id=i, vector={VECTOR: v}, payload=asdict(c))
        for i, (c, v) in enumerate(zip(chunks, vectors, strict=True))
    ]
    shard.update(UpdateOperation.upsert_points(points))
    shard.flush()
    (index_dir / "chunks.json").write_text(json.dumps([asdict(c) for c in chunks], indent=1))
    return len(chunks)


class Index:
    def __init__(self, api_key: str, index_dir: Path = INDEX_DIR):
        shard = index_dir / "shard"
        if not shard.exists():
            raise FileNotFoundError(f"{shard} not found. Run `make rag` first.")
        self.shard = EdgeShard.load(str(shard))
        self.embedder = Embedder(api_key)

    def search(self, query: str, k: int = 3) -> list[Chunk]:
        vector = self.embedder.embed([query], "RETRIEVAL_QUERY")[0]
        result = self.shard.query(
            QueryRequest(
                query=Query.Nearest(vector, using=VECTOR),
                limit=k,
                with_vector=False,
                with_payload=True,
            )
        )
        return [Chunk(**(point.payload or {})) for point in result]
