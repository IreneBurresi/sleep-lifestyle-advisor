"""Reference guidelines for the guidance model: two public-domain PDFs, chunked on their
headings, embedded with Gemini and stored in an embedded Qdrant Edge shard on disk."""

import collections
import json
import shutil
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import pdfplumber
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
INDEX_DIR = Path("artifacts/rag")
EMBEDDING_MODEL = "gemini-embedding-2"
EMBEDDING_SIZE = 768
VECTOR = "text"
BATCH = 20  # texts per embedding request
HEADING_SIZE_DELTA = 3  # a line whose font is this much larger than the body text is a heading
MAX_CHUNK_CHARS = 1200
MIN_CHUNK_CHARS = 200


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
    """One chunk per heading, split further when longer than MAX_CHUNK_CHARS."""
    chunks: list[Chunk] = []
    heading, buffer, page_of_heading = "", [], 1

    def flush() -> None:
        text = " ".join(buffer).strip()
        if len(text) >= MIN_CHUNK_CHARS:
            chunks.append(Chunk(source, heading, page_of_heading, text))
        buffer.clear()

    with pdfplumber.open(pdf_path) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            words = page.extract_words(extra_attrs=["size"])
            if not words:
                continue
            body = collections.Counter(round(w["size"]) for w in words).most_common(1)[0][0]
            lines: dict[int, list] = collections.defaultdict(list)
            for w in words:
                lines[round(w["top"])].append(w)
            for top in sorted(lines):
                line = lines[top]
                text = " ".join(w["text"] for w in line)
                if all(w["size"] >= body + HEADING_SIZE_DELTA for w in line) and len(text) < 90:
                    flush()
                    heading, page_of_heading = text, number
                else:
                    buffer.append(text)
                    if sum(len(t) for t in buffer) > MAX_CHUNK_CHARS:
                        flush()
                        page_of_heading = number
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
