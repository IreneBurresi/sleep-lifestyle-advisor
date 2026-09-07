# Sleep and Lifestyle Advisor

Segments users into wellness profiles with clustering, then uses those profiles
to generate personalised guidance with an LLM.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Python 3.12

## Setup

```bash
uv sync            # installs the project and the dev tools
make test          # 39 tests pass offline; the one skipped needs the guideline PDFs
```

The dataset and the fitted model are committed, so nothing has to be downloaded or trained
before running the guidance and the evaluation. `make data` and `make fit` rebuild them:

```bash
make data          # downloads the CSV again (same file, same sha256)
make fit           # refits the model and rewrites artifacts/
```

Without uv: `pip install -r requirements.txt` in a Python 3.12 environment, then run the
`python -m advisor.cli.<command>` lines that the Makefile wraps, from the repo root (the Makefile
targets call `uv run`). `pip install pytest` for the tests.

`make data` downloads the [Sleep Health and Lifestyle dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)
(Kaggle, CC0 public domain, 374 rows) into `data/raw/`. No Kaggle account is needed.

`make fit` trains the clustering model and writes `artifacts/`:

| file | what it is |
|---|---|
| `model.joblib` | one scikit-learn pipeline, features → scaler → KMeans; `predict()` takes raw rows |
| `profiles.json` | what each cluster looks like, plus population mean and std |
| `metadata.json` | when it was fitted, sha256 of the data, versions |

These three files are committed, so the guidance step runs on a fresh clone; it loads them and
does not retrain. Refitting is deterministic and overwrites them with the same content, except
`created_at`. The two guideline PDFs behind `--rag` are not committed (17 MB); `rag/index/` holds
what was built from them.

## Guidance

Copy `.env.example` to `.env` and fill in the provider, the model and one API key.
The default is `google` with `gemini-3.5-flash-lite`: a free key from
[Google AI Studio](https://ai.google.dev) is enough. Any provider pydantic-ai knows works
(`openai`, `anthropic`, `openrouter`, ...); some need an extra dependency group, for example
`uv add "pydantic-ai-slim[anthropic]"`. Free tiers cap requests per minute and per day: the
script paces calls, retries with backoff and reports the cases it could not generate.

```bash
uv run python -m advisor.cli.generate_guidance --all          # one typical user per cluster
uv run python -m advisor.cli.generate_guidance --cluster 3    # the most typical user of cluster 3
uv run python -m advisor.cli.generate_guidance --row 350      # a row of the dataset
uv run python -m advisor.cli.generate_guidance --user me.json # your own data, fields as in UserInput
```

Add `--show-prompt` to see the prompt without calling the model, `--out file.json` to save
the results, `--prompt-version v1` to run the earlier prompt (`advisor/prompts/`), `--testset file.json` to run
every case of a saved test set, `--rag` to give the model a search tool over two public guidelines
(off by default, see the notes). The index under `rag/index/` is committed; `make rag` rebuilds it
from the PDFs. Queries and chunks are embedded with Google's `gemini-embedding-2` through the same
`LLM_API_KEY`, so `--rag`, `make rag` and `make eval-rag` need the `google` provider even though
the guidance itself can run on any. The prompt has three labelled sections: the cluster profile with its focus
areas, the person's numbers compared with their cluster and with everyone, and fixed flags.
The flags are computed in code, printed with the guidance, and the model is told not to
advise on them; thresholds in `docs/GUIDANCE_NOTES.md`.

## Evaluation

```bash
make guidance   # regenerate eval/runs/flash-lite_v2*.json for the 9 test cases and the 5 red-team cases
make guidance-rag  # the same with the search tool; make eval-rag checks retrieval and the searches made
make eval       # keyword and structure checks on the saved runs, no model
make judge      # four LLM judges (relevance, safety, actionability, tone) on the saved runs
make judge-faithfulness  # two more, on two --rag outputs: claims about the person vs the prompt, citations vs the retrieved passages
```

The judges run on a larger model than the one that wrote the guidance (`JUDGE_MODEL` in the
Makefile, `LLM_JUDGE_MODEL` for the script), on the provider and key in `.env`: with a provider
other than `google`, pass a model it serves, `make judge JUDGE_MODEL=...`. They are pydantic-evals `LLMJudge` evaluators,
one per rubric file in `advisor/rubrics/`. `python -m advisor.cli.evaluate --help` lists the
options (`--repeat 3` for stability, `--rubrics` to pick a rubric directory). The test set
comes from `python -m advisor.cli.build_testset`. Runs and results are committed under `eval/`;
the rubric, the human ratings and what the checks catch or miss are in `docs/EVAL_NOTES.md`.

## Documents

The three documents the brief asks for, then the notes for each part:

- [`docs/SOLUTION_DESIGN.md`](docs/SOLUTION_DESIGN.md): framing and scope, the dataset and what the EDA decided, architecture, risks.
- [`docs/OBSERVABILITY_AND_MLOPS.md`](docs/OBSERVABILITY_AND_MLOPS.md): what to log, how to detect quality, data and cluster drift, what to version and with what.
- [`docs/STAKEHOLDER_REPORT.md`](docs/STAKEHOLDER_REPORT.md): what was built and not built, the evaluation in plain terms, residual risk, recommendation, next two days.
- [`docs/CLUSTERING_NOTES.md`](docs/CLUSTERING_NOTES.md): features, choice of k, profiles, checks, limits.
- [`docs/GUIDANCE_NOTES.md`](docs/GUIDANCE_NOTES.md): what the model receives, model choice with measured cost and latency, retrieval, prompt iteration v1 to v2.
- [`docs/EVAL_NOTES.md`](docs/EVAL_NOTES.md): test set, automatic checks, LLM judges, faithfulness, rubric and human ratings.

## Layout

```
advisor/            the package: features, profiling, prompting, llm, rag, models, config
advisor/cli/        fit, generate_guidance, build_testset, evaluate, evaluate_rag, build_rag
advisor/prompts/    one directory per prompt version (system.md, user.md.j2, rag.md)
advisor/rubrics/    one directory per judge rubric version, one file per dimension
docs/               the documents listed above
notebooks/          EDA, features, clustering; analysis/ for the checks behind k and the columns
eval/               testset.json, red_team.json, rag_questions.json; runs/ (saved outputs), results/ (checks, judge scores, human ratings)
data/raw/           the dataset; make data downloads it again
artifacts/          the fitted pipeline, the cluster profiles and the fit metadata
rag/index/          the committed retrieval index
figures/            plots used by the notes
tests/              offline, no model calls
```

## Notebooks

Read in this order: `notebooks/eda.ipynb`, `notebooks/features.ipynb`,
`notebooks/clustering.ipynb`. The two in `notebooks/analysis/` hold the checks
behind the choice of k and of the columns.

## Development

```bash
make test          # pytest, no network
uv run ruff check .
uv run ruff format .
uv run ty check
```
