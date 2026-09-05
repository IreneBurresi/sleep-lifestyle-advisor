# Sleep and Lifestyle Advisor

Segments users into wellness profiles with clustering, then uses those profiles
to generate personalised guidance with an LLM.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Python 3.12

## Setup

```bash
uv sync            # or: pip install -r requirements.txt, then run the commands from the repo root
make data          # downloads the CSV
make fit           # needs the CSV; everything below needs the artifacts it writes
```

`make data` downloads the [Sleep Health and Lifestyle dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)
(Kaggle, CC0 public domain, 374 rows) into `data/raw/`. No Kaggle account is needed.

`make fit` trains the clustering model and writes `artifacts/`:

| file | what it is |
|---|---|
| `model.joblib` | one scikit-learn pipeline, features → scaler → KMeans; `predict()` takes raw rows |
| `profiles.json` | what each cluster looks like, plus population mean and std |
| `metadata.json` | when it was fitted, sha256 of the data, versions |

Neither data nor artifacts are committed. The guidance step loads the artifacts and
does not retrain.

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
the results, `--prompt-version v1` to run the earlier prompt (`advisor/prompts/`), `--rag` to give
the model a search tool over two public guidelines (after `make rag`, which downloads the PDFs
and builds an embedded Qdrant index under `artifacts/rag/`; off by default, see the notes). The prompt has three labelled sections: the cluster profile with its focus
areas, the person's numbers compared with their cluster and with everyone, and fixed flags.
The flags are computed in code, printed with the guidance, and the model is told not to
advise on them; thresholds in `docs/GUIDANCE_NOTES.md`.

## Evaluation

```bash
make guidance   # regenerate eval/runs/flash-lite_v2*.json for the 9 test cases and the 5 red-team cases
make guidance-rag  # the same with the search tool; make eval-rag checks retrieval and the searches made
make eval       # keyword and structure checks on the saved runs, no model
make judge      # four LLM judges (relevance, safety, actionability, tone) on the saved runs
```

The judges run on a larger model than the one that wrote the guidance (`JUDGE_MODEL` in the
Makefile, `LLM_JUDGE_MODEL` for the script); they are pydantic-evals `LLMJudge` evaluators,
one per rubric file in `advisor/rubrics/`. `python -m advisor.cli.evaluate --help` lists the
options (`--repeat 3` for stability, `--rubrics` to pick a rubric directory). The test set
comes from `python -m advisor.cli.build_testset`. Runs and results are committed under `eval/`;
the rubric, the human ratings and what the checks catch or miss are in `docs/EVAL_NOTES.md`.

## Documents

- [`docs/CLUSTERING_NOTES.md`](docs/CLUSTERING_NOTES.md): features, choice of k, profiles, checks, limits.
- [`docs/GUIDANCE_NOTES.md`](docs/GUIDANCE_NOTES.md): what the model receives, model choice with measured cost and latency, prompt iteration v1 to v2.
- [`docs/EVAL_NOTES.md`](docs/EVAL_NOTES.md): test set, automatic checks, LLM judge, rubric and human ratings.

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
