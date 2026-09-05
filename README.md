# Sleep and Lifestyle Advisor

Segments users into wellness profiles with clustering, then uses those profiles
to generate personalised guidance with an LLM.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Python 3.12

## Setup

```bash
uv sync
make data
```

`make data` downloads the [Sleep Health and Lifestyle dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)
into `data/raw/`.

## Development

```bash
uv run ruff check .
uv run ruff format .
uv run ty check
```
