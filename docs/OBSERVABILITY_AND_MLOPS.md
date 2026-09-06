# Observability and MLOps

What this system would need to run for real users, sized to what it is: one fit a quarter, one LLM call per user per week, one person reviewing outputs. Nothing here is deployed; what exists in the repository is named as such.

## What to log per request

The guidance step already produces one `GuidanceRecord` per call (`advisor/models.py`), saved as JSON with `--out`, and `generate()` in `advisor/cli/generate_guidance.py` logs one line per case with the latency, the token counts, the searches and whether the referral fired. It holds:

| field | why it is there |
|---|---|
| `user`, `cluster`, `referral_reasons` | the input the model saw and the two decisions taken in code before the call: which group, which flags |
| `guidance` | the typed output, summary and recommendations |
| `model`, `prompt_version`, `reasoning`, `rag` | what produced the text: `provider:model`, the prompt directory, whether thinking was on, whether the search tool was on |
| `tool_calls` | each query the model made, the citations and the passages returned |
| `thoughts` | the model's thinking summary when reasoning is on, empty otherwise |
| `latency_seconds`, `input_tokens`, `output_tokens` | cost and speed per call, including retries |

For production, five more fields: a request id and a timestamp; the artifact version (the `created_at` and the dataset sha256 already in `artifacts/metadata.json`); the rubric version when a judge scores the output; the outcome of the keyword checks, run before the text is shown; and the user's reaction if the product collects one (dismissed, marked useful, reported).

**Privacy.** The record contains health data: measurements, a reported disorder, blood pressure. Logging the individual prompt is more sensitive than logging a cluster id, and the guidance step needs the individual data to work. Proposed split: the full record, with the raw prompt, kept 30 days in a store with restricted access, for debugging and for the human review sample; after that, only the derived fields survive (cluster, referral yes or no, check outcomes, judge scores, latency, tokens, prompt and model versions), keyed by a hashed user id. Occupation and gender are the only free text that reaches the model; they are context, not features, and the monitoring does not need them, so they are dropped from the long-term log. No record goes to a third party beyond the model provider, and the provider's data retention terms are part of the model choice. Blood pressure and a sleep disorder are health data, a special category under the GDPR and covered by HIPAA where that applies: consent for the processing, a data processing agreement with the provider and a documented retention period are required before a pilot.

## Detecting quality degradation

The three evaluation layers in `EVAL_NOTES.md` become three cadences:

| layer | on what | signal | threshold that opens a review |
|---|---|---|---|
| keyword checks (`KeywordChecks`, no model, milliseconds) | every output, before it is shown | pass rate per check per day | any check under 95% in a day, or any single failure on "no medical content"; a failed output is regenerated once and blocked if it fails again |
| LLM judges (four anchored rubrics, larger model) | a daily sample of 50 | mean per dimension, count of scores 1 and 2 | safety mean under 4.5, any safety score of 1 or 2, or a weekly mean down by 0.5 on any dimension from the baseline in `eval/results/` |
| a person (same four dimensions, 1 to 5) | 20 outputs a week, plus everything the checks blocked | agreement with the judge, notes | judge and human more than one point apart on more than a fifth of the sample: the rubric or the judge needs work |

Two cheaper signals that need no judge: the referral rate (162 of 374 in the training data; a sudden move means the input distribution or the thresholds changed) and the share of outputs whose summary says "typical" (a rise means the deltas stopped showing up in the text). One signal for monotony, which the human ratings flagged: the share of recommendation sets that contain the same phrase (a "ten minute" something is in 13 of 14, a walk in 12). A new prompt version is deployed only after it beats the current one on the saved test set and the red-team set, the way v2 was checked against v1.

## Detecting data drift

The population mean and standard deviation of every measure are in `profiles.json`, and every request is compared with them to compute the deltas. Drift is the aggregate of those deltas:

- **Input distribution**: the 30-day mean of each measure against the training mean, in training standard deviations. A shift of 0.3 SD on any measure, or of the BMI and disorder shares by ten points, opens a review.
- **Out-of-range inputs**: `UserInput` rejects values outside its plausibility bounds and the pipeline rejects an Underweight BMI. A rising daily rejection rate means the users are not the population the model saw.
- **Distance to the nearest centre**: the fit produces it for every training row (`typical_rows` uses it). A rising median over 30 days means new users fit the old groups less well.

## Detecting cluster drift

- **Group shares**: the training sizes are 101, 34, 145, 62, 32. A chi-square of the monthly assignment counts against them, or any group under 5% of assignments, means the segmentation describes fewer people than it did.
- **Refit against reassignment**: monthly, refit the same pipeline on the last 90 days of inputs and compare the two assignments of the same users with the adjusted Rand index, the test already used for stability in `CLUSTERING_NOTES.md`. Under 0.8 the groups have moved; the profiles and the focus areas, which are read from the group means, have to be rewritten for the new groups. The second number is the silhouette on the new data, against 0.43 on the distinct rows of the training data.
- **Reassignment and retraining are separate**: a user is reassigned on every request with the current artifacts; the artifacts change on a refit, quarterly or when the two checks above fire. A refit is a release: new `metadata.json`, new profiles, the guidance test set regenerated and re-judged before the switch.

## What to version, and with what

| thing | how it is versioned now | in production |
|---|---|---|
| dataset snapshot | sha256 and row count in `artifacts/metadata.json`; `make data` downloads a fixed source | the same hash, plus the snapshot itself in object storage |
| clustering model and profiles | `model.joblib`, `profiles.json`, `metadata.json` written together by one command; library versions inside | the three files as one artifact with the hash as its id |
| prompt templates | one directory per version under `advisor/prompts/`, the version name saved in every record | unchanged; a prompt change is a new directory and a run of `make guidance` and `make judge` |
| judge rubrics | one directory per version under `advisor/rubrics/`, the name in the result file | unchanged |
| LLM provider and model | the `provider:model` string in every record; `.env` for the running configuration | the same, with the provider's model version pinned where the API allows it |
| retrieval index | `rag/index/` committed with its chunks; the source documents and the embedding model named in `advisor/rag.py` | a hash of the chunks file in the record when the tool is on |

**Tooling.** For this system, git and JSON files are what is in place: every fit is reproducible from a hash and a seed, every LLM run and every evaluation is a file. MLflow, run locally with a file store and no server, is the next step when there is a second person or more than a handful of prompt and model combinations to compare: it gives the run table and the artifact links that `eval/runs/` and `eval/results/` provide today by naming convention. Weights & Biases or a hosted MLflow would add a service to operate for a model refit four times a year.
