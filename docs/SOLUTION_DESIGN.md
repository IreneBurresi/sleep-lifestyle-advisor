# Solution design

Framing, dataset, architecture and risks. The detail behind each part is in `CLUSTERING_NOTES.md`, `GUIDANCE_NOTES.md` and `EVAL_NOTES.md`.

## Problem framing

"User profiling" here means two things:

1. **Segment**: put each user into one of a small number of wellness groups, found by clustering, and describe each group in words a person can read.
2. **Personalise**: give each user guidance that starts from their group's profile and then says where they, as an individual, differ from it.

With the segment alone there are five possible texts, and a lookup table would serve them without a language model. With the individual data alone there is no comparison with similar people. So the prompt carries the group profile, the person's own values with their distance from the group and from everyone, and a fixed list of what a professional should look at.

**In scope**: choosing and checking a dataset; a clustering pipeline with persisted artifacts; an LLM step that turns a user's row into a short summary and three or four habit recommendations; an evaluation with automatic checks, LLM judges and a human rubric; an optional retrieval step over two public guidelines; the documents.

**Out of scope**: an API or a web app; real users and outcomes; clinical advice; deployment.

**Level of personalisation**: on the segment and on the individual, and the evaluation tests the second explicitly with three members of the same cluster, one typical and two far from the centre on different measures.

## Dataset

The [Sleep Health and Lifestyle dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset) (Kaggle, CC0): 374 people, 13 columns. Per person: age, gender, occupation, hours of sleep, self-rated sleep quality and stress (1 to 10), minutes of physical activity per day, daily steps, resting heart rate, blood pressure as text, BMI category, and a reported sleep disorder.

Two other candidates were tried first with the same null-baseline test (silhouette of the clustering against shuffled columns): a mental-health-and-lifestyle dataset of 3000 rows showed no structure at all (the gap to the shuffled baseline was 0.0001), and a food-and-nutrition dataset had no person-level rows to aggregate. Those two trials are not kept in the repository.

### What the EDA found, and what each finding decided

The counts below are computed from `data/raw/sleep_health.csv`; the distributions and correlations are in `notebooks/eda.ipynb`, the feature decisions in `notebooks/features.ipynb`, and the referral counts come from applying the rule in `advisor/profiling.py` to every row.

| finding | decision |
|---|---|
| Sleep Disorder is empty in 219 rows; every other column is complete | empty means "no disorder", made explicit at load; no imputation anywhere else |
| Ages 27 to 59, 189 men and 185 women; sleep 5.8 to 8.5 hours, quality 4 to 9, stress 3 to 8, resting heart rate 65 to 86; BMI 216 normal, 148 overweight, 10 obese | nothing to rebalance and no extreme values to clip; the narrow ranges are what the profiles describe, and a user outside them is out of the training range |
| Blood pressure is text ("126/83"); the two numbers correlate 0.97 | split, then reduced to one feature (mean arterial pressure) |
| Activity minutes and daily steps correlate 0.77 | averaged into one activity index after standardisation |
| Values sit on a grid: 20 distinct step counts, 91% of people on five of them; 242 of 374 rows repeat another row exactly (132 distinct rows, 106 on the clustering features) | repeated rows stay in the fit; every score is also reported on the distinct rows so the repeats do not inflate it; no outlier treatment, the extremes are rare levels |
| Correlations higher than in real measurements: sleep quality and stress −0.90, duration and quality 0.88, age and systolic pressure 0.61 | the data is synthetic and every document says so; the clusters describe the generator; the pipeline transfers to real data, the profiles do not |
| "Normal Weight" and "Normal" both appear in BMI Category (21 and 195 rows); no Underweight | merged; an Underweight input is rejected as out of the training range rather than silently mapped |
| Women have a median age of 50, men 38; occupation is one of eleven values | gender and occupation are not clustering features (with them, groups split by gender or by job); both are passed to the model as context only |
| Blood pressure of 140/90 or more in 100 people, a reported disorder in 155, together 162 of 374 | a referral rule in code with fixed thresholds (disorder, 140/90, resting heart rate 100), computed before the model runs and printed with the guidance |

Scaling: every feature standardised before KMeans. Encoding: BMI as an ordinal 0/1/2. Dropped: Person ID, gender, occupation, the disorder (kept aside to check the groups).

### Known limits

- **Size**: 374 rows, 106 distinct on the features; two of the five groups are 9 and 8 distinct rows repeated.
- **Self-reported**: sleep quality and stress are 1-to-10 ratings; steps and activity minutes are as reported, not measured.
- **Representativeness**: ages 27 to 59, eleven white-collar occupations, no Underweight, three of five groups almost entirely one gender because age and gender are correlated in this data.
- **Synthetic**: the grid, the repeats and the correlations show it. Silhouette 0.52 at k = 5 (0.43 on distinct rows) against 0.13 on shuffled data is a gap real measurements would not give.
- **Missing values**: only the disorder column, where empty means none.

## Architecture

```
data/raw/sleep_health.csv  ──make fit──▶  artifacts/            ──▶  advisor.profiling.assign
   (374 rows, sha256 kept)                 model.joblib                 cluster, deltas, referral
                                           profiles.json                        │
                                           metadata.json                        ▼
                                                                        advisor/prompts/v2/   ──▶  pydantic-ai Agent  ──▶  Guidance
                                                                        system.md, user.md.j2       Gemini Flash-Lite        summary + 3-4 recs
                                                                        (+ rag.md, off by default)  typed output, retries         │
                                                                                  ▲                                             ▼
                                                                        rag/index/  (Qdrant Edge,                     eval/runs/*.json  (GuidanceRecord)
                                                                        375 chunks of two public                                │
                                                                        guidelines, search tool)                                ▼
                                                                                                          advisor.cli.evaluate: keyword checks,
                                                                                                          LLM judges (pydantic-evals); human ratings in a CSV
```

- **Fit** (`advisor/cli/fit.py`): one scikit-learn pipeline, features → scaler → KMeans with k = 5 and 200 restarts, saved with joblib; `profiles.json` holds each group's summary, its focus areas, its means and its distance from everyone; `metadata.json` holds the dataset hash, the parameters and the library versions. The fit refuses to write if the cluster sizes are not the ones the written summaries describe.
- **Assign** (`advisor/profiling.py`): a validated `UserInput` goes through the same pipeline; the result is the cluster, the deltas from the group and from everyone in population standard deviations, and the referral reasons.
- **Guidance** (`advisor/cli/generate_guidance.py`): a Jinja template renders three labelled sections; a pydantic-ai agent with a typed output returns a summary and three or four recommendations; provider, model and key come from `.env`; retries with backoff on rate limits and server errors. With `--rag` the agent has a search tool over the guidelines, at most two searches per user.
- **Evaluate** (`advisor/cli/evaluate.py`, `evaluate_rag.py`): every saved output becomes a pydantic-evals case; five keyword checks run without a model; four judges with anchored rubrics and two faithfulness judges call a larger model; a human rubric with the same four dimensions closes the loop.

At guidance time `model.joblib` and `profiles.json` are read, not retrained. Reassigning a user is a prediction; changing the groups is a refit; the two happen on different schedules.

## Risks

| risk | what is done about it | what remains |
|---|---|---|
| Wrong cluster, wrong guidance | assignment is deterministic; the prompt carries the individual deltas, so the text depends on the person's numbers and not only on the group; boundary cases (smallest margin to the next centre) are in the test set | on real data the groups would have to be found again; boundary users get a group that fits them less |
| Medical advice, or advice on flagged measures | referral thresholds in code, never set by the model; the prompt forbids advice on blood pressure, heart rate and disorders, and clinical words; keyword checks on every output; an anchored safety rubric; validation that rejects markup | the recurring defect is a referral written as a recommendation, seen in one case of nine; the keyword check catches it, and the judge does too now that the rubric names it; in two cases the summary sends a measure that was not flagged to a professional, which only the human rating caught |
| Overconfident model | temperature 0.3, form limits, "compared with people in this data" instead of "healthy" or "normal" | two evaluative words in fourteen outputs ("healthy baselines", "great health") got through the checks and were caught by the judge and the human rating |
| Stale profiles | the fit is dated and hashed; `OBSERVABILITY_AND_MLOPS.md` says what to watch and when to refit | nothing runs on a schedule in this repository |
| Overfitting a small sample | 200 restarts and a stability test by resampling (ARI 0.95 to 1.00 for every k); scores on distinct rows; a shuffled-data baseline | with values on a grid every partition is stable, so stability does not choose k; k = 5 was chosen for the guidance step |
| Prompt injection through free-text fields | occupation and gender are context, not instructions; one injection case in the red-team set | one phrasing in one field was tested |
