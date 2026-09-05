DATASET = data/raw/sleep_health.csv
URL = https://www.kaggle.com/api/v1/datasets/download/uom190346a/sleep-health-and-lifestyle-dataset
MODEL = artifacts/model.joblib
JUDGE_MODEL = gemini-3.5-flash

.DELETE_ON_ERROR:

data: $(DATASET)

$(DATASET):
	mkdir -p data/raw
	curl -fsSL -o data/raw/dataset.zip "$(URL)"
	unzip -p data/raw/dataset.zip "*.csv" > $(DATASET)
	rm data/raw/dataset.zip

fit: $(MODEL)

# also writes profiles.json and metadata.json next to it
$(MODEL): $(DATASET) advisor/features.py advisor/cli/fit.py
	uv run python -m advisor.cli.fit

test:
	uv run pytest -q

# 1. regenerate the guidance for the test set and the red-team cases (calls the model)
guidance:
	uv run python -m advisor.cli.generate_guidance --testset eval/testset.json --out eval/runs/flash-lite_v2.json
	uv run python -m advisor.cli.generate_guidance --testset eval/red_team.json --out eval/runs/flash-lite_v2_redteam.json

# 2. keyword and structure checks on the saved runs (no model)
eval:
	uv run python -m advisor.cli.evaluate eval/runs/flash-lite_v2.json eval/runs/flash-lite_v2_redteam.json

# 3. the four LLM judges on the saved runs (calls the judge model)
judge:
	LLM_JUDGE_MODEL=$(JUDGE_MODEL) uv run python -m advisor.cli.evaluate eval/runs/flash-lite_v2.json eval/runs/flash-lite_v2_redteam.json --judge

.PHONY: data fit test guidance eval judge
