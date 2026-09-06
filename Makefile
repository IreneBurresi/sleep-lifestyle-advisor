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

# rebuild the reference guidelines index in rag/index/ (downloads two public-domain PDFs, embeds with Gemini)
rag:
	uv run python -m advisor.cli.build_rag

# also writes profiles.json and metadata.json next to it
$(MODEL): $(DATASET) advisor/features.py advisor/cli/fit.py
	uv run python -m advisor.cli.fit

test:
	uv run pytest -q

# 1. regenerate the guidance for the test set and the red-team cases (calls the model)
guidance:
	uv run python -m advisor.cli.generate_guidance --testset eval/testset.json --out eval/runs/flash-lite_v2.json
	uv run python -m advisor.cli.generate_guidance --testset eval/red_team.json --out eval/runs/flash-lite_v2_redteam.json

# 1b. the same, with the search tool over the guidelines (needs `make rag`)
guidance-rag:
	uv run python -m advisor.cli.generate_guidance --testset eval/testset.json --rag --out eval/runs/flash-lite_v2_rag.json
	uv run python -m advisor.cli.generate_guidance --testset eval/red_team.json --rag --out eval/runs/flash-lite_v2_rag_redteam.json

# retrieval hit@3 on five questions, and the searches the model made in a run
eval-rag:
	uv run python -m advisor.cli.evaluate_rag --run eval/runs/flash-lite_v2_rag.json

# 2. keyword and structure checks on the saved runs (no model)
eval:
	uv run python -m advisor.cli.evaluate eval/runs/flash-lite_v2.json eval/runs/flash-lite_v2_redteam.json

# 3. the four LLM judges on the saved runs (calls the judge model)
judge:
	LLM_JUDGE_MODEL=$(JUDGE_MODEL) uv run python -m advisor.cli.evaluate eval/runs/flash-lite_v2.json eval/runs/flash-lite_v2_redteam.json --judge

# 4. two faithfulness judges on two --rag outputs: claims about the person vs the prompt, citations vs the retrieved passages
judge-faithfulness:
	LLM_JUDGE_MODEL=$(JUDGE_MODEL) uv run python -m advisor.cli.evaluate eval/runs/faithfulness/cluster-3-typical.json eval/runs/faithfulness/row-6.json --judge --rubrics faithfulness

.PHONY: data fit rag test guidance guidance-rag eval eval-rag judge judge-faithfulness
