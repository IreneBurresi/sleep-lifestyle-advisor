DATASET = data/raw/sleep_health.csv
URL = https://www.kaggle.com/api/v1/datasets/download/uom190346a/sleep-health-and-lifestyle-dataset

.DELETE_ON_ERROR:

data: $(DATASET)

$(DATASET):
	mkdir -p data/raw
	curl -fsSL -o data/raw/dataset.zip "$(URL)"
	unzip -p data/raw/dataset.zip "*.csv" > $(DATASET)
	rm data/raw/dataset.zip

.PHONY: data
