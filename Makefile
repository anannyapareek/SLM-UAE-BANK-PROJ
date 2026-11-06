PY := .venv/bin/python
SAMPLE_USERS := 20
SAMPLE_DIR := data/processed/sample_$(SAMPLE_USERS)
PROCESSED_DIR ?= data/processed

.PHONY: test-fast test test-mutation sample features report-prep stage2

test-fast:
	PROCESSED_DIR=$(PROCESSED_DIR) $(PY) -m pytest -m "not slow"

test:
	PROCESSED_DIR=$(PROCESSED_DIR) $(PY) -m pytest

sample:
	$(PY) -m src.run features --sample-users $(SAMPLE_USERS)
	$(PY) -m src.run features_text --sample-users $(SAMPLE_USERS)

test-mutation: sample
	$(PY) tests/mutation/mutate.py

features: sample
	$(MAKE) test-fast PROCESSED_DIR=$(SAMPLE_DIR)
	$(PY) -m src.run features
	$(PY) -m src.run features_text
	$(MAKE) test-fast

report-prep:
	$(PY) -m src.run prep_report > docs/prep_output.txt

stage2:
	$(PY) src/prep.py
	$(PY) -m src.run sessions
	$(MAKE) features
	$(MAKE) report-prep
	$(MAKE) test
	$(MAKE) test-mutation
