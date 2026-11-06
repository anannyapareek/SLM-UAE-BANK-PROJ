PY := .venv/bin/python
SAMPLE_USERS := 20
SAMPLE_DIR := data/processed/sample_$(SAMPLE_USERS)
PROCESSED_DIR ?= data/processed

# Run a stage into a temp file and replace the doc only if the stage succeeds.
report = $(PY) -m src.run $(1) > $(2).tmp && mv $(2).tmp $(2) || { rm -f $(2).tmp; exit 1; }

.PHONY: test-fast test test-mutation sample features report-prep report-sessions report-features report-text stage2

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
	$(MAKE) report-features
	$(MAKE) report-text
	$(MAKE) test-fast

report-prep:
	$(call report,prep_report,docs/prep_output.txt)

report-sessions:
	$(call report,sessions,docs/sessions_output.txt)

report-features:
	$(call report,features,docs/features_output.txt)

report-text:
	$(call report,features_text,docs/features_text_output.txt)

stage2:
	$(PY) src/prep.py
	$(MAKE) report-sessions
	$(MAKE) features
	$(MAKE) report-prep
	$(MAKE) test
	$(MAKE) test-mutation
