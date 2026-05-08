.PHONY: help setup smoke data train train-quick eda threshold interpret transfer evaluate all report clean test lint

PYTHON ?= python3
PIP ?= pip

# Hyperparameter search budgets. Override on the command line:
#   make train RF_RANDOM_N_ITER=10 XGB_BAYES_N_ITER=15
RF_RANDOM_N_ITER  ?= 30
XGB_BAYES_N_ITER  ?= 50
N_SYNTHETIC       ?= 80000

# nbconvert flags. Long timeout because tuning takes a while on real BTS data.
NBCONVERT = jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200

help:
	@echo "Targets:"
	@echo "  setup        Install Python dependencies (requirements.txt)"
	@echo "  smoke        Run end-to-end smoke test on synthetic data (~30s)"
	@echo "  data         Download BTS + FAA registry (~2-3 GB)"
	@echo ""
	@echo "  train        Run NB_03 with full Bayesian tuning budget"
	@echo "                 RF: $(RF_RANDOM_N_ITER) iters | XGB: $(XGB_BAYES_N_ITER) iters"
	@echo "                 Writes artefacts/best_hyperparams.json,"
	@echo "                        artefacts/model_comparison.csv,"
	@echo "                        artefacts/best_model.joblib"
	@echo "                 Prints the headline-numbers block for the report."
	@echo "  train-quick  Same as train but with reduced budget (~3 min on synthetic)"
	@echo ""
	@echo "  eda          Run NB_01 (LAMA's notebook -- EDA + figures)"
	@echo "  threshold    Run NB_04 (ADAM's notebook -- tau* heatmap + profit curve)"
	@echo "  interpret    Run NB_05 (SALMA's notebook -- SHAP + failure modes)"
	@echo "  transfer     Run NB_06 (SANAD's notebook -- EU transfer validation)"
	@echo ""
	@echo "  evaluate     Run NB_04, NB_05, NB_06 in sequence"
	@echo "  all          Full pipeline:  smoke -> train -> eda -> evaluate -> report"
	@echo "  report       Render NB_99 to HTML in reports/build/"
	@echo ""
	@echo "  test         Run pytest"
	@echo "  clean        Remove caches and intermediate artefacts"

setup:
	$(PIP) install -r requirements.txt

smoke:
	$(PYTHON) scripts/smoke_test.py

data:
	$(PYTHON) scripts/download_bts.py --years 2018 2019 2020 2021 2022 2023 2024
	$(PYTHON) scripts/download_faa.py

# ---------- Modelling and evaluation -----------------------------------------

train:
	@echo ">>> NB_03: Bayesian tuning on profit_scorer (RF=$(RF_RANDOM_N_ITER), XGB=$(XGB_BAYES_N_ITER) iters)"
	RF_RANDOM_N_ITER=$(RF_RANDOM_N_ITER) \
	XGB_BAYES_N_ITER=$(XGB_BAYES_N_ITER) \
	N_SYNTHETIC=$(N_SYNTHETIC) \
	$(NBCONVERT) notebooks/03_modeling.ipynb
	@echo ""
	@echo ">>> Headline numbers (also printed inline in the notebook):"
	@echo ""
	@$(PYTHON) -c "import json; \
	d = json.load(open('artefacts/best_hyperparams.json')); \
	print('  best_hyperparams.json entries:', list(d.keys())); \
	print()"
	@$(PYTHON) -c "import pandas as pd; \
	r = pd.read_csv('artefacts/model_comparison.csv').sort_values('ROI_perflight', ascending=False); \
	print('  Top 3 by ROI:'); print(r.head(3).to_string(index=False))"

train-quick:
	@$(MAKE) train RF_RANDOM_N_ITER=4 XGB_BAYES_N_ITER=6 N_SYNTHETIC=15000

eda:
	@echo ">>> NB_01: Exploratory data analysis"
	$(NBCONVERT) notebooks/01_eda.ipynb

threshold:
	@echo ">>> NB_04: Threshold optimisation, tau*(T,d), profit curve"
	$(NBCONVERT) notebooks/04_threshold_optimization.ipynb

interpret:
	@echo ">>> NB_05: SHAP, permutation importance, failure modes"
	$(NBCONVERT) notebooks/05_interpretation_shap.ipynb

transfer:
	@echo ">>> NB_06: EU transfer validation against EUROCONTROL"
	$(NBCONVERT) notebooks/06_eu_transfer_validation.ipynb

evaluate: threshold interpret transfer

all: smoke train eda evaluate report

report:
	$(NBCONVERT) notebooks/99_final_pipeline.ipynb
	jupyter nbconvert --to html notebooks/99_final_pipeline.ipynb --output-dir reports/build/

# ---------- Maintenance ------------------------------------------------------

test:
	pytest tests/ -v

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf data/interim/* data/processed/* data/cache/*
	rm -rf reports/build/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
