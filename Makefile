# SURGeLLM reproduction targets.
PY      ?= python3
THREADS ?= 2
SCALE   ?= 0.20
SEEDS   ?= 0 1 2

.PHONY: help install test test-fast smoke data main ablations theory sig \
        failures multilingual figures latex crosswalk all clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:       ## install the package and demo dependencies
	$(PY) -m pip install -e .

test-fast:     ## unit tests only (no training)
	$(PY) -m pytest tests -q -m "not slow"

test:          ## full test suite, integration tests included
	$(PY) -m pytest tests -q

smoke:         ## one short training run, end to end
	$(PY) scripts/01_train.py --variant surgellm-iwn --backbone roberta \
	  --mode demo --scale 0.05 --epochs 2 --threads $(THREADS)

data:          ## Table 2, feature stats, contamination diagnostics
	$(PY) scripts/00_prepare_data.py --scale $(SCALE)

main:          ## Tables 3, 5, 6, 12, 14 (33 runs)
	$(PY) scripts/02_main_table.py --scale $(SCALE) --seeds $(SEEDS) --threads $(THREADS)

ablations:     ## Tables 7-11
	$(PY) scripts/03_ablations.py --scale $(SCALE) --threads $(THREADS)

theory:        ## Table 15, Theorem 1, Lemma 4, Proposition 2, Corollary 3
	$(PY) scripts/04_theory.py --scale $(SCALE) --threads $(THREADS)

sig:           ## Table 4 and bootstrap CIs
	$(PY) scripts/05_significance.py

failures:      ## §7.5 failure-mode analysis
	$(PY) scripts/07_failures.py --scale $(SCALE) --threads $(THREADS)

multilingual:  ## Table 16 (Appendix J)
	$(PY) scripts/08_multilingual.py --threads $(THREADS)

figures:       ## Figures 1-6
	$(PY) scripts/09_figures.py --source run
	$(PY) scripts/09_figures.py --source paper

latex:         ## LaTeX fragments for every table
	$(PY) scripts/10_latex_tables.py --source run

crosswalk:     ## claim-by-claim verdicts
	$(PY) scripts/11_crosswalk.py

all:           ## the whole pipeline
	bash scripts/run_all.sh

clean:         ## remove generated artifacts
	rm -rf results/tables/*.csv results/tables/latex results/figures/* \
	       results/logs/*.json checkpoints/*.pt checkpoints/*.json .cache
