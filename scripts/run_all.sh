#!/usr/bin/env bash
# Run the complete SURGeLLM reproduction pipeline.
#
#   bash scripts/run_all.sh                 # demo mode, ~2-3 h on 2 CPU cores
#   MODE=paper bash scripts/run_all.sh      # real encoders + real corpora (GPU)
#   SCALE=0.05 bash scripts/run_all.sh      # fast smoke pass, ~20 min
#
# Every step writes CSVs to results/tables/ and logs to results/logs/.
# Steps are independent apart from the documented dependencies:
#   05 needs 02   (reads per_seed_raw.json)
#   03 table7 needs 02
#   09/10/11 need whatever steps you ran

set -euo pipefail

MODE="${MODE:-demo}"
SCALE="${SCALE:-0.20}"
SEEDS="${SEEDS:-0 1 2}"
THREADS="${THREADS:-2}"
PY="${PY:-python3}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="$ROOT/results/logs"

mkdir -p "$LOGDIR" "$ROOT/results/tables" "$ROOT/results/figures" "$ROOT/checkpoints"
cd "$ROOT"

banner() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }
step()   { banner "$1"; shift; "$@" 2>&1 | tee -a "$LOGDIR/run_all.log"; }

banner "SURGeLLM pipeline  (mode=$MODE scale=$SCALE seeds='$SEEDS')"
date | tee "$LOGDIR/run_all.log"

step "00  corpora, vocabulary, feature statistics" \
  $PY scripts/00_prepare_data.py --mode "$MODE" --scale "$SCALE"

step "02  main multi-seed comparison (Tables 3, 5, 6, 12, 14)" \
  $PY scripts/02_main_table.py --mode "$MODE" --scale "$SCALE" \
      --seeds $SEEDS --threads "$THREADS"

step "05  significance (Table 4) and bootstrap CIs" \
  $PY scripts/05_significance.py

step "03  ablations (Tables 7-11)" \
  $PY scripts/03_ablations.py --mode "$MODE" --scale "$SCALE" --threads "$THREADS"

step "04  theory (Table 15, Theorem 1, Lemma 4, Proposition 2, Corollary 3)" \
  $PY scripts/04_theory.py --mode "$MODE" --scale "$SCALE" --threads "$THREADS"

step "07  failure-mode analysis (§7.5)" \
  $PY scripts/07_failures.py --mode "$MODE" --scale "$SCALE" --threads "$THREADS"

step "08  preliminary multilingual (Table 16)" \
  $PY scripts/08_multilingual.py --mode "$MODE" --threads "$THREADS" || \
  echo "   (skipped: paper mode needs ALLOCINE / GERMANSENTIMENT)"

step "09  figures 1-6" \
  $PY scripts/09_figures.py --source run

step "09b figures from the published values" \
  $PY scripts/09_figures.py --source paper

step "10  LaTeX table fragments" \
  $PY scripts/10_latex_tables.py --source run

step "11  claim crosswalk" \
  $PY scripts/11_crosswalk.py --mode "$MODE"

banner "done"
date | tee -a "$LOGDIR/run_all.log"
echo "tables : $ROOT/results/tables"
echo "figures: $ROOT/results/figures"
echo "logs   : $LOGDIR"
