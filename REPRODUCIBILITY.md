# Reproducibility checklist

Mirrors Appendix I of the paper, and records what this repository can and
cannot establish on its own.

## Code

Released under MIT. Entry points: `make smoke`, `make all`,
`bash scripts/run_all.sh`. The paper lists a project webpage at
`https://surgellm-iwn.github.io`.

## Data

| Task | Corpus | Status |
|---|---|---|
| D1 | SST-2 (GLUE) | public; `glue/sst2` |
| D2 | HotPotQA | public; `hotpot_qa/distractor`, validation split |
| D3 | LLM-7 | **identifier pending** — the paper's citation is marked "Author-check required" |
| D4 | HumLLM | **identifier pending** — same |

Set `tasks.D3.hf.dataset` and `tasks.D4.hf.dataset` in
`configs/datasets.yaml` before running `--mode paper` on those two tasks.
The loader raises a descriptive error rather than substituting something
else.

Preprocessing is reproduced in code: stratified 70/15/15 splits, label
reindexing, training-only computation of `(s̄, σ)` with balanced variants
for IWN, pre-tokenisation and chunked caching at 2,048.

## Random seeds

Seeds `{0, 1, 2}` throughout. `surgellm/utils/seeding.py::seed_everything`
seeds Python, NumPy and torch, and sets `cudnn.deterministic = True`,
`cudnn.benchmark = False` and `CUBLAS_WORKSPACE_CONFIG`. Data splits,
weight initialisation and dropout masks all derive from the run seed.

`tests/test_training_pipeline.py::test_same_seed_reproduces_the_same_score`
asserts bit-level reproducibility of a full run on one machine.

## Software versions

Paper: PyTorch 2.1, Transformers 4.35, Accelerate 0.24, scikit-learn 1.3,
sentence-transformers 2.2.

This implementation requires only `torch`, `numpy`, `pandas` and
`matplotlib` for demo mode. `scipy` and `scikit-learn` are optional:
the t-distribution tail falls back to a continued-fraction incomplete beta,
k-means and PPMI+SVD embeddings are implemented in NumPy, and ROC-AUC is
computed by a tie-aware rank statistic. Every fallback is announced in the
artifact it affects (for example `autoextract_report.json` records
`embedding_backend`).

`surgellm.utils.seeding.determinism_report()` dumps the actual versions and
determinism flags of the machine a run executed on; `scripts/01_train.py`
writes it into each run's JSON sidecar.

## Hardware

Paper: 2× NVIDIA T4 (16 GB), FP16 via Accelerate, effective batch size 32
through accumulation, ~38 GPU-hours to reproduce all main and ablation
results.

Demo mode: CPU only. The reference run in `results/` was produced on 2 CPU
cores in roughly 2–3 hours at `--scale 0.20`.

## Hyperparameters

Table 13 verbatim in `surgellm/experiment.py::HPARAMS_TABLE13` and
`configs/hparams_table13.yaml`:

- AdamW, `λ = 0.01`, `β = (0.9, 0.999)`, `ε = 1e-8`
- linear warmup at 6% of total steps, then linear decay
- gradient clipping at 1.0
- early stopping on validation macro-F1, patience 2
- per-variant learning rates 1.5e-5 – 3e-4, max length 96 or 128

Demo mode substitutes a uniform schedule (`DEMO_HP`) because the offline
backbones are trained from scratch and because a per-variant learning rate
would confound the component ablation. This is a deliberate deviation and
is recorded in `CLAIMS.md` §6.

## Statistical tests

Bootstrap `B = 2,000`, seed 0. Paired Welch t-tests with
Benjamini-Hochberg FDR at `q = 0.05` over the 4 × 4 family. Implemented in
`surgellm/eval/significance.py`; the BH step-up is checked for monotonicity
and against Bonferroni in `tests/test_data_and_eval.py`.

With three seeds per group, Welch's test has roughly 2–4 degrees of
freedom. The tests reproduce the *procedure*; individual p-values are
sample-dependent and will not match the paper digit for digit.

## Compute budget for a full demo pass

| Step | Runs | ~Time (2 CPU cores, `--scale 0.20`) |
|---|---|---|
| 00 data | — | < 1 min |
| 02 main table | 33 | ~55 min |
| 05 significance | 0 | seconds |
| 03 ablations | ~30 | ~45 min |
| 04 theory | 3 + simulations | ~8 min |
| 07 failures | 1 | ~2 min |
| 08 multilingual | 12 short | ~6 min |
| 09–11 figures, LaTeX, crosswalk | 0 | < 1 min |

`SCALE=0.05` cuts this to roughly 20 minutes at the cost of much noisier
estimates.

## Verifying a run

```bash
python scripts/11_crosswalk.py          # claim-by-claim verdicts
make test                               # the full test suite
```

The crosswalk marks absolute-value claims as `shape_only` in demo mode.
That is the honest verdict: a demo run can show that IWN fixes an
imbalance pathology and that the gate degenerates safely, and cannot show
that macro-F1 on HumLLM is 0.892.
