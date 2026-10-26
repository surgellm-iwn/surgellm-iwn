# Usage

## Install

```bash
pip install -e .                      # demo mode: torch, numpy, pandas, matplotlib
pip install -e ".[paper]"             # adds transformers, datasets, accelerate, SBERT
pip install -e ".[dev]"               # pytest, pyyaml
```

Python ≥ 3.9. No GPU required for demo mode.

## Train one model

```bash
python scripts/01_train.py \
    --variant surgellm-iwn \          # baseline | t5 | surgellm-g | surgellm-s
                                      # | surgellm-full | surgellm-iwn
    --backbone roberta \              # distilbert | bert | roberta | albert | t5 | xlmr
    --mode demo \                     # demo | paper
    --scale 0.2 \                     # fraction of each corpus (demo only)
    --seed 0 \
    --threads 2
```

Prints per-task macro-F1, accuracy, ROC-AUC and per-class precision/recall,
plus the parameter report; writes `checkpoints/<run_id>.pt` and a JSON
sidecar with the full config, metrics and determinism report.

## Programmatic use

```python
from surgellm.experiment import RunSpec, run_single

art = run_single(RunSpec(
    variant="surgellm-iwn", backbone="roberta",
    offline_backbone=True, profile="demo", scale=0.2, seed=0,
))

art.avg_f1                                  # aggregate macro-F1
art.task_f1()                               # {"D1": .., "D2": .., ...}
art.result.test_metrics["D4"]["recall_llm"] # the per-class number IWN fixes
art.normalizers["D4"].mode                  # "iwn"
art.contamination["D4"]                     # Eq. (9) diagnostics
```

### Just the feature pipeline

```python
from surgellm.features import SurgicalFeatureExtractor
from surgellm.normalization import FeatureNormalizer

ex = SurgicalFeatureExtractor()                    # Appendix D vocabulary
s  = ex.extract_batch(texts)                       # Eq. (4), (N, 16)

iwn = FeatureNormalizer("iwn").fit(s, labels, n_classes=2)   # Eq. (10)
s_tilde = iwn.transform(s)                                   # Eq. (11)

ex.prefix_string("D4", s[0])     # Eq. (12): "[TASK:D4|F1:2|F2:0|...]"
```

`FeatureNormalizer` is a drop-in for any pipeline with a skewed corpus and
handcrafted features — nothing about it is specific to this model.

### A custom vocabulary

```python
from surgellm.vocab import SurgicalVocabulary

mine = SurgicalVocabulary.from_dict({
    "legal_hedge": ["allegedly", "purportedly", "ostensibly"],
    "citation":    ["see also", "cf.", "ibid"],
}, name="legal")

ex = SurgicalFeatureExtractor(mine)      # dim = 2 groups + 6 surface = 8
```

Pass it to `run_single(spec, vocabulary=mine)`; `feature_dim` follows
automatically.

## Reproduce the paper's tables

```bash
bash scripts/run_all.sh                       # demo, ~2-3 h on 2 cores
SCALE=0.05 bash scripts/run_all.sh            # fast pass, ~20 min
MODE=paper SEEDS="0 1 2" bash scripts/run_all.sh
```

Or one stage at a time (`make help` lists them):

```bash
make data ablations theory sig figures latex crosswalk
```

Stage dependencies: `05` and `03`'s Table 7 both read
`results/logs/per_seed_raw.json`, so run `02` first. Everything else is
independent.

## Paper mode

1. Install the extras: `pip install -e ".[paper]"`.
2. Fill in `tasks.D3.hf.dataset` and `tasks.D4.hf.dataset` in
   `configs/datasets.yaml`. The paper's citations for LLM-7 and HumLLM are
   marked "Author-check required", so this repository does not guess them;
   the loader raises a descriptive error if they are blank.
3. Run with `--mode paper --device cuda`.

D1 (SST-2) and D2 (HotPotQA) resolve to public HuggingFace datasets and
need no configuration.

## Reading the output

```
results/
  tables/              one CSV per paper table, plus the theory checks
    latex/run/*.tex    booktabs fragments to \input
    latex/paper/*.tex  the same layout with the published numbers
    claim_crosswalk.csv
  figures/run/*.png    Figures 1-6 from this run
  figures/paper/*.png  Figures 1-6 from the published values
  logs/                per-stage logs, per_seed_raw.json, run sidecars
checkpoints/           <run_id>.pt + <run_id>.json
```

Every CSV row carries a `source` column: `run` for numbers this repository
produced, `paper` for values transcribed from the publication. They are
never merged into one column without that tag.

## Interpreting a demo run

Start with `claim_crosswalk.csv`. Verdicts mean:

- **`supported`** — the run reproduces the claim's direction and mechanism.
- **`shape_only`** — an absolute number from the real corpora that a demo
  run cannot speak to.
- **`not_supported`** — the run contradicts the direction. Worth
  investigating; it may be a bug here or a limit of the surrogate corpora.
- **`not_tested`** — the artifact is missing; run the stage that makes it.

A demo run can show that IWN removes an imbalance pathology, that the gate
degenerates safely at zero alignment, and that curated vocabulary beats a
random control. It cannot show that macro-F1 on HumLLM is 0.892.

## Common adjustments

| Want | How |
|---|---|
| faster | `--scale 0.05`, or `--seeds 0` |
| less noise | `--scale 0.5`, more seeds |
| only some models | `scripts/02_main_table.py --models "Baseline-RoBERTa" "SURGeLLM-IWN-RoBERTa"` |
| skip an ablation | `scripts/03_ablations.py --skip table11` |
| more MC samples for ρ_k | `scripts/04_theory.py --mc-examples 1000` |
| a different FDR level | `scripts/05_significance.py --q 0.10` |
