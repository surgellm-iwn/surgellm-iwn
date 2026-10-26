# SURGeLLM — reference implementation

Code for **"SURGeLLM: Rethinking Multi-Task Evaluation through Task-Aware
Feature Gating with Class-Balanced Normalization"**, Noor Islam S. Mohammad
and Uluğ Bayazıt, *Proceedings of the 6th Workshop on Trustworthy NLP
(TrustNLP 2026)*, pages 600–617.

Every equation, theorem, algorithm, table and figure in the paper has a
corresponding module, script and generated artifact here. The map is in
[`CLAIMS.md`](CLAIMS.md).

---

## Read this first: two run modes

| | `--mode paper` | `--mode demo` (default) |
|---|---|---|
| Encoders | BERT / RoBERTa / DistilBERT / ALBERT / T5 from HuggingFace | small transformers trained from scratch |
| Corpora | SST-2, HotPotQA, LLM-7, HumLLM | seeded surrogate corpora |
| Needs | network, `transformers`, `datasets`, ~38 GPU-hours (T4) | nothing but `torch`, ~2 h on 2 CPU cores |
| Establishes | the paper's numbers | the **mechanism and direction** of each claim |

**A demo run does not reproduce the paper's numbers and this repository never
claims it does.** Demo artifacts are tagged `source=run`; the published values
live in `surgellm/paper_reference.py` tagged `source=paper`, and
`scripts/11_crosswalk.py` reports which claims the run supports, contradicts,
or cannot speak to.

The demo mode exists because most of what the paper argues is structural —
IWN removes a prior-contamination pathology, the gate degenerates safely at
zero alignment, random vocabulary loses to curated vocabulary — and structure
can be tested on corpora whose generative process you control. The surrogate
corpora are built to carry exactly the properties the claims depend on
(§ [Surrogate corpora](#surrogate-corpora)).

---

## Quick start

```bash
pip install -e .                 # torch, numpy, pandas, matplotlib
make smoke                       # one training run, ~1 minute
make test-fast                   # unit tests, no training
```

Three short examples, no downloads, seconds each:

```bash
python examples/iwn_mechanism.py      # what IWN changes, with no model at all
python examples/surgical_features.py  # the vocabulary firing, and the Eq. (12) prefix
python examples/degeneracy.py         # Proposition 2, constructed and measured
```

Full pipeline:

```bash
bash scripts/run_all.sh          # demo, ~2-3 h on 2 cores
SCALE=0.05 bash scripts/run_all.sh   # fast pass, ~20 min
MODE=paper bash scripts/run_all.sh   # the real thing (GPU)
```

Single configuration:

```bash
python scripts/01_train.py --variant surgellm-iwn --backbone roberta \
    --mode demo --scale 0.2 --seed 0
```

---

## Layout

```
surgellm/
  vocab.py            Appendix D: the ten indicator groups, prefix matching
  features.py         Eq. (4): s(x) in R^16, computed on untruncated text
  normalization.py    Eqs. (8)-(11): standard vs Instance-Weighted Normalization
  autoextract.py      Appendix E: Monroe log-odds + SBERT k-means transfer recipe
  train.py            Algorithm 1, line by line
  experiment.py       Table 13 schedules; one entry point for every experiment
  paper_reference.py  the published numbers, transcribed and tagged
  models/
    backbone.py       §3.2: HF encoders, plus offline stand-ins
    gate.py           Eqs. (3), (5)-(7): task embedding and the surgical gate
    heads.py          Eqs. (13)-(14): per-task MLP heads, task-mask routing
    surgellm.py       the assembled model and the Table 1 variants
  data/
    schema.py         Table 2 as data
    splits.py         stratified capping and 70/15/15 splits
    registry.py       HuggingFace loaders and the synthetic provider
    synthetic.py      the surrogate corpora
    cache.py          Alg. 1 line 8: chunked pre-tokenisation; per-task sampler
  eval/
    metrics.py        macro-F1, per-class P/R, rank-based ROC-AUC
    significance.py   Welch t-test, Benjamini-Hochberg, bootstrap CI
    failure_analysis.py  §7.5 failure modes; Table 12 efficiency formulas
  theory/
    alignment.py      Definition 1: Monte Carlo rho_k (Appendix G)
    bound.py          Theorem 1 and Lemma 4
    degeneracy.py     Proposition 2, Corollary 3, the §A.3 imbalance loophole
scripts/ 00..11       one script per stage; all write CSVs
tests/                pytest suite, one test per stated property
configs/              base, datasets, Table 1 variants, Table 13 hyperparameters
```

---

## What each script produces

| Script | Artifacts | Paper |
|---|---|---|
| `00_prepare_data.py` | `table2_corpus_statistics.csv`, `feature_statistics.csv`, `normalization_contamination.csv` | Table 2, §3.3, Eq. (9) |
| `01_train.py` | a checkpoint + JSON sidecar | Algorithm 1 |
| `02_main_table.py` | Tables 3, 5, 6, 12, 14; `parameter_report.csv` | §6, §7.4 |
| `03_ablations.py` | Tables 7, 8, 9, 10, 11; `autoextract_report.json` | §7.1–7.2, App. E |
| `04_theory.py` | Table 15, `theorem1_bound.csv`, `lemma4_lipschitz.csv`, `proposition2_*.csv`, `corollary3_safety.csv`, `imbalance_loophole.csv` | App. A, C, G |
| `05_significance.py` | Table 4, `bootstrap_ci.csv` | §6.2 |
| `07_failures.py` | `failure_modes.csv`, `failure_cases.csv` | §7.5 |
| `08_multilingual.py` | Table 16 | Appendix J |
| `09_figures.py` | Figures 1–6 (`--source run` or `--source paper`) | Figs. 1–6 |
| `10_latex_tables.py` | `results/tables/latex/*.tex` | all tables |
| `11_crosswalk.py` | `claim_crosswalk.csv` | the whole paper |

---

## The model in code

```python
h       = backbone.encode(ids, mask)              # Eq. (2)  [CLS]
h_tilde = h + 0.1 * task_embedding[t_k]           # Eq. (3)
s_proj  = relu(W_s @ s + b_s)                     # Eq. (5)
g       = sigmoid(W_g @ [h_tilde ; s_proj] + b_g) # Eq. (6)  per-dimension
h_hat   = LayerNorm(g * h_tilde + (1-g) * s_proj) # Eq. (7)
logits  = W_2 @ gelu(W_1 @ h_hat)                 # Eqs. (13)-(14)
```

with `s` standardised beforehand by Eq. (8) or, for IWN, Eq. (11), and the
Eq. (12) prefix `[TASK:t_k|F1:v1|...|F16:v16]` prepended to the text before
tokenisation.

### Why IWN is parameter-free

Normalisation is a *data-side* transform fitted once on the training
partition. `FeatureNormalizer.n_parameters == 0`, and `transform()` never
takes a label — the two properties the paper calls "parameter-free" and
"test-time class-agnostic", each asserted in `tests/test_normalization.py`.

---

## Surrogate corpora

`surgellm/data/synthetic.py` generates four corpora whose class-conditional
structure matches what each claim needs:

- **D1** polarity words on both sides of mixed reviews — moderate alignment,
  and a non-trivial error floor. (The paper replaced its original D1 precisely
  because a saturated task makes every comparison unreadable, Appendix L.3;
  recreating that flaw here would be self-defeating.)
- **D2** attribution phrases that are rare in questions and frequent in
  context — the cleanest signal, and the highest ρ_k.
- **D3** prompt-scaffolding markers on longer texts, 11.8:1 source skew.
- **D4** the hard case: a 9.3:1 source pool, overlapping indicator rates, and
  heteroscedastic length. Human text almost never fires the LLM indicators, so
  at that skew the pooled σ_k of Eq. (8) is a poor description of either
  class, and the marginal mean sits on top of the majority. Normalisation
  statistics are fitted on the **full skewed pool**, not the 50/50 cap —
  exactly as §4 describes — which is why IWN still matters after balancing.

`scripts/00_prepare_data.py` prints the realised skew and the Eq. (9)
contamination gap so you can check these properties rather than trust them.

---

## One finding worth flagging

Writing the theory checks surfaced something about **Proposition 2**.

Conclusions 1 and 2 reproduce strongly — construct `ρ_k = 0` by feeding the
gate label-independent features, and `‖W_s‖` and `‖s'(x)‖` both collapse by
~45×, with the closed-form decay `W_s ← (1−λη)W_s` matching to `1e-5`.

Conclusion 3, `g*_i → 1`, does not. Weight decay shrinks `W_g` alongside
`W_s`, so the gate pre-activation decays and `g → σ(0) = 0.5`. The proof
sketch's step — "to minimize loss, the gate routes all signals through `h̃`,
forcing `g_i → 1`" — needs a loss gradient pushing `g` upward, and once
`s' = 0` there is none: the fusion is `LN(g ⊙ h̃)` for any `g`, and the head
adapts to whichever one it gets. `g` is unconstrained, not driven to 1.

The safety property is unaffected. Conclusion 4 and Corollary 3 both follow
from `s' → 0` alone. But the route to them in the proof sketch is imprecise
about `g`, and a revision could either drop conclusion 3 or add "weight
decay applied to `W_s` only" as a hypothesis. `examples/degeneracy.py`
prints the distinction; `CLAIMS.md` §6.8 has the full argument.

## Reproducibility

Seeds `{0, 1, 2}`; data splits, weight initialisation, dropout masks and CUDA
determinism are all seeded (`surgellm/utils/seeding.py`). Each checkpoint
carries its full config and metrics, so a run can be rebuilt from the file
alone. See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

Two corpora in the paper (LLM-7, HumLLM) are cited with "Author-check
required" and no canonical URL/DOI, so `configs/datasets.yaml` leaves their
identifiers blank rather than guessing. Paper mode on D3/D4 needs them filled
in; the loader says so rather than silently falling back.

---

## Testing

```bash
make test-fast     # ~100 unit tests, no training
make test          # adds the Algorithm 1 integration tests
```

The suite is organised by claim, not by file: `test_normalization.py` checks
the four stated IWN properties, `test_theory.py` checks Definition 1,
Lemma 4, Theorem 1 and Proposition 2, and `test_training_pipeline.py` checks
that early stopping, best-checkpoint restoration and the prefix wiring behave
as Algorithm 1 says.

---

## Documentation

| File | What's in it |
|---|---|
| [`CLAIMS.md`](CLAIMS.md) | every claim → the code that implements it → the artifact that evidences it, plus known gaps and deviations |
| [`docs/EQUATIONS.md`](docs/EQUATIONS.md) | each equation, its statement, and the function that implements it |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | how the pieces fit, and why a few are shaped the way they are |
| [`docs/USAGE.md`](docs/USAGE.md) | CLI, programmatic use, custom vocabularies, reading the output |
| [`docs/OUTPUTS.md`](docs/OUTPUTS.md) | every generated file and its columns |
| [`docs/FINDINGS.md`](docs/FINDINGS.md) | four things implementing the paper surfaced, including the Proposition 2 issue above |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Appendix I checklist, compute budget, what each mode establishes |

## Citation

```bibtex
@inproceedings{mohammad2026surgellm,
  title     = {{SURGeLLM}: Rethinking Multi-Task Evaluation through Task-Aware
               Feature Gating with Class-Balanced Normalization},
  author    = {Mohammad, Noor Islam S. and Bayaz{\i}t, Ulu{\u{g}}},
  booktitle = {Proceedings of the 6th Workshop on Trustworthy NLP (TrustNLP)},
  pages     = {600--617},
  year      = {2026},
  publisher = {Association for Computational Linguistics}
}
```

MIT licensed.
