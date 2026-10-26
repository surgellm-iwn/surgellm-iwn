# Output index

Every file the pipeline writes, what is in it, and which script makes it.

## `results/tables/` — the paper's tables

| File | Paper | Script | Key columns |
|---|---|---|---|
| `table2_corpus_statistics.csv` | Table 2 | 00 | `n`, `n_c`, `pct_minority`, `source_pool_skew` |
| `table3_main_results.csv` | Table 3 | 02 | `D1..D4` + `_sd`, `avg_f1`, `delta_vs_baseline_roberta`, `train_s` |
| `table4_significance.csv` | Table 4 | 05 | `comparison`, `task`, `delta`, `p_raw`, `p_bh`, `significant` |
| `table5_iwn_ablation.csv` | Table 5 | 02 | `D1..D4`, `hum_p/r`, `llm_p/r` — the per-class breakdown |
| `table6_training_dynamics.csv` | Table 6 | 02 | `init_loss`, `final_loss`, `best_epoch`, `delta_loss` |
| `table7_component_ablation.csv` | Table 7 | 03 | `P`, `G`, `E`, `I`, per-task F1, `delta`, `delta_D4` |
| `table8_group_count.csv` | Table 8 | 03 | `n_groups` ∈ {0,5,10,15,20} |
| `table9_vocab_control.csv` | Table 9 | 03 | none / random / auto-extracted / curated + deltas |
| `table10_surface_ablation.csv` | Table 10 | 03 | `G + S`, `G only`, `S only` + deltas |
| `table11_leave_one_out.csv` | Table 11 | 03 | per-group F1 drop, `dominant_for` |
| `table12_efficiency.csv` | Table 12 | 02 | `f1_per_min`, `efficiency`, `pareto_efficient` |
| `table14_per_seed.csv` | Table 14 | 02 | `seed0`, `seed1`, `seed2`, `mean`, `sd` |
| `table15_rho_estimates.csv` | Table 15 | 04 | `rho_no_iwn`, `rho_iwn`, `rho_increase`, `f1_gain` |
| `table16_multilingual.csv` | Table 16 | 08 | `french`, `german` |

## `results/tables/` — theory and diagnostics

| File | What it holds |
|---|---|
| `theorem1_bound.csv` | Eq. (16) terms per task: generalisation, approximation, total, excess-risk proxy, `bound_holds`, `slack`, `lambda_max_WgTWg` |
| `lemma4_lipschitz.csv` | `L_encoder`, `L_gate`, `L_head`, the composed bound, the empirical finite-difference constant, and whether the bound holds |
| `proposition2_degeneracy.csv` | the four conclusions measured before and after zero-alignment training, plus pass/fail for each directional check |
| `proposition2_decay_trace.csv` | observed vs predicted `‖W_s‖` under `W_s ← (1−λη)W_s` |
| `corollary3_safety.csv` | whether the gate moved toward the identity map |
| `imbalance_loophole.csv` | §A.3: ρ and F1 rising together on D4 |
| `alignment_vs_gain.csv` | does the gain ordering across tasks track the ρ_k ordering |
| `normalization_contamination.csv` | Eq. (9): majority prior, skew, normalised mean gap, max |z| under each scheme |
| `feature_statistics.csv` | per-dimension mean / sd / nonzero-rate of `s(x)`, per task |
| `parameter_report.csv` | §3.1 shared fraction, Appendix H surgical overhead |
| `bootstrap_ci.csv` | 95% CI per model and whether it overlaps the baseline's |
| `failure_modes.csv` | §7.5 mode counts and fractions per task |
| `failure_cases.csv` | the inspected misclassified examples with their detected modes |
| `claim_crosswalk.csv` | one row per headline claim with a verdict |

## `results/tables/latex/`

`run/*.tex` and `paper/*.tex` — booktabs fragments with identical layout,
the first from this run, the second from the published values. Useful for
checking a fragment renders before swapping numbers in.

## `results/figures/`

`run/` and `paper/`, each with:

| File | Paper |
|---|---|
| `figure1_main_results.png` | Figure 1 — macro-F1 by model × task |
| `figure2_iwn_mechanism.png` | Figure 2 — D4 per-class P/R, and ρ_k vs gain |
| `figure3_component_ablation.png` | Figure 3 — absolute and Δ heatmaps |
| `figure4_efficiency_vocab.png` | Figure 4 — Pareto frontier and vocabulary sensitivity |
| `figure5_training_dynamics.png` | Figure 5 — loss trajectories and convergence epoch |
| `figure6_leave_one_out.png` | Figure 6 — per-group F1 drop heatmap |

Each figure is watermarked bottom-right with its source.

## `results/logs/`

| File | Contents |
|---|---|
| `per_seed_raw.json` | every per-seed score; the input to step 05 and Table 7 |
| `autoextract_report.json` | Appendix E: candidate count, embedding backend, cluster sizes, discovered groups, top terms per class |
| `leave_one_out_reference.json` | the full-vocabulary reference and the dominant group per task |
| `04_theory.json` | decay error, Proposition 2 checks, alignment-gain correlation |
| `failure_summary.json` | §7.5 rollup |
| `multilingual_vocabularies.json` | the auto-extracted French and German groups |
| `run_<run_id>.json` | per-run sidecar: spec, metrics, determinism report |
| `00..11.log`, `pipeline.log` | per-stage stdout |

## `checkpoints/`

`<variant>-<backbone>-seed<N>.pt` — weights, the full `SurgeLLMConfig`, and
the run's metrics in one payload, plus a `.json` sidecar with the metrics
alone. Reload with:

```python
from surgellm.train import load_checkpoint
model, result = load_checkpoint("checkpoints/surgellm-iwn-roberta-seed0.pt")
```

## The `source` column

Every generated CSV carries it:

- `source=run` — produced by this repository on this machine.
- `source=paper` — transcribed from TrustNLP 2026 pp. 600–617.

They are never mixed into one column without the tag. `paper_reference.py`
is the only place published numbers live.
