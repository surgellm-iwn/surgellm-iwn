# Claim → code → artifact crosswalk

Every claim the paper makes, where it is implemented, and which generated
file carries the evidence. `scripts/11_crosswalk.py` turns the headline rows
into `results/tables/claim_crosswalk.csv` with a verdict per claim.

---

## 1. Equations

| Paper | Statement | Implementation | Test |
|---|---|---|---|
| Eq. (1) | multi-task empirical risk, summed per-task CE, `w_k = 1` | `models/heads.py::MultiTaskHeads.loss` | `test_model.py::test_per_task_losses_are_summed` |
| Eq. (2) | `h = E_φ(x)_[0]`, the `[CLS]` embedding | `models/backbone.py::*.encode` | `test_model.py::test_every_demo_backbone_builds_and_differs` |
| Eq. (3) | `h̃ = h + α E_{t_k}`, `α = 0.1` | `models/gate.py::TaskEmbedding` | `test_model.py::test_task_embedding_is_a_small_residual_perturbation` |
| Eq. (4) | `s_j = Σ_{w∈v_j} 1[w ∈ x̃]`, prefix matching | `features.py::SurgicalFeatureExtractor.extract` | `test_features_and_vocab.py` (5 tests) |
| Eq. (5) | `s' = ReLU(W_s s + b_s)` | `models/gate.py::SurgicalFeatureGate.project` | `test_model.py::test_feature_projection_is_nonnegative` |
| Eq. (6) | `g = σ(W_g [h̃; s'] + b_g)`, per-dimension | `models/gate.py::SurgicalFeatureGate.forward` | `test_model.py::test_gate_output_is_in_the_open_unit_interval_per_dimension` |
| Eq. (7) | `ĥ = LN(g ⊙ h̃ + (1−g) ⊙ s')` | same | `test_model.py::test_fusion_is_a_convex_combination_before_layernorm` |
| Eq. (8) | marginal standardisation | `normalization.py` mode `standard` | `test_normalization.py::test_standard_moments_are_the_marginal_ones` |
| Eq. (9) | `s̄_k → s̄_{c*,k}` as `π_{c*} → 1` | `normalization.py::prior_contamination` | `test_normalization.py::test_marginal_mean_approaches_majority_mean_as_skew_grows` |
| Eq. (10) | class-balanced moments | `normalization.py` mode `iwn` | `test_normalization.py::test_balanced_moments_are_the_unweighted_class_averages` |
| Eq. (11) | IWN standardisation | same | `test_normalization.py::test_iwn_centres_the_minority_class_closer_to_zero` |
| Eq. (12) | `[TASK:t_k | F1:v1 | … | F16:v16] ⊕ x` | `features.py::prefix_string`, `train.py::prepare_tensors` | `test_features_and_vocab.py::test_prefix_string_format_matches_eq12`, `test_training_pipeline.py::test_prefix_changes_the_token_sequence` |
| Eqs. (13)–(14) | two-layer GELU head, dropout 0.1 / 0.05 | `models/heads.py::TaskHead` | `test_model.py::test_heads_route_by_task_mask_and_mask_invalid_classes` |
| Eq. (16) | the excess-risk bound | `theory/bound.py::evaluate_bound` | `test_theory.py` (4 tests) |

## 2. Theory

| Paper | Statement | Implementation | Artifact |
|---|---|---|---|
| Definition 1 | surgical feature alignment `ρ_k` | `theory/alignment.py::estimate_rho` | `table15_rho_estimates.csv` |
| Lemma 4 | `L_θ ≤ L_φ · L_G · L_head` | `theory/bound.py::estimate_lipschitz` | `lemma4_lipschitz.csv` |
| Theorem 1 | gate approximation bound | `theory/bound.py::evaluate_bound` | `theorem1_bound.csv` |
| Proposition 2 | degeneracy at `ρ_k = 0`, four conclusions | `theory/degeneracy.py` | `proposition2_degeneracy.csv`, `proposition2_decay_trace.csv` |
| Corollary 3 | the gate cannot hurt at `ρ_k = 0` | `theory/degeneracy.py` | `corollary3_safety.csv` |
| §A.3 | the imbalance loophole | `theory/degeneracy.py::imbalance_loophole_check` | `imbalance_loophole.csv` |
| App. C.1–C.3 | proofs | docstrings in `theory/bound.py`, `theory/degeneracy.py` | — |
| App. G | Monte Carlo `ρ_k` on 1,000 validation examples | `theory/alignment.py` (`MC_EXAMPLES = 1000`) | `table15_rho_estimates.csv` |

### What "verified" means here

Only **Proposition 2** is verified in the strong sense: its hypothesis
(`ρ_k = 0`) can be constructed by feeding the gate label-independent
features, and all four conclusions are then measured before and after
training under weight decay. `scripts/04_theory.py` does exactly that.

**Theorem 1** bounds `R(f_θ) − R(f*)`, and neither the Bayes risk `R(f*)`
nor the oracle `s*` is observable. The code therefore: computes every
computable term, substitutes a conservative proxy (`R(f*) = 0`, which only
makes the check harder to pass), approximates `s*` by ridge regression of
the score direction onto the feature space, and reports the slack. It also
checks the two monotonicities the theorem predicts — `1/√N_k` in the
generalisation term, `λ_max(W_gᵀW_g)` in the approximation term — which are
falsifiable. The paper itself notes the bound is standard Rademacher
complexity and may be loose (Limitations, §8); the reported slack quantifies
that.

**Lemma 4** is checked by comparing the composed spectral bound against the
largest finite-difference ratio observed over random example pairs. A
measured ratio above the bound would falsify it.

## 3. Algorithm 1

| Lines | Step | Implementation |
|---|---|---|
| 1 | 70/15/15 stratified split | `data/splits.py::stratified_split` |
| 2–7 | IWN ? class-balanced moments : marginal | `train.py::_fit_normalizers` |
| 8 | pre-tokenise, chunk-cache (C = 2048) | `data/cache.py::tokenize_with_cache` |
| 9 | construct `f_θ`, AdamW, scheduler | `train.py::Trainer._build_optimizer` |
| 10 | DDP + FP16 prepare | `train.py::Trainer` (`device`/`fp16`) |
| 11 | `F1* ← −∞`, `p ← 0`, `θ* ← θ` | `train.py::Trainer.fit` |
| 14–17 | mini-batch, `s(x)`, standardise, prefix, forward | `train.py::prepare_tensors` + `Trainer._train_epoch` |
| 18–22 | `backward(ℓ/τ)`, clip 1.0, step every τ | `train.py::Trainer._train_epoch` |
| 24 | QuickVal | `train.py::Trainer.evaluate` |
| 25–30 | early stopping, patience `P = 2` | `train.py::Trainer.fit` |
| 33–34 | restore `θ*`, evaluate on test | `train.py::Trainer.fit` |

Tested by `test_training_pipeline.py`:
`test_early_stopping_respects_the_patience_budget`,
`test_best_checkpoint_is_restored`,
`test_per_task_sampler_yields_homogeneous_batches`.

## 4. Tables and figures

| Paper | Generated by | File |
|---|---|---|
| Table 1 variants | `models/surgellm.py::VARIANT_TABLE` | `configs/variants.yaml` |
| Table 2 corpora | `scripts/00_prepare_data.py` | `table2_corpus_statistics.csv` |
| Table 3 main results | `scripts/02_main_table.py` | `table3_main_results.csv` |
| Table 4 significance | `scripts/05_significance.py` | `table4_significance.csv` |
| Table 5 IWN ablation | `scripts/02_main_table.py` | `table5_iwn_ablation.csv` |
| Table 6 training dynamics | `scripts/02_main_table.py` | `table6_training_dynamics.csv` |
| Table 7 components | `scripts/03_ablations.py` | `table7_component_ablation.csv` |
| Table 8 group count | `scripts/03_ablations.py` | `table8_group_count.csv` |
| Table 9 vocabulary control | `scripts/03_ablations.py` | `table9_vocab_control.csv` |
| Table 10 surface features | `scripts/03_ablations.py` | `table10_surface_ablation.csv` |
| Table 11 leave-one-out | `scripts/03_ablations.py` | `table11_leave_one_out.csv` |
| Table 12 efficiency | `scripts/02_main_table.py` | `table12_efficiency.csv` |
| Table 13 hyperparameters | `experiment.py::HPARAMS_TABLE13` | `configs/hparams_table13.yaml` |
| Table 14 per-seed | `scripts/02_main_table.py` | `table14_per_seed.csv` |
| Table 15 `ρ_k` | `scripts/04_theory.py` | `table15_rho_estimates.csv` |
| Table 16 multilingual | `scripts/08_multilingual.py` | `table16_multilingual.csv` |
| Table 17 crosswalk | `paper_reference.py` | `claim_crosswalk.csv` |
| Figures 1–6 | `scripts/09_figures.py` | `results/figures/{run,paper}/*.png` |
| §7.5 failure analysis | `scripts/07_failures.py` | `failure_modes.csv`, `failure_cases.csv` |
| App. D vocabulary | `vocab.py` | `configs/surgical_vocabulary.json` |
| App. E auto-extraction | `autoextract.py` | `autoextract_report.json` |
| App. H complexity | `models/surgellm.py::parameter_report` | `parameter_report.csv` |
| App. I reproducibility | `utils/seeding.py` | `REPRODUCIBILITY.md` |

## 5. Headline claims

`scripts/11_crosswalk.py` scores these. Verdicts are `supported`
(direction and mechanism reproduce), `shape_only` (an absolute paper number
that a demo run cannot speak to), `not_supported`, or `not_tested`.

| ID | Claim | Paper | Evidence |
|---|---|---|---|
| C1 | IWN-RoBERTa reaches aggregate macro-F1 0.940 | 0.940 | `table3_main_results.csv` |
| C2 | +0.036 over the strongest non-IWN variant | +0.036 | `table3_main_results.csv` |
| C3 | +0.130 on authorship detection (D4) | +0.130 | `table5_iwn_ablation.csv` |
| C4 | the gate *without* IWN regresses on D4 | −0.051 | `table7_component_ablation.csv` |
| C5 | random vocabulary costs 0.028 avg F1 | −0.028 | `table9_vocab_control.csv` |
| C6 | auto-extraction recovers 99.5% of curated | 0.995 | `table9_vocab_control.csv` |
| C7 | removing surface features costs 0.011 on D4 | −0.011 | `table10_surface_ablation.csv` |
| C8 | `ρ_4` rises 0.61 → 2.13 after IWN | 2.13 | `table15_rho_estimates.csv` |
| C9 | shared parameters exceed 99% | >0.99 | `parameter_report.csv` |
| C10 | the gate degenerates to identity at `ρ_k = 0` | — | `proposition2_degeneracy.csv` |
| C11 | surgical overhead ≈ 1.2M params, ~1% of RoBERTa | ~0.01 | `parameter_report.csv` |
| C12 | each task has a dominant indicator group | — | `table11_leave_one_out.csv` |

### Claims a demo run cannot settle

- **C1** and any other absolute macro-F1 on SST-2 / HotPotQA / LLM-7 /
  HumLLM. The surrogate corpora have their own difficulty; a number from
  them is not comparable to 0.940.
- **C9** at the stated threshold. The offline stand-in encoders are two
  orders of magnitude smaller than RoBERTa-base, so their task-specific
  heads are a proportionally larger share. The >99% figure needs a real
  125M-parameter encoder; the demo reports ~97–99%.
- **Table 12 wall-clock and Pareto frontier.** Parameter counts in that
  table are the paper's; the timings are the demo backbones'. The rows are
  labelled accordingly.
- **Table 16.** Appendix J's +0.009 on ALLOCINE and GERMANSENTIMENT needs
  those corpora and XLM-R-base. The demo runs the *recipe* on
  French- and German-flavoured surrogates, which tests that auto-extraction
  needs no per-language curation — the appendix's actual argument — not the
  magnitude.

## 6. Known gaps and deviations

1. **LLM-7 and HumLLM identifiers.** Both citations are marked
   "Author-check required" in the paper with no canonical URL/DOI.
   `configs/datasets.yaml` leaves them blank and the loader raises a clear
   error rather than guessing a dataset.
2. **T5 is an encoder here.** The paper treats T5-base as a text-to-text
   reference. This implementation mean-pools T5's encoder states and uses
   the same classification heads, so the comparison is "same framework,
   larger encoder" rather than full seq2seq decoding. §6.4's argument is
   about parameter efficiency on classification, which this preserves, but
   it is a deviation.
3. **`L_φ` is a spectral-norm product proxy.** An exact Lipschitz constant
   for a transformer is intractable; this is the standard surrogate and the
   same one the Rademacher argument of Appendix C.1 leans on.
4. **`s*` is a ridge-regression surrogate** for the oracle feature vector,
   not the oracle itself. `theorem1_bound.csv` reports an *estimate* of the
   approximation term.
5. **The demo profile replaces Table 13's per-variant schedules** with a
   uniform one, because at demo scale a per-variant learning rate would
   confound the component ablation. Paper mode uses Table 13 verbatim.
6. **FP16 / multi-GPU paths are present but untested here** — the
   development machine has no GPU. `Trainer` accepts `device` and `fp16`;
   Accelerate integration is left as the thin wrapper Algorithm 1 line 10
   describes.
7. **Table 4's exact p-values depend on the three-seed sample.** With
   `n = 3` per group, Welch's test has ~2–4 degrees of freedom; the paper's
   `< 0.001` entries reflect very large effect sizes on D4, and the
   implementation reproduces the test, not the specific p-value.

8. **Proposition 2, conclusion 3 (`g*_i → 1`) does not reproduce as
   stated.** This one is worth reading carefully, because it is a finding
   about the paper rather than a limitation of the harness.

   Conclusions 1 and 2 reproduce strongly and are the load-bearing ones:
   with `ρ_k = 0` by construction, `‖W_s‖` falls by roughly 45× and
   `‖s'(x)‖` by roughly 46× over 150 steps, and the measured `ρ_k` goes to
   exactly 0. The proof's mechanism — zero expected gradient plus weight
   decay gives pure exponential decay — is confirmed in closed form to
   within `1e-5` (`proposition2_decay_trace.csv`).

   Conclusion 3 does not follow. Weight decay shrinks `W_g` alongside
   `W_s`, so the gate pre-activation decays and `g → σ(0) = 0.5`, not 1.
   The proof sketch says "to minimize loss, the gate routes all signals
   through `h̃`, forcing `g_i → 1`", but once `s' = 0` the fusion is
   `LN(g ⊙ h̃)` for *any* `g`, and the head can adapt to any fixed
   per-dimension scaling. There is no loss gradient pushing `g` upward;
   `g` is unconstrained, not driven to 1.

   This does not damage the paper's argument. Conclusion 4 — that `ĥ`
   carries no surgical-feature signal — follows from `s' → 0` alone, and
   Corollary 3 (the gate cannot increase the minimum achievable risk at
   `ρ_k = 0`) likewise rests on conclusion 2, not on conclusion 3. The
   safety property stands; the route to it in the proof sketch is
   imprecise about `g`.

   `check_proposition2` therefore reports `core_conclusions_1_2_pass`
   separately from the aggregate directional check, and
   `examples/degeneracy.py` prints the distinction. If conclusion 3 matters
   to a revision, excluding `W_g` from weight decay is the change that
   would make it recoverable, and that should then be stated as a
   hypothesis of the proposition.
