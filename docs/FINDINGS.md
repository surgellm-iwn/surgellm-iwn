# Findings from building this

Implementing a paper equation by equation tends to surface things a reading
does not. Four observations, ordered by how much they matter.

---

## 1. Proposition 2's conclusion 3 does not follow from its own proof

**The claim.** If `ρ_k = 0`, then at any local minimum of the regularised
loss with weight decay `λ > 0`: (1) `‖W_s‖ → 0`, (2) `s'(x) → 0`,
(3) `g*_i → 1`, (4) `ĥ → LN(h̃)`.

**What reproduces.** Conclusions 1 and 2, strongly. Constructing the
hypothesis — feeding the gate label-independent features — and training
under weight decay gives `‖W_s‖` falling ~45× and `‖s'(x)‖` ~46× over 150
steps, with the measured `ρ_k` reaching exactly 0. The closed-form step
`W_s ← (1 − λη)W_s` matches the observed trajectory to within `1e-5` over
400 steps (`proposition2_decay_trace.csv`).

**What does not.** Conclusion 3. Weight decay shrinks `W_g` as well as
`W_s`, so the gate pre-activation decays toward zero and `g → σ(0) = 0.5`,
not 1. Measured: `g` moves from 0.478 to 0.500.

The proof sketch says *"to minimize loss, the gate routes all signals
through `h̃`, forcing `g_i → 1`"*. But once `s' = 0`, the fusion is

```
ĥ = LN(g ⊙ h̃ + (1 − g) ⊙ 0) = LN(g ⊙ h̃)
```

for **any** `g`, and the downstream head can absorb any fixed
per-dimension scaling. There is no loss gradient pushing `g` upward; `g` is
*unconstrained*, not driven to 1.

**Why it doesn't matter much.** Conclusion 4 — that `ĥ` carries no
surgical-feature signal — follows from `s' → 0` alone. So does Corollary 3,
the safety property that actually gets used ("adding the gate cannot
increase the minimum achievable empirical risk"). The result survives; the
route to it is imprecise about `g`.

**What a revision could do.** Either drop conclusion 3, or add "weight
decay applied to `W_s` only" as a hypothesis and verify `g → 1` under it.
Either way the statement becomes checkable, which it currently is not.

See `examples/degeneracy.py` and `CLAIMS.md` §6.8.

---

## 2. IWN's effect is not about linear separability, and showing that is useful

A converged linear probe on the surgical features gets **identical**
macro-F1 under Eq. (8) and Eq. (11) — exactly as it must, since an affine
model absorbs an affine change of input. `examples/iwn_mechanism.py`
reports this deliberately as a control.

The effect lives downstream, where the gate's sigmoid and a finite
optimisation budget make the input's location and scale matter. Measured at
initialisation on a 9.3:1 pool: the ratio of minority to majority
`‖s'(x)‖` entering the gate is 1.23 under Eq. (8) and 0.97 under Eq. (11).
The gate receives a class-asymmetric input under marginal standardisation
and a balanced one under IWN.

This is worth stating in the paper. §3.5 currently motivates IWN through
the gate "finding it harder to discriminate minority instances", which is
right but reads as if it were about the features' information content. It
is not — the information is identical. It is about where the features land
relative to a nonlinearity.

---

## 3. The D4 "9.3:1 skew" only bites because normalisation uses the full pool

D4 is capped to 50/50, so a reader may reasonably ask how an imbalance
remedy can help a balanced task. The answer is one sentence in §4:
*"feature normalization uses the full training data"*. The normaliser is
fitted on the 788,922-text source corpus at 9.3:1, not on the 5,000-example
balanced cap.

This is load-bearing and easy to miss. The implementation makes it
structural rather than incidental: `TaskData` carries a `normalizer_pool`
separately from `splits`, and `_fit_normalizers` prefers the pool whenever
it is larger. `scripts/00_prepare_data.py` prints the realised skew of each
pool so the reader can check it.

If a future version fitted the normaliser on the balanced cap instead, IWN
would reduce to standard normalisation on D4 and the headline +0.130 would
vanish. Worth a sentence in §3.5 rather than only §4.

---

## 4. Smaller points

**Eq. (4) is ambiguous between an indicator sum and a term frequency.**
`s_j = Σ_{w ∈ v_j} 1[w ∈ x̃]` sums over *vocabulary entries*, so repeating
a term cannot increase the count — `"great great great"` scores 1 for
`sst_pos`. Reading it as a term frequency would give 3 and change every
downstream number. The implementation takes the literal reading and tests
it (`test_count_is_an_indicator_per_entry_not_a_term_frequency`).

**T5-base is an encoder-decoder, and §6.4's comparison is not quite
like-for-like.** The paper treats it as a text-to-text reference, but the
four tasks are classification and the reported metric is macro-F1 from
per-task heads. This implementation mean-pools T5's encoder states into the
same head machinery, which preserves §6.4's actual argument (parameter
efficiency on classification) but is a deviation worth stating.

**Two corpora lack canonical identifiers.** LLM-7 and HumLLM are both
cited with "Author-check required" and no URL or DOI. A reference
implementation cannot guess them, so `configs/datasets.yaml` leaves them
blank and the loader raises a descriptive error. This blocks independent
reproduction of D3 and D4 — the two tasks carrying the paper's headline
result — and is the single highest-value thing to fix before camera-ready.

**Table 4's `< 0.001` entries come from `n = 3`.** Welch's test on three
seeds per group has roughly 2–4 degrees of freedom. The D4 effect sizes are
large enough that the conclusion is safe, but reporting `p < 0.001` from
three paired observations invites a reviewer question. The bootstrap CIs in
`bootstrap_ci.csv` are the more defensible summary, and the paper already
reports them in the revised abstract.
