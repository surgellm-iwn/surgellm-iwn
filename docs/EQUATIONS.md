# Equation reference

Each equation in the paper, its statement, and the exact code that
implements it. Line numbers are omitted deliberately — symbol names are
stable, line numbers are not.

---

### Eq. (1) — multi-task empirical risk

```
L(θ) = Σ_k  (w_k / |D_k|) · Σ_{(x,y,t_k) ∈ D_k}  ℓ(f_θ(x, t_k), y)
```

`w_k = 1` throughout; per-task batch sampling supplies the balance.

`models/heads.py::MultiTaskHeads.loss` — routes each row to its head by a
task-integer mask, computes per-task cross-entropy, sums over the tasks
present in the batch.

---

### Eq. (2) — encoder

```
h = E_φ(x)_[0] ∈ R^d
```

`models/backbone.py::HFEncoderBackbone.encode` takes index 0 (`[CLS]` for
the BERT family, `<s>` for RoBERTa). T5 has no `[CLS]`, so its encoder
states are mean-pooled over the attention mask — a documented deviation
(`CLAIMS.md` §6.2).

---

### Eq. (3) — task embedding

```
h̃ = h + α · E_{t_k},     α = 0.1
```

`models/gate.py::TaskEmbedding`. The §3.2 justification is that
`α‖E_{t_k}‖` is small relative to `‖h‖ ≈ √d·σ_h`;
`TaskEmbedding.perturbation_ratio` measures that ratio directly and the
test asserts it stays below 0.25.

---

### Eq. (4) — surgical feature extraction

```
s_j = Σ_{w ∈ v_j}  1[w ∈ x̃],     j = 1 … 10
```

plus six surface features `s_11 … s_16`: word count, mean word length,
sentence count (split on `.!?`), question marks, exclamation marks, and a
digit-presence indicator. Full vector `s(x) ∈ R^16_{≥0}`.

`features.py::SurgicalFeatureExtractor.extract`,
`vocab.py::IndicatorGroup.count`.

Two subtleties the tests pin down:

- The sum is an **indicator per vocabulary entry**, not a term frequency.
  `"great great great"` scores 1 for `sst_pos`, not 3.
- Features are computed on the **untruncated** text. This is the
  truncation-loss argument of §7.2.3: the encoder sees at most
  `L ∈ {96, 128}` tokens, so a 500-word document's word count is
  information the encoder cannot recover.

Prefix entries (`oscillat*`) match token prefixes; multi-word entries
(`according to`) and symbolic entries (`(1)`) match the raw string.

---

### Eq. (5) — feature projection

```
s' = ReLU(W_s s + b_s),     W_s ∈ R^{d×16}
```

`models/gate.py::SurgicalFeatureGate.project`. ReLU places `s'` in the
same orthant as a typical post-LayerNorm activation.

---

### Eq. (6) — gate computation

```
g = σ(W_g [h̃ ; s'] + b_g),     W_g ∈ R^{d×2d},     g ∈ (0,1)^d
```

`models/gate.py::SurgicalFeatureGate.forward`. Sigmoid, not softmax;
per-dimension, not scalar. Both are tested.

---

### Eq. (7) — gated fusion

```
ĥ = LN( g ⊙ h̃ + (1 − g) ⊙ s' )
```

Same function. `test_fusion_is_a_convex_combination_before_layernorm`
reconstructs the pre-LayerNorm combination from the returned `g`, `h̃` and
`s'` and checks it matches.

---

### Eq. (8) — standard standardisation

```
ŝ(x) = ( s(x) − s̄_k ) / ( σ_k + ε )
```

`normalization.py`, `mode="standard"`. Moments are the marginal mean and
standard deviation over the training partition.

---

### Eq. (9) — the class-imbalance pathology

```
s̄_k = Σ_c π_c · s̄_{c,k}  →  s̄_{c*,k}   as  π_{c*} → 1
```

Not a computation but a statement about Eq. (8) under skew.
`normalization.py::prior_contamination` quantifies it: the normalised L2
gap between the marginal mean and the class-balanced mean, plus the
majority prior and the skew ratio.

`test_marginal_mean_approaches_majority_mean_as_skew_grows` constructs
corpora at skew 1, 3, 9.3 and 50 and asserts the gap is monotone.

---

### Eq. (10) — class-balanced moments

```
s̄^bal_k = (1/n_c) Σ_c s̄_{c,k}       σ^bal_k = (1/n_c) Σ_c σ_{c,k}
```

Unweighted averages of per-class moments — note that `σ^bal` averages the
per-class standard deviations rather than pooling them, so it excludes
between-class variance.

`normalization.py::FeatureNormalizer.fit`, `mode="iwn"`.

---

### Eq. (11) — IWN standardisation

```
s̃(x) = ( s(x) − s̄^bal_k ) / ( σ^bal_k + ε )
```

Same class, `transform`. Label-free by construction.

The four claimed properties, each a test in `tests/test_normalization.py`:

| Property | Test |
|---|---|
| parameter-free | `test_iwn_introduces_no_learnable_parameters` |
| test-time class-agnostic | `test_transform_requires_no_labels` |
| reduces to Eq. (8) when balanced | `test_iwn_matches_standard_on_a_balanced_corpus` |
| compositional with loss/sampling remedies | structural: nothing here touches the loss |

---

### Eq. (12) — task-conditioned prefix tokens

```
x' = [TASK:t_k | F1:v1 | … | F16:v16] ⊕ x,      v_j = ⌊s_j⌋
```

`features.py::prefix_string` builds the string;
`train.py::prepare_tensors` concatenates and tokenises it jointly with the
text so its representations are co-attended to by every layer (§3.6).

Unlike soft prompts (Lester et al., 2021) or prefix tuning (Li and Liang,
2021), this prefix is a deterministic, interpretable string — you can read
it.

---

### Eqs. (13)–(14) — task-specific heads

```
u_k    = GELU( W_{1,k} ĥ + b_{1,k} ),      W_{1,k} ∈ R^{(d/2)×d}
ŷ_k    = softmax( W_{2,k} u_k + b_{2,k} ), W_{2,k} ∈ R^{n_{c,k}×(d/2)}
```

Dropout 0.1 before `W_1`, 0.05 before `W_2`.
`models/heads.py::TaskHead`. The module returns **logits**; the softmax is
folded into the cross-entropy for numerical stability, which is
mathematically identical and avoids a log of a near-zero probability.

---

### Eq. (15) — Definition 1, surgical feature alignment

```
ρ_k = E_{(x,y) ~ P_{t_k}} [ | ⟨ s'(x), ∇_ĥ log p(y | ĥ) ⟩ | ]
```

`theory/alignment.py::estimate_rho`. Monte Carlo on 1,000 validation
examples per task (Appendix G). For each example the score-function
gradient is obtained by backpropagating `log p(y | ĥ)` to `ĥ`.

`ρ_k = 0` iff `s'` is orthogonal in expectation to the score function —
the hypothesis of Proposition 2. The test
`test_rho_is_zero_when_the_projection_is_zero` zeroes `W_s` and confirms
the estimator returns exactly 0.

---

### Eq. (16) — Theorem 1, excess-risk bound

```
R(f_θ) − R(f*)  ≤   C / √N_k   +   λ_max(W_gᵀ W_g)/2 · ‖s' − s*‖²
                    └ generalization ┘   └──── approximation ────┘
```

with `C = O(L_φ · L_head · ρ · √log(1/δ))`.

`theory/bound.py::evaluate_bound`. See `CLAIMS.md` §2 for what is and is
not checkable; in short, the two monotonicities (`1/√N_k`,
`λ_max(W_gᵀW_g)`) are falsifiable and are tested, and the bound is checked
against a conservative excess-risk proxy.

The constant `sup_z σ'(z) = 1/4` used when propagating through the
bilinear gate is `theory/bound.py::SIGMOID_SUP_DERIVATIVE`, and
`test_sigmoid_sup_derivative_is_one_quarter` verifies it numerically
rather than taking it on faith.

---

### Eq. (17)–(18) — Rademacher complexity (Appendix C.2)

```
R̂_N(F) ≤ L_θ · rad(X) / √N_k
R(f_θ) − R̂(f_θ) ≤ 2 R̂_N(F) + O( √(log(1/δ)/N_k) ) ≤ C/√N_k
```

Folded into `evaluate_bound`'s generalisation term. `L_θ` comes from
Lemma 4 via `estimate_lipschitz`.

---

### Eq. (19) — Proposition 2, proof step

```
E[∇_{W_s} L] = E[∇_{s'} L] · sᵀ = 0      when ρ_k = 0
```

so under SGD with weight decay the update reduces to pure exponential
decay `W_s ← (1 − λη) W_s`.

`theory/degeneracy.py::run_decay_simulation` drives an optimiser with a
zero data gradient and compares the observed norm trajectory against the
closed-form geometric decay; the test requires agreement to `< 1e-5`.

The four conclusions — `‖W_s‖ → 0`, `s'(x) → 0`, `g*_i → 1`,
`ĥ → LN(h̃)` — are measured by
`theory/degeneracy.py::degeneracy_diagnostics` before and after training
on label-independent features, and scored by `check_proposition2`.
