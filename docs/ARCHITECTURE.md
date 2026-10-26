# Architecture notes

How the pieces fit, and why a few of them are shaped the way they are.

## Data flow for one example

```
raw text x
   │
   ├─► SurgicalFeatureExtractor.extract(x)          Eq. (4)
   │        s ∈ R^16_{≥0}, computed on the FULL text
   │        │
   │        ├─► FeatureNormalizer.transform(s)      Eq. (8) or (11)
   │        │        s̃  ──────────────────────────────┐
   │        │                                         │
   │        └─► prefix_string("D1", s)               Eq. (12)
   │                 "[TASK:D1|F1:3|...|F16:1]"       │
   │                        │                         │
   └────────────────────────┴── concat ──► tokenizer  │
                                              │       │
                                        input_ids     │
                                              │       │
                                   backbone.encode    │  Eq. (2)
                                              h       │
                                   + α·E[t_k]         │  Eq. (3)
                                              h̃       │
                                              └───────┤
                                                      ▼
                                      SurgicalFeatureGate   Eqs. (5)–(7)
                                                      ĥ
                                                      ▼
                                      MultiTaskHeads[t_k]   Eqs. (13)–(14)
                                                   logits
```

Two things are worth noticing in that diagram.

**The raw `s` feeds the prefix; the standardised `s̃` feeds the gate.**
Eq. (12) defines `v_j = ⌊s_j⌋` — integer *counts*, which is what makes the
prefix interpretable. Standardised values would be fractional and
corpus-dependent. The gate, by contrast, needs the standardised vector,
because the whole IWN argument is about the scale and centre of its input.

**Normalisation sits outside the model.** `FeatureNormalizer` is fitted
once on the training partition and then applied as a fixed affine map. That
is what makes IWN parameter-free (nothing is learned) and test-time
class-agnostic (`transform` never sees a label). Putting it inside the
module would have made both properties harder to state and impossible to
assert in a test.

## Where the normaliser is fitted

§4 says feature normalisation uses the full training data. For D3 and D4
the full training data is the *skewed source pool*, not the balanced cap —
which is the only reason IWN can still help a 50/50-capped D4. `TaskData`
therefore carries two corpora:

```python
TaskData(
    splits          = SplitBundle(train, val, test),   # the Table 2 subset
    normalizer_pool = Corpus(...),                      # the uncapped source
)
```

and `train.py::_fit_normalizers` prefers the pool whenever it is larger.
`scripts/00_prepare_data.py` prints the realised skew of each pool so this
is checkable, not assumed.

## The gate's three design choices

Each is a claim in §3.4 and each has a test.

| Choice | Alternative rejected | Why |
|---|---|---|
| sigmoid | softmax over dimensions | a softmax imposes a unit budget across `d`; modality fusion is dimension-wise, not competitive |
| per-dimension `g ∈ (0,1)^d` | scalar gate | a scalar forces every hidden unit to use the same modality mix |
| LayerNorm after fusion | no normalisation | keeps `ĥ` in the same statistical regime as the unfused encoder output |

`SurgicalFeatureGate` also exposes `spectral_norm_Wg()`,
`lipschitz_constant()` and `gate_statistics()` — not for the forward pass,
but because Theorem 1, Lemma 4 and Proposition 2 need exactly those
quantities. Theory hooks live on the module that owns the weights.

## Variants

`VARIANT_TABLE` is Table 1 as data. Three of the four flags are
architectural and resolve inside `SurgeLLMConfig.__post_init__`:

- `prefix` → whether `prepare_tensors` prepends the Eq. (12) string
- `gate` → `SurgicalFeatureGate.enabled`
- `iwn` → which moments `_fit_normalizers` computes

The fourth, `extended`, is a *schedule* flag, so it lives in the training
config: `_extended_epochs` gives extended variants the full epoch budget
and the others one fewer. Keeping it out of the module matters, because
otherwise "extended training" would be a label rather than a factor, and
the `E` column of Table 7 would measure nothing.

## Backbones

```
EncoderBackbone                      interface: tokenize, encode, hidden_size
├── HFEncoderBackbone                AutoModel + AutoTokenizer  (paper mode)
└── TinyEncoderBackbone              hash tokenizer + small pre-LN encoder
```

`build_backbone` never silently substitutes the tiny model for a failed
HuggingFace download: a "paper-faithful" run that quietly was not one is a
worse failure than a crash.

In demo mode each paper backbone maps to a *distinct* tiny configuration
(`DEMO_TINY_CONFIG`) whose capacity ordering mirrors the real one. Without
that, all eleven rows of Table 3 would be the same model and the
across-model significance tests would compare a sample with itself.

## Multi-task batching

`PerTaskBatchSampler` yields homogeneous-task batches in shuffled order.
§3.1 uses `w_k = 1` and relies on per-task batch sampling for balance, so
the sampler is doing load-balancing work that the loss is not. Two
properties are tested: every batch has one task, and every example appears
exactly once per epoch.

Heads mask invalid class slots to `−inf` rather than zero, so an argmax can
never select a class the task does not have, even if a head's logits are
poorly scaled early in training.

## Caching

`tokenize_with_cache` chunks at 2,048 (Algorithm 1 line 8) and keys the
on-disk cache on tokenizer identity, max length, prefix presence and a hash
of the texts. The prefix is part of the key because the same corpus
tokenises differently with and without it — a cache that ignored this would
silently serve prefix-free tensors to a prefix variant, and the ablation
would show the prefix doing nothing.

## Theory module boundaries

```
theory/alignment.py   Definition 1          needs: s', ∇_ĥ log p(y|ĥ)
theory/bound.py       Lemma 4, Theorem 1    needs: spectral norms, an s* surrogate
theory/degeneracy.py  Prop. 2, Cor. 3       needs: ‖W_s‖, ‖s'‖, mean g, ‖ĥ − LN(h̃)‖
```

All three read the model; none of them modify it, apart from the
degeneracy *experiment* in `scripts/04_theory.py`, which deliberately
trains a model on label-independent features to construct the `ρ_k = 0`
hypothesis. That is the one place a theorem's premise can be manufactured,
and so the one theorem that is genuinely verified rather than measured.
