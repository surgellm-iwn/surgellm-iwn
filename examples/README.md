# Examples

Short, self-contained scripts. None of them needs a download or a GPU.

| Script | Runtime | Shows |
|---|---|---|
| `iwn_mechanism.py` | ~5 s | What IWN actually changes, with no model and no training: prior contamination (Eq. 9), where each class lands after standardisation, a control showing the effect is *not* about linear separability, and the class asymmetry at the gate input. |
| `surgical_features.py` | ~1 s | The Appendix D vocabulary firing on real sentences, and the Eq. (12) prefix that gets prepended to each input. |
| `degeneracy.py` | ~20 s | Proposition 2 end to end: construct `ρ_k = 0`, train under weight decay, watch `‖W_s‖ → 0`, `g → 1`, `ĥ → LN(h̃)`. |

```bash
python examples/iwn_mechanism.py
python examples/surgical_features.py
python examples/degeneracy.py
```

`iwn_mechanism.py` is the one to read first — it is the shortest honest
statement of what the paper's central contribution does, and it is explicit
about what it cannot show (the F1 effect needs training; see
`scripts/04_theory.py`).
