#!/usr/bin/env python3
"""The surgical vocabulary and the Eq. (12) prefix, on real sentences.

    python examples/surgical_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from surgellm.features import SurgicalFeatureExtractor
from surgellm.vocab import default_vocabulary

SENTENCES = [
    ("D1", "A genuinely superb, captivating film — one of the year's great "
           "surprises!"),
    ("D1", "Dull, mediocre and painful to sit through. I was disappointed."),
    ("D2", "[Q] Which river is longer? [CTX] According to the article, as "
           "stated in the passage, the supporting context gives 6,650 km."),
    ("D3", "Firstly, the results show a clear trend. Moreover, the data "
           "demonstrate consistency. In conclusion, the effect is robust."),
    ("D4", "I think I probably messed this up, maybe? I love it anyway, sort "
           "of. Personally I feel it's fine."),
]


def main() -> int:
    vocab = default_vocabulary()
    ex = SurgicalFeatureExtractor(vocab)
    names = ex.feature_names()

    print(f"Surgical vocabulary: {vocab.n_groups} groups, "
          f"{sum(g.cardinality() for g in vocab.groups)} entries, "
          f"feature dimension {ex.dim}\n")
    print("Groups:", ", ".join(vocab.group_names), "\n")
    print("=" * 78)

    for task, text in SENTENCES:
        s = ex.extract(text)
        fired = [(n, int(v)) for n, v in zip(names[:vocab.n_groups], s) if v > 0]
        print(f"\n[{task}] {text}")
        print(f"   lexical groups  : "
              + (", ".join(f"{n}={v}" for n, v in fired) or "(none fired)"))
        print(f"   surface         : "
              + ", ".join(f"{n}={int(v) if n != 'mean_word_length' else round(float(v), 1)}"
                          for n, v in zip(names[vocab.n_groups:], s[vocab.n_groups:])))
        print(f"   Eq. (12) prefix : {ex.prefix_string(task, s)}")

    print("\n" + "=" * 78)
    print("Note the prefix carries integer counts floor(s_j), not standardised")
    print("values: that is what makes it readable. The gate receives the")
    print("standardised vector instead (Eq. 8 or Eq. 11).")

    # Prefix matching
    print("\nPrefix matching (entries ending in *):")
    for form in ("captivate", "captivating", "captivated", "capture"):
        hit = int(ex.extract(form)[vocab.group_names.index("sst_pos")])
        print(f"   {form:14s} -> sst_pos = {hit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
