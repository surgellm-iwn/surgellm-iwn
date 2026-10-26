#!/usr/bin/env python3
"""Step 8 - preliminary multilingual experiment (Appendix J, Table 16).

The paper probes cross-lingual transfer of the Appendix E auto-extraction
recipe on French (ALLOCINE) and German (GERMANSENTIMENT) sentiment corpora
with XLM-R-base, capping at 5,000 training examples, evaluating on the
official test splits with three seeds, and reports +0.009 F1 in both
languages.

Two paths:

``--mode paper``  loads the real corpora and XLM-R-base.
``--mode demo``   generates French- and German-flavoured surrogate corpora
                  whose indicator vocabulary is in the target language, and
                  runs the full auto-extraction recipe on them.  This tests
                  the thing the appendix is actually about - that the
                  recipe needs no per-language manual curation - without
                  requiring either download.

Emits ``table16_multilingual.csv`` and ``multilingual_vocabularies.json``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from surgellm.autoextract import auto_extract_vocabulary
from surgellm.data.schema import Corpus, TaskSpec
from surgellm.data.splits import stratified_split
from surgellm.experiment import RunSpec, run_single
from surgellm.utils.io import ensure_dir, write_csv, write_json
from surgellm.vocab import SurgicalVocabulary

#: Target-language polarity lexicons for the demo corpora.  Deliberately
#: *not* translations of the English ``sst_pos`` / ``sst_neg`` groups: the
#: point of Appendix J is that the recipe discovers them on its own.
LANG_LEXICON: Dict[str, Dict[str, List[str]]] = {
    "french": {
        "pos": ["excellent", "magnifique", "formidable", "superbe", "captivant",
                "merveilleux", "brillant", "remarquable", "passionnant", "genial"],
        "neg": ["horrible", "ennuyeux", "mediocre", "decevant", "affreux",
                "nul", "penible", "insipide", "rate", "lamentable"],
        "filler": ["film", "scene", "acteur", "histoire", "realisateur", "image",
                   "moment", "salle", "public", "musique", "rythme", "dialogue",
                   "lumiere", "ville", "jour", "soir", "livre", "page"],
    },
    "german": {
        "pos": ["hervorragend", "wunderbar", "grossartig", "fesselnd", "brillant",
                "ausgezeichnet", "bemerkenswert", "spannend", "gelungen", "stark"],
        "neg": ["schrecklich", "langweilig", "mittelmaessig", "enttaeuschend",
                "furchtbar", "misslungen", "zaeh", "fade", "schwach", "aergerlich"],
        "filler": ["film", "szene", "schauspieler", "geschichte", "regisseur",
                   "bild", "moment", "saal", "publikum", "musik", "dialog",
                   "licht", "stadt", "tag", "abend", "buch", "seite", "kamera"],
    },
}


def _make_corpus(language: str, n: int = 5000, seed: int = 0,
                 confusion: float = 0.06) -> Corpus:
    lex = LANG_LEXICON[language]
    rng = np.random.default_rng(abs(hash((language, seed))) % (2**32))
    labels = rng.integers(0, 2, size=n)
    texts: List[str] = []
    for y in labels:
        c = 1 - y if rng.random() < confusion else y
        strong = lex["pos"] if c == 1 else lex["neg"]
        weak = lex["neg"] if c == 1 else lex["pos"]
        toks: List[str] = []
        toks += rng.choice(strong, size=int(rng.poisson(2.2)) or 1, replace=True).tolist()
        toks += rng.choice(weak, size=int(rng.poisson(0.7)), replace=True).tolist()
        toks += rng.choice(lex["filler"],
                           size=int(np.clip(rng.lognormal(3.0, 0.45), 8, 120)),
                           replace=True).tolist()
        rng.shuffle(toks)
        texts.append(" ".join(toks) + ".")
    return Corpus(f"{language[:2].upper()}1", texts, labels.astype(np.int64), 2)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="demo", choices=["demo", "paper"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n", type=int, default=2500)
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    if args.mode == "paper":
        print("[08] paper mode needs ALLOCINE / GERMANSENTIMENT and XLM-R-base; "
              "configure them in configs/datasets.yaml and rerun.")
        return 1

    out = ensure_dir(Path(args.out) / "tables")
    logs = ensure_dir(Path(args.out) / "logs")

    from surgellm.data.registry import TaskData
    from surgellm.data.schema import TASKS

    rows: List[Dict] = []
    vocab_dump: Dict[str, Dict] = {}
    scores: Dict[str, Dict[str, float]] = {"baseline": {}, "auto": {}}

    for language in ("french", "german"):
        corpus = _make_corpus(language, n=args.n, seed=0)
        splits = stratified_split(corpus, seed=0)

        report = auto_extract_vocabulary(
            {"D1": splits.train.texts}, {"D1": splits.train.labels},
            {"D1": 2}, seed=0, use_sbert=True,
        )
        vocab_dump[language] = {
            "backend": report.embedding_backend,
            "n_candidates": report.n_candidates,
            "cluster_sizes": report.cluster_sizes,
            "groups": report.vocabulary.to_dict(),
        }
        print(f"[08] {language}: {report.n_candidates} auto-extracted candidates "
              f"({report.embedding_backend})")

        # Reuse the four-task machinery with a single task slot.
        td = TaskData("D1", splits, splits.train, "synthetic")
        tasks = {"D1": td}
        orig = TASKS["D1"]
        TASKS["D1"] = TaskSpec("D1", language, 2, len(corpus), 50.0,
                               language.capitalize(), 1.0, ("negative", "positive"))
        try:
            for label, variant, vocab in (
                ("XLM-R-base baseline", "baseline", SurgicalVocabulary([], "none")),
                ("SURGeLLM-G-XLM-R-base (auto)", "surgellm-g", report.vocabulary),
            ):
                vals = []
                for seed in args.seeds:
                    art = run_single(RunSpec(
                        variant=variant, backbone="xlmr", seed=seed,
                        provider="synthetic", offline_backbone=True, scale=1.0,
                        profile="demo", tag=f"{language}-{variant}",
                        n_classes_override=[2],
                    ), tasks=tasks, vocabulary=vocab, verbose=False)
                    vals.append(float(art.result.test_metrics["D1"]["macro_f1"]))
                key = "baseline" if variant == "baseline" else "auto"
                scores[key][language] = float(np.mean(vals))
                print(f"     {label:<32s} {np.mean(vals):.4f} "
                      f"(sd {np.std(vals):.4f}, {len(vals)} seeds)")
        finally:
            TASKS["D1"] = orig

    for label, key in (("XLM-R-base baseline", "baseline"),
                       ("SURGeLLM-G-XLM-R-base (auto)", "auto")):
        rows.append({"config": label, "source": "run",
                     "french": round(scores[key]["french"], 4),
                     "german": round(scores[key]["german"], 4)})
    rows.append({
        "config": "Delta", "source": "run",
        "french": round(scores["auto"]["french"] - scores["baseline"]["french"], 4),
        "german": round(scores["auto"]["german"] - scores["baseline"]["german"], 4),
    })

    write_csv(rows, out / "table16_multilingual.csv")
    write_json(vocab_dump, logs / "multilingual_vocabularies.json")
    print(f"[08] -> {out / 'table16_multilingual.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
