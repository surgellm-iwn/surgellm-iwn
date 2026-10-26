"""Experiment orchestration: one place that knows how to run a configuration.

Everything downstream (the main table, the ablations, the theory checks,
the efficiency analysis) calls :func:`run_single` or :func:`run_grid`, so a
change to the training recipe cannot silently diverge between experiments.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .data.registry import TaskData, load_all_tasks
from .data.schema import TASK_ORDER, TASKS
from .features import SurgicalFeatureExtractor
from .models.backbone import BACKBONE_REGISTRY, build_backbone
from .models.surgellm import SurgeLLM, SurgeLLMConfig, variant_flags
from .normalization import FeatureNormalizer, prior_contamination
from .train import TrainConfig, Trainer, TrainResult, prepare_tensors, save_checkpoint
from .utils.io import ensure_dir, write_json
from .utils.seeding import seed_everything
from .vocab import SurgicalVocabulary, default_vocabulary

__all__ = ["RunSpec", "RunArtifacts", "run_single", "run_grid",
           "HPARAMS_TABLE13", "hparams_for"]


# ---------------------------------------------------------------------------
# Table 13: full hyperparameter configuration
# ---------------------------------------------------------------------------
HPARAMS_TABLE13: Dict[str, Dict] = {
    # key: "<variant>-<backbone>"            LR      EP  BS  GA  MaxL  WU
    "baseline-distilbert":      dict(lr=2.0e-5, ep=3, bs=32, ga=1, maxl=96,  wu=0.06),
    "baseline-bert":            dict(lr=2.0e-5, ep=3, bs=16, ga=2, maxl=128, wu=0.06),
    "baseline-roberta":         dict(lr=2.0e-5, ep=3, bs=16, ga=2, maxl=128, wu=0.06),
    "baseline-albert":          dict(lr=2.0e-5, ep=3, bs=32, ga=1, maxl=96,  wu=0.06),
    "t5-t5":                    dict(lr=3.0e-4, ep=5, bs=8,  ga=4, maxl=128, wu=0.06),
    "surgellm-s-distilbert":    dict(lr=2.0e-5, ep=4, bs=32, ga=1, maxl=96,  wu=0.06),
    "surgellm-s-bert":          dict(lr=2.0e-5, ep=4, bs=16, ga=2, maxl=128, wu=0.06),
    "surgellm-s-roberta":       dict(lr=2.0e-5, ep=4, bs=16, ga=2, maxl=128, wu=0.06),
    "surgellm-g-roberta":       dict(lr=1.5e-5, ep=4, bs=16, ga=2, maxl=128, wu=0.06),
    "surgellm-g-bert":          dict(lr=1.5e-5, ep=4, bs=16, ga=2, maxl=128, wu=0.06),
    "surgellm-full-roberta":    dict(lr=1.5e-5, ep=5, bs=16, ga=2, maxl=128, wu=0.06),
    "surgellm-full-albert":     dict(lr=2.0e-5, ep=5, bs=32, ga=1, maxl=96,  wu=0.06),
    "surgellm-iwn-roberta":     dict(lr=1.5e-5, ep=5, bs=16, ga=2, maxl=128, wu=0.06),
    "surgellm-iwn-bert":        dict(lr=2.0e-5, ep=5, bs=16, ga=2, maxl=128, wu=0.06),
}

_FALLBACK_HP = dict(lr=2.0e-5, ep=4, bs=16, ga=2, maxl=128, wu=0.06)

#: Uniform schedule for the offline ``tiny`` backbone.
#:
#: The Table 13 learning rates (1.5e-5 - 3e-4) are tuned for *pretrained*
#: encoders being fine-tuned.  The tiny backbone is trained from scratch, so
#: it needs a far larger step size.  Equally important: Table 13 gives each
#: variant a different LR, epoch budget and batch size, which at demo scale
#: confounds a component ablation - a difference between SURGeLLM-FULL and
#: SURGeLLM-IWN could be the schedule rather than IWN.  The demo profile
#: therefore holds everything fixed across variants except the architectural
#: flags, with ``extended`` still spending one extra epoch so the ``E``
#: column of Table 7 remains a real factor.
DEMO_HP = dict(lr=1.0e-3, ep=5, bs=32, ga=1, maxl=128, wu=0.06)


def hparams_for(variant: str, backbone: str, profile: str = "paper") -> Dict:
    """Hyperparameters for a run.

    ``profile="paper"`` -> Table 13 (with a documented fallback for pairs the
    table does not list).  ``profile="demo"`` -> the uniform offline schedule.
    """
    if profile == "demo":
        return dict(DEMO_HP)
    if profile != "paper":
        raise ValueError(f"unknown hyperparameter profile {profile!r}")
    return dict(HPARAMS_TABLE13.get(f"{variant}-{backbone}", _FALLBACK_HP))


@dataclass
class RunSpec:
    """A single (variant, backbone, seed) configuration."""

    variant: str = "surgellm-iwn"
    backbone: str = "roberta"
    seed: int = 0
    provider: str = "synthetic"
    offline_backbone: bool = False
    scale: float = 1.0
    device: str = "cpu"
    max_epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    cache_dir: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    vocabulary_name: str = "curated"
    use_surface_features: bool = True
    use_lexical_groups: bool = True
    tag: str = ""
    n_classes_override: Optional[List[int]] = None
    #: "paper" -> Table 13 schedules; "demo" -> uniform offline schedule.
    profile: str = "paper"


    @property
    def name(self) -> str:
        base = f"{self.variant}-{self.backbone}"
        return f"{base}-{self.tag}" if self.tag else base

    @property
    def run_id(self) -> str:
        return f"{self.name}-seed{self.seed}"

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RunArtifacts:
    """Outputs of one run."""

    spec: RunSpec
    result: TrainResult
    model: SurgeLLM
    tensors: Dict[str, Dict]
    normalizers: Dict[str, FeatureNormalizer]
    tasks: Dict[str, TaskData]
    extractor: SurgicalFeatureExtractor
    contamination: Dict[str, Dict] = field(default_factory=dict)

    @property
    def avg_f1(self) -> float:
        return self.result.avg_test_macro_f1

    def task_f1(self) -> Dict[str, float]:
        return {t: m["macro_f1"] for t, m in self.result.test_metrics.items()}


def _extended_epochs(variant: str, base: int) -> int:
    """The ``E`` column of Table 1 is a training-schedule flag.

    ``extended`` variants get the full epoch budget; the others get one
    fewer, so "extended training" is an actual experimental factor rather
    than a label."""
    return base if variant_flags(variant)["extended"] else max(base - 1, 1)


def run_single(spec: RunSpec,
               tasks: Optional[Dict[str, TaskData]] = None,
               vocabulary: Optional[SurgicalVocabulary] = None,
               verbose: bool = True) -> RunArtifacts:
    """Train and evaluate one configuration end to end."""
    seed_everything(spec.seed)
    flags = variant_flags(spec.variant)
    hp = hparams_for(spec.variant, spec.backbone, profile=spec.profile)

    if tasks is None:
        tasks = load_all_tasks(provider=spec.provider, seed=spec.seed, scale=spec.scale,
                               cache_dir=spec.cache_dir)
    order = [t for t in TASK_ORDER if t in tasks]

    extractor = SurgicalFeatureExtractor(
        vocabulary=vocabulary or default_vocabulary(),
        use_surface=spec.use_surface_features,
        use_groups=spec.use_lexical_groups,
    )

    # In demo mode the *paper* backbone key is kept in the config (so Table 3
    # can report the right parameter count and max_length) while the offline
    # stand-in from DEMO_TINY_CONFIG is what actually gets built.
    model_cfg = SurgeLLMConfig(
        variant=spec.variant, backbone=spec.backbone,
        task_names=order,
        n_classes_per_task=spec.n_classes_override or [TASKS[t].n_classes for t in order],
        feature_dim=extractor.dim,
        offline_backbone=spec.offline_backbone,
        max_length=hp["maxl"],
    )
    model = SurgeLLM(model_cfg)

    # Algorithm 1, lines 2-7.
    from .train import _fit_normalizers
    normalizers = _fit_normalizers(tasks, extractor, use_iwn=bool(flags["iwn"]))
    contamination = {}
    for tid, nrm in normalizers.items():
        try:
            contamination[tid] = prior_contamination(nrm)
        except ValueError:
            pass

    # Algorithm 1, line 8 + Eq. 12.
    tensors = prepare_tensors(
        tasks, model, extractor, normalizers,
        use_prefix=bool(flags["prefix"]), cache_dir=spec.cache_dir,
        task_order=order,
    )

    train_cfg = TrainConfig(
        learning_rate=spec.learning_rate or hp["lr"],
        max_epochs=spec.max_epochs or _extended_epochs(spec.variant, hp["ep"]),
        batch_size=spec.batch_size or hp["bs"],
        grad_accumulation=hp["ga"], max_length=hp["maxl"],
        warmup_fraction=hp["wu"], seed=spec.seed, device=spec.device,
        cache_dir=spec.cache_dir,
    )

    if verbose:
        print(f"[run] {spec.run_id}  flags={flags}  lr={train_cfg.learning_rate} "
              f"epochs={train_cfg.max_epochs} bs={train_cfg.batch_size}", flush=True)

    trainer = Trainer(model, train_cfg, task_order=order)
    ckpt = None
    if spec.checkpoint_dir:
        ckpt = Path(spec.checkpoint_dir) / f"{spec.run_id}.pt"
    result = trainer.fit(tensors, checkpoint_path=ckpt, verbose=verbose)
    result.normalization = {t: {"mode": n.mode, **contamination.get(t, {})}
                            for t, n in normalizers.items()}

    return RunArtifacts(spec=spec, result=result, model=model, tensors=tensors,
                        normalizers=normalizers, tasks=tasks, extractor=extractor,
                        contamination=contamination)


def run_grid(specs: Sequence[RunSpec], shared_tasks: bool = True,
             verbose: bool = True) -> List[RunArtifacts]:
    """Run a list of configurations, reusing loaded corpora per seed."""
    cache: Dict[Tuple, Dict[str, TaskData]] = {}
    out: List[RunArtifacts] = []
    for spec in specs:
        tasks = None
        if shared_tasks:
            key = (spec.provider, spec.seed, spec.scale)
            if key not in cache:
                cache[key] = load_all_tasks(provider=spec.provider, seed=spec.seed,
                                            scale=spec.scale, cache_dir=spec.cache_dir)
            tasks = cache[key]
        out.append(run_single(spec, tasks=tasks, verbose=verbose))
    return out
