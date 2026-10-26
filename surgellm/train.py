"""Algorithm 1: SURGeLLM multi-GPU training (with optional IWN).

Line-by-line correspondence with the paper's pseudocode:

===== ==========================================================  ==================
Line  Algorithm 1 step                                            Implementation
===== ==========================================================  ==================
1     stratify D_k into 70/15/15                                  ``splits.stratified_split``
2-7   IWN ? class-balanced (sbar^bal, sigma^bal) : marginal       ``_fit_normalizers``
8     pre-tokenize, chunk-cache (C = 2048)                        ``cache.tokenize_with_cache``
9     construct f_theta; AdamW; scheduler gamma                   ``_build_optimizer``
10    accelerator.prepare (DDP + FP16)                            ``Trainer._prepare``
11    F1* <- -inf; p <- 0; theta* <- theta                        ``Trainer.fit``
12-13 epoch loop; train mode                                      ``Trainer.fit``
14-15 mini-batch; s(x_i) (Eq. 4); standardize (Eq. 8 or 11)       precomputed in cache
16    build prefix x'_i (Eq. 12)                                  precomputed in cache
17    forward, per-task CE (Eq. 1)                                ``SurgeLLM.forward``
18    backward(loss / tau)                                        ``Trainer._step``
19-22 every tau steps: clip 1.0, opt.step, sched.step, zero_grad  ``Trainer._step``
24    QuickVal on the validation loader                           ``Trainer.evaluate``
25-30 early stopping with patience P = 2                          ``Trainer.fit``
33-34 restore theta*, evaluate on test                            ``Trainer.fit``
===== ==========================================================  ==================

Features are standardised *before* tokenisation caching because both the
gate input and the Eq. (12) prefix values derive from ``s(x)``; this keeps
each example's prefix consistent with the gate it feeds.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data.cache import (CHUNK_SIZE, MultiTaskDataset, PerTaskBatchSampler,
                         TokenizedSplit, collate, tokenize_with_cache)
from .data.registry import TaskData
from .data.schema import TASKS
from .eval.metrics import aggregate_metrics, classification_metrics
from .features import SurgicalFeatureExtractor
from .models.surgellm import SurgeLLM, SurgeLLMConfig
from .normalization import FeatureNormalizer
from .utils.seeding import seed_everything

__all__ = ["TrainConfig", "Trainer", "TrainResult", "prepare_tensors"]


@dataclass
class TrainConfig:
    """Hyperparameters.  Defaults follow Table 13 for the IWN variants."""

    learning_rate: float = 1.5e-5
    max_epochs: int = 5
    batch_size: int = 16              # per-GPU
    grad_accumulation: int = 2        # tau
    max_length: int = 128
    warmup_fraction: float = 0.06
    weight_decay: float = 0.01        # AdamW lambda
    adam_betas: Tuple[float, float] = (0.9, 0.999)
    adam_eps: float = 1e-8
    max_grad_norm: float = 1.0
    patience: int = 2                 # P
    seed: int = 0
    device: str = "cpu"
    fp16: bool = False
    num_workers: int = 0
    chunk_size: int = CHUNK_SIZE
    cache_dir: Optional[str] = None
    log_every: int = 50

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TrainResult:
    """Everything a run emits, serialisable to JSON."""

    variant: str
    backbone: str
    seed: int
    best_epoch: int
    best_val_macro_f1: float
    initial_loss: float
    final_loss: float
    loss_reduction: float
    early_stopped: bool
    wall_clock_s: float
    epochs_run: int
    test_metrics: Dict[str, Dict] = field(default_factory=dict)
    val_metrics: Dict[str, Dict] = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)
    parameter_report: Dict[str, float] = field(default_factory=dict)
    normalization: Dict[str, Dict] = field(default_factory=dict)
    config: Dict = field(default_factory=dict)

    @property
    def avg_test_macro_f1(self) -> float:
        return aggregate_metrics(self.test_metrics, "macro_f1")

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["avg_test_macro_f1"] = self.avg_test_macro_f1
        return d


# ---------------------------------------------------------------------------
# Data preparation: Algorithm 1 lines 1-8
# ---------------------------------------------------------------------------
def _fit_normalizers(tasks: Dict[str, TaskData], extractor: SurgicalFeatureExtractor,
                     use_iwn: bool) -> Dict[str, FeatureNormalizer]:
    """Algorithm 1 lines 2-7.

    Fitted on the training partition only.  For tasks whose *source* pool
    is skewed (D3, D4), the pool - not the balanced cap - is used, matching
    §4: "feature normalization uses the full training data, and since
    P(s | y) differs in moments across classes, IWN corrects residual
    imbalance effects."
    """
    norms: Dict[str, FeatureNormalizer] = {}
    for tid, td in tasks.items():
        pool = td.normalizer_pool if len(td.normalizer_pool) >= len(td.splits.train) \
            else td.splits.train
        s = extractor.extract_batch(pool.texts)
        norm = FeatureNormalizer(mode="iwn" if use_iwn else "standard")
        norm.fit(s, pool.labels, n_classes=pool.n_classes)
        norms[tid] = norm
    return norms


def prepare_tensors(tasks: Dict[str, TaskData], model: SurgeLLM,
                    extractor: SurgicalFeatureExtractor,
                    normalizers: Dict[str, FeatureNormalizer],
                    use_prefix: bool, cache_dir: Optional[str],
                    chunk_size: int = CHUNK_SIZE,
                    task_order: Optional[Sequence[str]] = None
                    ) -> Dict[str, Dict[str, TokenizedSplit]]:
    """Algorithm 1 lines 8, 15-16, for every split.

    Returns ``{"train"|"val"|"test": {task_id: TokenizedSplit}}``.
    """
    order = list(task_order or tasks.keys())
    out: Dict[str, Dict[str, TokenizedSplit]] = {"train": {}, "val": {}, "test": {}}
    tag = f"{model.config.backbone}-{'pfx' if use_prefix else 'nopfx'}"

    for tid in order:
        td = tasks[tid]
        t_index = order.index(tid)
        for split_name in ("train", "val", "test"):
            corpus = getattr(td.splits, split_name)
            s_raw = extractor.extract_batch(corpus.texts)            # Eq. (4)
            s_std = normalizers[tid].transform(s_raw)                # Eq. (8) or (11)

            prefixes = None
            if use_prefix:
                # Eq. (12) uses the *integer counts* floor(s_j), i.e. the raw
                # feature values, not the standardised ones.
                prefixes = [extractor.prefix_string(tid, s_raw[i])
                            for i in range(len(corpus.texts))]

            enc = tokenize_with_cache(
                model.backbone, corpus.texts, prefixes=prefixes,
                cache_dir=cache_dir, tokenizer_tag=f"{tag}-{tid}-{split_name}",
                chunk_size=chunk_size,
            )
            out[split_name][tid] = TokenizedSplit(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                features=torch.tensor(s_std, dtype=torch.float32),
                labels=torch.tensor(corpus.labels, dtype=torch.long),
                task_ids=torch.full((len(corpus),), t_index, dtype=torch.long),
                texts=list(corpus.texts),
            )
    return out


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class Trainer:
    """Executes Algorithm 1 for one (variant, backbone, seed)."""

    def __init__(self, model: SurgeLLM, config: TrainConfig,
                 task_order: Sequence[str]):
        self.model = model
        self.cfg = config
        self.task_order = list(task_order)
        self.device = torch.device(config.device)
        self.model.to(self.device)

    # -- line 9-10 --------------------------------------------------------
    def _build_optimizer(self, n_update_steps: int):
        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if (p.ndim == 1 or name.endswith(".bias")) else decay).append(p)
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": self.cfg.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=self.cfg.learning_rate, betas=self.cfg.adam_betas, eps=self.cfg.adam_eps,
        )
        warmup = max(int(self.cfg.warmup_fraction * n_update_steps), 1)

        def lr_lambda(step: int) -> float:
            if step < warmup:                                   # linear warmup (6%)
                return step / max(warmup, 1)
            prog = (step - warmup) / max(n_update_steps - warmup, 1)
            return max(0.0, 1.0 - prog)                         # linear decay
        return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # -- lines 14-22 ------------------------------------------------------
    def _train_epoch(self, loader: DataLoader, optimizer, scheduler,
                     scaler: Optional["torch.cuda.amp.GradScaler"]) -> Dict[str, float]:
        self.model.train()
        tau = max(int(self.cfg.grad_accumulation), 1)
        losses: List[float] = []
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = self.model(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                task_ids=batch["task_ids"], s=batch["features"], labels=batch["labels"],
            )
            loss = out["loss"]
            losses.append(float(loss.detach()))
            (loss / tau).backward()                                   # line 18

            if (step + 1) % tau == 0:                                 # line 19
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.cfg.max_grad_norm)      # line 20
                optimizer.step()                                      # line 21
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        # Flush a trailing partial accumulation window.
        if len(loader) % tau != 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        return {"mean_loss": float(np.mean(losses)) if losses else float("nan"),
                "n_batches": len(losses)}

    # -- line 24 / line 33 -------------------------------------------------
    @torch.no_grad()
    def evaluate(self, splits: Dict[str, TokenizedSplit],
                 batch_size: int = 64) -> Dict[str, Dict]:
        """Per-task metrics on a set of tokenised splits (QuickVal / test)."""
        self.model.eval()
        results: Dict[str, Dict] = {}
        for tid, sp in splits.items():
            if len(sp) == 0:
                continue
            preds, probs = [], []
            for i in range(0, len(sp), batch_size):
                out = self.model(
                    input_ids=sp.input_ids[i:i + batch_size].to(self.device),
                    attention_mask=sp.attention_mask[i:i + batch_size].to(self.device),
                    task_ids=sp.task_ids[i:i + batch_size].to(self.device),
                    s=sp.features[i:i + batch_size].to(self.device),
                )
                logits = out["logits"]
                n_c = TASKS[tid].n_classes if tid in TASKS else logits.size(-1)
                logits = logits[:, :n_c]
                preds.append(logits.argmax(-1).cpu().numpy())
                probs.append(torch.softmax(logits.float(), -1).cpu().numpy())
            y_pred = np.concatenate(preds)
            p = np.concatenate(probs)
            spec = TASKS.get(tid)
            results[tid] = classification_metrics(
                sp.labels.numpy(), y_pred, probs=p,
                n_classes=spec.n_classes if spec else p.shape[1],
                class_names=spec.class_names if spec else None,
            )
        return results

    # -- lines 11-34 -------------------------------------------------------
    def fit(self, tensors: Dict[str, Dict[str, TokenizedSplit]],
            checkpoint_path: Optional[str | Path] = None,
            verbose: bool = True) -> TrainResult:
        seed_everything(self.cfg.seed)
        t0 = time.time()

        train_ds = MultiTaskDataset(tensors["train"], self.task_order)
        sampler = PerTaskBatchSampler(train_ds, self.cfg.batch_size, seed=self.cfg.seed)
        loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate,
                            num_workers=self.cfg.num_workers)

        steps_per_epoch = max(len(sampler) // max(self.cfg.grad_accumulation, 1), 1)
        optimizer, scheduler = self._build_optimizer(steps_per_epoch * self.cfg.max_epochs)

        best_f1, best_state, patience, best_epoch = -math.inf, None, 0, 0
        history: List[Dict] = []
        initial_loss = final_loss = float("nan")
        early_stopped = False

        for epoch in range(1, self.cfg.max_epochs + 1):             # line 12
            sampler.set_epoch(epoch)
            stats = self._train_epoch(loader, optimizer, scheduler, None)
            val = self.evaluate(tensors["val"])                      # line 24
            val_f1 = aggregate_metrics(val, "macro_f1")

            if epoch == 1:
                initial_loss = stats["mean_loss"]
            final_loss = stats["mean_loss"]

            history.append({
                "epoch": epoch, "train_loss": stats["mean_loss"],
                "val_macro_f1": val_f1,
                "lr": float(scheduler.get_last_lr()[0]),
                **{f"val_{tid}_macro_f1": m["macro_f1"] for tid, m in val.items()},
            })
            if verbose:
                print(f"  epoch {epoch}  loss={stats['mean_loss']:.4f}  "
                      f"val_macroF1={val_f1:.4f}", flush=True)

            if val_f1 > best_f1:                                     # line 25
                best_f1, best_epoch, patience = val_f1, epoch, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in self.model.state_dict().items()}
            else:                                                    # line 27
                patience += 1
                if patience >= self.cfg.patience:                    # line 29
                    early_stopped = True
                    if verbose:
                        print(f"  early stopping at epoch {epoch} (patience="
                              f"{self.cfg.patience})", flush=True)
                    break

        if best_state is not None:                                   # line 33
            self.model.load_state_dict(best_state)
        test = self.evaluate(tensors["test"])
        val_best = self.evaluate(tensors["val"])

        result = TrainResult(
            variant=self.model.config.variant, backbone=self.model.config.backbone,
            seed=self.cfg.seed, best_epoch=best_epoch, best_val_macro_f1=float(best_f1),
            initial_loss=float(initial_loss), final_loss=float(final_loss),
            loss_reduction=float(initial_loss - final_loss),
            early_stopped=early_stopped, wall_clock_s=float(time.time() - t0),
            epochs_run=len(history), test_metrics=test, val_metrics=val_best,
            history=history, parameter_report=self.model.parameter_report(),
            config={**self.cfg.to_dict(), "model": self.model.config.to_dict()},
        )

        if checkpoint_path is not None:
            save_checkpoint(self.model, result, checkpoint_path)
        return result


def save_checkpoint(model: SurgeLLM, result: TrainResult,
                    path: str | Path) -> Path:
    """Persist weights + config + metrics so a run is fully reconstructible."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.checkpoint_payload({"result": result.to_dict()}), path)
    path.with_suffix(".json").write_text(
        json.dumps(result.to_dict(), indent=2, default=float), encoding="utf-8"
    )
    return path


def load_checkpoint(path: str | Path, offline_backbone: Optional[bool] = None) -> Tuple[SurgeLLM, Dict]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    cfg = SurgeLLMConfig.from_dict(payload["config"])
    if offline_backbone is not None:
        cfg.offline_backbone = offline_backbone
    model = SurgeLLM(cfg)
    model.load_state_dict(payload["state_dict"])
    return model, payload.get("result", {})
