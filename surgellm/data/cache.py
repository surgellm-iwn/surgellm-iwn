"""Pre-tokenisation with chunked caching (Algorithm 1, line 8).

The paper caches training/validation texts as tensors in chunks of 2,048
for efficient multi-GPU loading, reporting a ~25% speedup.  This module
implements that cache and the multi-task dataset that feeds Algorithm 1,
including the per-task batch sampler used "for balance" (§3.1).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

__all__ = [
    "CHUNK_SIZE", "TokenizedSplit", "MultiTaskDataset",
    "PerTaskBatchSampler", "collate", "tokenize_with_cache",
]

CHUNK_SIZE = 2048


@dataclass
class TokenizedSplit:
    """Tensors for one task split."""

    input_ids: torch.Tensor        # (N, L) long
    attention_mask: torch.Tensor   # (N, L) long
    features: torch.Tensor         # (N, 16) float, already standardised
    labels: torch.Tensor           # (N,) long
    task_ids: torch.Tensor         # (N,) long
    texts: List[str]

    def __len__(self) -> int:
        return int(self.labels.numel())


def _cache_key(texts: Sequence[str], prefixes: Optional[Sequence[str]],
               tokenizer_tag: str, max_length: int) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(tokenizer_tag.encode())
    h.update(str(max_length).encode())
    h.update(str(prefixes is not None).encode())
    h.update(str(len(texts)).encode())
    # Hash in chunks so the key is cheap but still collision-safe.
    for i in range(0, len(texts), 97):
        h.update(texts[i].encode("utf-8", "ignore"))
        if prefixes is not None:
            h.update(prefixes[i].encode("utf-8", "ignore"))
    return h.hexdigest()


def tokenize_with_cache(backbone, texts: Sequence[str],
                        prefixes: Optional[Sequence[str]] = None,
                        cache_dir: Optional[str | Path] = None,
                        tokenizer_tag: str = "default",
                        chunk_size: int = CHUNK_SIZE) -> Dict[str, torch.Tensor]:
    """Tokenise in chunks of ``chunk_size``, caching the result on disk.

    When ``prefixes`` is given (the Eq. 12 surgical prefix), each prefix is
    concatenated *in front of* its text and tokenised jointly, so the prefix
    representations are co-attended to by every layer (§3.6).
    """
    joined = (
        list(texts) if prefixes is None
        else [f"{p} {t}" for p, t in zip(prefixes, texts)]
    )
    max_length = int(getattr(backbone, "max_length", 128))

    path: Optional[Path] = None
    if cache_dir is not None:
        key = _cache_key(list(texts), list(prefixes) if prefixes else None,
                         tokenizer_tag, max_length)
        path = Path(cache_dir) / f"tok_{key}.pt"
        if path.exists():
            blob = torch.load(path, map_location="cpu")
            return {"input_ids": blob["input_ids"], "attention_mask": blob["attention_mask"]}

    ids_chunks, mask_chunks = [], []
    for i in range(0, len(joined), chunk_size):
        enc = backbone.tokenize(joined[i:i + chunk_size])
        ids_chunks.append(enc["input_ids"])
        mask_chunks.append(enc["attention_mask"])

    out = {
        "input_ids": torch.cat(ids_chunks, 0) if ids_chunks else torch.zeros(0, max_length, dtype=torch.long),
        "attention_mask": torch.cat(mask_chunks, 0) if mask_chunks else torch.zeros(0, max_length, dtype=torch.long),
    }
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(out, path)
    return out


class MultiTaskDataset(Dataset):
    """Concatenation of per-task :class:`TokenizedSplit` objects."""

    def __init__(self, splits: Dict[str, TokenizedSplit], task_order: Sequence[str]):
        self.task_order = list(task_order)
        self.splits = splits
        self.index: List[tuple] = []
        for t, tid in enumerate(self.task_order):
            if tid not in splits:
                continue
            for i in range(len(splits[tid])):
                self.index.append((tid, i, t))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        tid, j, t = self.index[i]
        sp = self.splits[tid]
        return {
            "input_ids": sp.input_ids[j],
            "attention_mask": sp.attention_mask[j],
            "features": sp.features[j],
            "labels": sp.labels[j],
            "task_ids": torch.tensor(t, dtype=torch.long),
        }

    def task_index_groups(self) -> Dict[str, List[int]]:
        groups: Dict[str, List[int]] = {tid: [] for tid in self.task_order}
        for pos, (tid, _, _) in enumerate(self.index):
            groups[tid].append(pos)
        return groups


class PerTaskBatchSampler(Sampler):
    """Homogeneous-task batches, interleaved in random order.

    §3.1: "We use ``w_k = 1`` throughout and rely on per-task batch sampling
    for balance."  Each batch therefore draws from a single task, and task
    turns are shuffled per epoch so no task dominates a gradient window.
    """

    def __init__(self, dataset: MultiTaskDataset, batch_size: int,
                 seed: int = 0, drop_last: bool = False):
        self.groups = {k: v for k, v in dataset.task_index_groups().items() if v}
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed * 10_000 + self.epoch)
        batches: List[List[int]] = []
        for idxs in self.groups.values():
            arr = np.array(idxs)
            rng.shuffle(arr)
            for i in range(0, len(arr), self.batch_size):
                chunk = arr[i:i + self.batch_size].tolist()
                if self.drop_last and len(chunk) < self.batch_size:
                    continue
                batches.append(chunk)
        order = rng.permutation(len(batches))
        for i in order:
            yield batches[int(i)]

    def __len__(self) -> int:
        n = 0
        for idxs in self.groups.values():
            n += (len(idxs) // self.batch_size) if self.drop_last else \
                 -(-len(idxs) // self.batch_size)
        return n


def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch], 0) for k in batch[0]}
