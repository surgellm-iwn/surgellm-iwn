"""Encoder backbones (§3.2).

Two backends share one interface:

``HFEncoderBackbone``
    Wraps any HuggingFace ``AutoModel`` (BERT, RoBERTa, DistilBERT, ALBERT
    in the paper).  This is the backend used for the reported numbers.

``TinyEncoderBackbone``
    A small, self-contained transformer encoder with a hash-based tokenizer.
    It exists so that the *entire* pipeline - Algorithm 1, IWN, the gate,
    every ablation and every theory check - runs end to end on CPU with no
    model download and no network.  It is a reproduction harness, not a
    paper baseline; every artifact produced with it is labelled ``demo``.

Both expose:

    ``encode(input_ids, attention_mask) -> (B, d)``   the [CLS] embedding, Eq. (2)
    ``hidden_size``, ``n_parameters``, ``max_length``, ``tokenize(texts)``
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

__all__ = [
    "BackboneSpec", "EncoderBackbone", "TinyEncoderBackbone",
    "HFEncoderBackbone", "build_backbone", "BACKBONE_REGISTRY",
]


@dataclass(frozen=True)
class BackboneSpec:
    """Static facts about a backbone, used for the Table 3/12 ``Par.`` column."""

    key: str
    hf_name: str
    reported_params: int      # as printed in the paper
    hidden_size: int = 768
    max_length: int = 128


#: Paper §5 (Setup).  ALBERT-base-v2 is 11M, DistilBERT 66M, BERT 110M,
#: RoBERTa 125M, T5-base 220M.
BACKBONE_REGISTRY: Dict[str, BackboneSpec] = {
    "distilbert": BackboneSpec("distilbert", "distilbert-base-uncased", 66_000_000, 768, 96),
    "bert": BackboneSpec("bert", "bert-base-uncased", 110_000_000, 768, 128),
    "roberta": BackboneSpec("roberta", "roberta-base", 125_000_000, 768, 128),
    "albert": BackboneSpec("albert", "albert-base-v2", 11_000_000, 768, 96),
    "t5": BackboneSpec("t5", "t5-base", 220_000_000, 768, 128),
    "xlmr": BackboneSpec("xlmr", "xlm-roberta-base", 278_000_000, 768, 128),
    # demo-only backbones (TinyEncoderBackbone)
    "tiny": BackboneSpec("tiny", "tiny-demo", 0, 128, 128),
    "tiny-small": BackboneSpec("tiny-small", "tiny-demo-small", 0, 64, 96),
}


class EncoderBackbone(nn.Module):
    """Interface every backbone implements."""

    hidden_size: int
    max_length: int

    def tokenize(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Tiny offline backbone
# ---------------------------------------------------------------------------
class _HashTokenizer:
    """Deterministic hashing tokenizer (no vocab file, no download).

    Word-level hashing into ``vocab_size`` buckets.  Reserved ids:
    0 = PAD, 1 = CLS, 2 = UNK/SEP.
    """

    PAD, CLS, SEP = 0, 1, 2

    def __init__(self, vocab_size: int = 8192, max_length: int = 128):
        self.vocab_size = vocab_size
        self.max_length = max_length

    def _id(self, token: str) -> int:
        h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return 3 + (int.from_bytes(h, "big") % (self.vocab_size - 3))

    def __call__(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        import re
        pat = re.compile(r"\[TASK:[^\]]*\]|[A-Za-z0-9']+|[^\sA-Za-z0-9]")
        L = self.max_length
        ids = torch.zeros(len(texts), L, dtype=torch.long)
        mask = torch.zeros(len(texts), L, dtype=torch.long)
        for i, text in enumerate(texts):
            toks = pat.findall(text.lower())[: L - 2]
            row = [self.CLS] + [self._id(t) for t in toks] + [self.SEP]
            ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
            mask[i, : len(row)] = 1
        return {"input_ids": ids, "attention_mask": mask}


class TinyEncoderBackbone(EncoderBackbone):
    """Small pre-LN transformer encoder; CPU-friendly offline stand-in."""

    def __init__(self, hidden_size: int = 128, n_layers: int = 2, n_heads: int = 4,
                 vocab_size: int = 8192, max_length: int = 128, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.tokenizer = _HashTokenizer(vocab_size, max_length)
        self.embed = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.pos = nn.Embedding(max_length, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=n_heads, dim_feedforward=4 * hidden_size,
            dropout=dropout, batch_first=True, norm_first=True, activation="gelu",
        )
        # enable_nested_tensor is incompatible with norm_first and only emits
        # a warning; disable it explicitly rather than let it print per model.
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers,
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden_size)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

    def tokenize(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        return self.tokenizer(list(texts))

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B, L = input_ids.shape
        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        h = self.embed(input_ids) * math.sqrt(self.hidden_size) + self.pos(positions)
        out = self.encoder(h, src_key_padding_mask=(attention_mask == 0))
        return self.norm(out[:, 0])            # Eq. (2): the [CLS] slot


# ---------------------------------------------------------------------------
# HuggingFace backbone
# ---------------------------------------------------------------------------
class HFEncoderBackbone(EncoderBackbone):
    """Thin wrapper around ``AutoModel``; used for the paper's results."""

    def __init__(self, hf_name: str, max_length: int = 128, cache_dir: Optional[str] = None):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer  # local import: optional dep

        self.hf_name = hf_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(hf_name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(hf_name, cache_dir=cache_dir)
        self.hidden_size = int(self.model.config.hidden_size)

    def tokenize(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            list(texts), padding="max_length", truncation=True,
            max_length=self.max_length, return_tensors="pt",
        )
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # Eq. (2): h = E_phi(x)_[0]. Index 0 is [CLS] for BERT-family and <s>
        # for RoBERTa; T5 has no CLS, so we mean-pool its encoder states.
        last = out.last_hidden_state
        if "t5" in self.hf_name:
            m = attention_mask.unsqueeze(-1).to(last.dtype)
            return (last * m).sum(1) / m.sum(1).clamp(min=1.0)
        return last[:, 0]


#: Offline counterparts of the paper's backbones.
#:
#: In demo mode every row of Table 3 would otherwise be the *same* model,
#: making the eleven-variant comparison degenerate and the across-model
#: significance tests vacuous.  Each paper backbone therefore maps to a
#: distinct tiny configuration whose capacity ordering mirrors the real
#: one (ALBERT smallest, DistilBERT < BERT < RoBERTa < T5), and whose
#: ``max_length`` matches the paper's per-backbone setting from Table 13.
#: These are *stand-ins*: they preserve the shape of the comparison, not
#: its absolute numbers.
DEMO_TINY_CONFIG: Dict[str, Dict[str, int]] = {
    "albert":     dict(hidden_size=64,  n_layers=2, n_heads=4, max_length=96),
    "distilbert": dict(hidden_size=96,  n_layers=2, n_heads=4, max_length=96),
    "bert":       dict(hidden_size=128, n_layers=2, n_heads=4, max_length=128),
    "roberta":    dict(hidden_size=160, n_layers=3, n_heads=4, max_length=128),
    "t5":         dict(hidden_size=192, n_layers=3, n_heads=4, max_length=128),
    "xlmr":       dict(hidden_size=160, n_layers=3, n_heads=4, max_length=128),
    "tiny":       dict(hidden_size=128, n_layers=2, n_heads=4, max_length=128),
    "tiny-small": dict(hidden_size=64,  n_layers=2, n_heads=4, max_length=96),
}


def build_backbone(key: str, offline: bool = False, cache_dir: Optional[str] = None,
                   **tiny_kwargs) -> EncoderBackbone:
    """Construct a backbone by registry key.

    ``offline=True`` (or a ``tiny*`` key) builds the :class:`TinyEncoderBackbone`
    stand-in from :data:`DEMO_TINY_CONFIG`; otherwise the corresponding
    HuggingFace encoder is loaded.  If ``transformers`` is missing or the
    download fails we fail loudly rather than silently substituting the tiny
    model - a silent substitution would make a "paper-faithful" run quietly
    not be one.
    """
    if key not in BACKBONE_REGISTRY:
        raise KeyError(f"unknown backbone {key!r}; known: {sorted(BACKBONE_REGISTRY)}")
    spec = BACKBONE_REGISTRY[key]
    if offline or key.startswith("tiny"):
        cfg = dict(DEMO_TINY_CONFIG.get(key, DEMO_TINY_CONFIG["tiny"]))
        cfg.update(tiny_kwargs)
        return TinyEncoderBackbone(**cfg)
    return HFEncoderBackbone(spec.hf_name, max_length=spec.max_length, cache_dir=cache_dir)
