"""The SURGeLLM model (§3) and the Table 1 variant families.

Table 1 (P = surgical prefix, G = gate, E = extended training, I = IWN):

    ============================  ====  ====  ====  ====
    Variant                       P     G     E     I
    ============================  ====  ====  ====  ====
    Baseline                      no    no    no    no
    T5-base                       n/a   n/a   n/a   n/a
    SURGeLLM-G                    yes   no    no    no
    SURGeLLM-S                    yes   yes   no    no
    SURGeLLM-FULL                 yes   yes   yes   no
    SURGeLLM-IWN  (this work)     yes   yes   yes   yes
    ============================  ====  ====  ====  ====

Note the naming in the paper: ``-G`` carries the *prefix only* and ``-S``
adds the gate (Table 7 columns are authoritative; RQ2 compares
"surgical features without the gate" against SURGeLLM-G).  ``E`` is a
training-schedule flag, not an architectural one, so it lives in the
training config rather than the module.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .backbone import EncoderBackbone, build_backbone, BACKBONE_REGISTRY
from .gate import SurgicalFeatureGate, TaskEmbedding, GateOutput
from .heads import MultiTaskHeads, parameter_share_report

__all__ = ["SurgeLLMConfig", "SurgeLLM", "VARIANT_TABLE", "variant_flags"]


#: Table 1 as data.  ``extended`` only changes ``max_epochs`` / LR schedule.
VARIANT_TABLE: Dict[str, Dict[str, bool]] = {
    "baseline":      {"prefix": False, "gate": False, "extended": False, "iwn": False},
    "t5":            {"prefix": False, "gate": False, "extended": True,  "iwn": False},
    "surgellm-g":    {"prefix": True,  "gate": False, "extended": False, "iwn": False},
    "surgellm-s":    {"prefix": True,  "gate": True,  "extended": False, "iwn": False},
    "surgellm-full": {"prefix": True,  "gate": True,  "extended": True,  "iwn": False},
    "surgellm-iwn":  {"prefix": True,  "gate": True,  "extended": True,  "iwn": True},
}


def variant_flags(variant: str) -> Dict[str, bool]:
    key = variant.lower()
    if key not in VARIANT_TABLE:
        raise KeyError(f"unknown variant {variant!r}; known: {sorted(VARIANT_TABLE)}")
    return dict(VARIANT_TABLE[key])


@dataclass
class SurgeLLMConfig:
    """Everything needed to rebuild a model from a checkpoint."""

    variant: str = "surgellm-iwn"
    backbone: str = "roberta"
    task_names: List[str] = field(default_factory=lambda: ["D1", "D2", "D3", "D4"])
    n_classes_per_task: List[int] = field(default_factory=lambda: [2, 2, 2, 2])
    feature_dim: int = 16
    alpha: float = 0.1                 # Eq. (3)
    dropout1: float = 0.1
    dropout2: float = 0.05
    gate_bias_init: float = 0.0
    offline_backbone: bool = False     # TinyEncoderBackbone
    hidden_size_override: Optional[int] = None
    max_length: Optional[int] = None
    # resolved from VARIANT_TABLE unless explicitly overridden
    use_prefix: Optional[bool] = None
    use_gate: Optional[bool] = None
    use_iwn: Optional[bool] = None
    extended: Optional[bool] = None

    def __post_init__(self) -> None:
        flags = variant_flags(self.variant)
        if self.use_prefix is None:
            self.use_prefix = flags["prefix"]
        if self.use_gate is None:
            self.use_gate = flags["gate"]
        if self.use_iwn is None:
            self.use_iwn = flags["iwn"]
        if self.extended is None:
            self.extended = flags["extended"]
        if len(self.task_names) != len(self.n_classes_per_task):
            raise ValueError("task_names and n_classes_per_task must align")

    @property
    def n_tasks(self) -> int:
        return len(self.task_names)

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(payload: Dict) -> "SurgeLLMConfig":
        known = {f for f in SurgeLLMConfig.__dataclass_fields__}  # type: ignore[attr-defined]
        return SurgeLLMConfig(**{k: v for k, v in payload.items() if k in known})


class SurgeLLM(nn.Module):
    """Unified multi-task encoder with prefix, gate, IWN and per-task heads.

    Forward path (paper numbering):

    ``x`` --[prefix, Eq. 12]--> tokenizer --> encoder --[Eq. 2]--> ``h``
    --[Eq. 3]--> ``htilde`` --[Eqs. 5-7 with ``s``]--> ``hhat``
    --[Eqs. 13-14]--> logits

    ``s`` arrives already standardised (Eq. 8 or Eq. 11): normalisation is a
    *data-side* transform fitted on the training partition, which is what
    makes IWN parameter-free and test-time class-agnostic.
    """

    def __init__(self, config: SurgeLLMConfig, backbone: Optional[EncoderBackbone] = None):
        super().__init__()
        self.config = config
        self.backbone = backbone or build_backbone(
            config.backbone,
            offline=config.offline_backbone,
            **({"hidden_size": config.hidden_size_override}
               if config.hidden_size_override else {}),
        )
        d = self.backbone.hidden_size
        self.hidden_size = d

        self.task_embedding = TaskEmbedding(config.n_tasks, d, alpha=config.alpha)
        self.gate = SurgicalFeatureGate(
            d, feature_dim=config.feature_dim,
            gate_bias_init=config.gate_bias_init,
            enabled=bool(config.use_gate),
        )
        self.heads = MultiTaskHeads(
            d, config.n_classes_per_task,
            dropout1=config.dropout1, dropout2=config.dropout2,
        )

    # -- forward ----------------------------------------------------------
    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
               task_ids: torch.Tensor, s: torch.Tensor) -> GateOutput:
        h = self.backbone.encode(input_ids, attention_mask)            # Eq. (2)
        h_tilde = self.task_embedding(h, task_ids)                     # Eq. (3)
        return self.gate(h_tilde, s)                                   # Eqs. (5)-(7)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                task_ids: torch.Tensor, s: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        gate_out = self.encode(input_ids, attention_mask, task_ids, s)
        logits = self.heads(gate_out.fused, task_ids)
        result: Dict[str, torch.Tensor] = {
            "logits": logits,
            "fused": gate_out.fused,
            "gate": gate_out.gate,
            "projected": gate_out.projected,
        }
        if labels is not None:
            loss_out = self.heads.loss(gate_out.fused, task_ids, labels)
            result["loss"] = loss_out["loss"]
            result["per_task_loss"] = loss_out["per_task"]  # type: ignore[assignment]
        return result

    # -- bookkeeping ------------------------------------------------------
    def parameter_report(self) -> Dict[str, float]:
        rep = parameter_share_report(self, head_module_name="heads")
        rep["backbone_parameters"] = float(sum(p.numel() for p in self.backbone.parameters()))
        rep["gate_parameters"] = float(sum(p.numel() for p in self.gate.parameters()))
        rep["task_embedding_parameters"] = float(
            sum(p.numel() for p in self.task_embedding.parameters())
        )
        rep["surgical_overhead_parameters"] = (
            rep["gate_parameters"] + rep["task_embedding_parameters"]
        )
        rep["surgical_overhead_fraction"] = (
            rep["surgical_overhead_parameters"] / max(rep["backbone_parameters"], 1.0)
        )
        rep["reported_backbone_parameters"] = float(
            BACKBONE_REGISTRY[self.config.backbone].reported_params
        )
        return rep

    def tokenize(self, texts: Sequence[str]) -> Dict[str, torch.Tensor]:
        return self.backbone.tokenize(texts)

    # -- checkpointing ----------------------------------------------------
    def checkpoint_payload(self, extra: Optional[Dict] = None) -> Dict:
        payload = {
            "config": self.config.to_dict(),
            "state_dict": self.state_dict(),
            "format_version": 1,
        }
        if extra:
            payload.update(extra)
        return payload

    @staticmethod
    def from_checkpoint(payload: Dict, backbone: Optional[EncoderBackbone] = None) -> "SurgeLLM":
        cfg = SurgeLLMConfig.from_dict(payload["config"])
        model = SurgeLLM(cfg, backbone=backbone)
        model.load_state_dict(payload["state_dict"])
        return model
