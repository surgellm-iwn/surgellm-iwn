"""§3.2-3.7: the model, the gate, the heads and the variant table."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from surgellm.models.gate import SurgicalFeatureGate, TaskEmbedding
from surgellm.models.heads import MultiTaskHeads, parameter_share_report
from surgellm.models.surgellm import (VARIANT_TABLE, SurgeLLM, SurgeLLMConfig,
                                      variant_flags)

D = 32
B = 8
FD = 16


@pytest.fixture
def model():
    torch.manual_seed(0)
    return SurgeLLM(SurgeLLMConfig(
        variant="surgellm-iwn", backbone="tiny", offline_backbone=True,
        hidden_size_override=D, task_names=["D1", "D2"], n_classes_per_task=[2, 3],
    ))


# -- Table 1 ---------------------------------------------------------------
def test_variant_table_matches_paper():
    assert variant_flags("baseline") == dict(prefix=False, gate=False, extended=False, iwn=False)
    assert variant_flags("surgellm-g") == dict(prefix=True, gate=False, extended=False, iwn=False)
    assert variant_flags("surgellm-s") == dict(prefix=True, gate=True, extended=False, iwn=False)
    assert variant_flags("surgellm-full") == dict(prefix=True, gate=True, extended=True, iwn=False)
    assert variant_flags("surgellm-iwn") == dict(prefix=True, gate=True, extended=True, iwn=True)


def test_unknown_variant_raises():
    with pytest.raises(KeyError):
        variant_flags("surgellm-xyz")


# -- Eq. (3): the task embedding -------------------------------------------
def test_task_embedding_is_a_small_residual_perturbation():
    torch.manual_seed(0)
    te = TaskEmbedding(4, D, alpha=0.1)
    h = torch.randn(B, D)
    tids = torch.randint(0, 4, (B,))
    out = te(h, tids)
    assert out.shape == h.shape
    # The §3.2 argument: alpha||E|| is small relative to ||h||.
    assert te.perturbation_ratio(h, tids) < 0.25


def test_alpha_zero_is_the_identity():
    te = TaskEmbedding(2, D, alpha=0.0)
    h = torch.randn(B, D)
    torch.testing.assert_close(te(h, torch.zeros(B, dtype=torch.long)), h)


# -- Eqs. (5)-(7): the gate ------------------------------------------------
def test_gate_output_is_in_the_open_unit_interval_per_dimension():
    torch.manual_seed(0)
    g = SurgicalFeatureGate(D, FD)
    out = g(torch.randn(B, D), torch.rand(B, FD))
    assert out.gate.shape == (B, D), "per-dimension, not scalar"
    assert torch.all(out.gate > 0) and torch.all(out.gate < 1)


def test_gate_is_sigmoid_not_softmax():
    """A softmax would force the gate to sum to 1 across dimensions."""
    torch.manual_seed(0)
    g = SurgicalFeatureGate(D, FD)
    out = g(torch.randn(B, D), torch.rand(B, FD))
    sums = out.gate.sum(-1)
    assert torch.all(sums > 1.5), "sigmoid gates are not budget-constrained"


def test_feature_projection_is_nonnegative():
    """ReLU in Eq. (5) keeps s' in the positive orthant."""
    torch.manual_seed(0)
    g = SurgicalFeatureGate(D, FD)
    assert torch.all(g.project(torch.randn(B, FD)) >= 0)


def test_fusion_is_a_convex_combination_before_layernorm():
    torch.manual_seed(0)
    g = SurgicalFeatureGate(D, FD)
    h, s = torch.randn(B, D), torch.rand(B, FD)
    out = g(h, s)
    pre_ln = out.gate * out.task_conditioned + (1 - out.gate) * out.projected
    torch.testing.assert_close(out.fused, g.post_norm(pre_ln))


def test_disabled_gate_is_the_identity_path():
    torch.manual_seed(0)
    g = SurgicalFeatureGate(D, FD, enabled=False)
    h = torch.randn(B, D)
    out = g(h, torch.rand(B, FD))
    torch.testing.assert_close(out.fused, g.post_norm(h))
    assert torch.all(out.gate == 1.0)


def test_gate_with_zero_Ws_reduces_to_LN_of_h():
    """Proposition 2 conclusions (2)-(4) in their exact limit."""
    torch.manual_seed(0)
    g = SurgicalFeatureGate(D, FD)
    with torch.no_grad():
        g.feature_proj.weight.zero_()
        g.feature_proj.bias.zero_()
        g.gate_proj.bias.fill_(20.0)      # drives sigmoid -> 1
    h = torch.randn(B, D)
    out = g(h, torch.rand(B, FD))
    assert torch.all(out.projected == 0), "s'(x) -> 0"
    assert out.gate.min() > 0.99, "g_i -> 1"
    assert float((out.fused - g.post_norm(h)).abs().max()) < 1e-2


def test_spectral_norm_hooks_are_finite():
    g = SurgicalFeatureGate(D, FD)
    assert np.isfinite(g.spectral_norm_Wg()) and g.spectral_norm_Wg() > 0
    assert np.isfinite(g.lipschitz_constant()) and g.lipschitz_constant() > 0


# -- Eqs. (13)-(14): heads -------------------------------------------------
def test_heads_route_by_task_mask_and_mask_invalid_classes():
    torch.manual_seed(0)
    heads = MultiTaskHeads(D, [2, 3])
    h = torch.randn(6, D)
    tids = torch.tensor([0, 0, 0, 1, 1, 1])
    logits = heads(h, tids)
    assert logits.shape == (6, 3)
    assert torch.all(torch.isinf(logits[:3, 2])), "task 0 has only 2 classes"
    assert torch.all(torch.isfinite(logits[3:]))


def test_per_task_losses_are_summed():
    torch.manual_seed(0)
    heads = MultiTaskHeads(D, [2, 3])
    h = torch.randn(6, D)
    tids = torch.tensor([0, 0, 0, 1, 1, 1])
    y = torch.tensor([0, 1, 0, 2, 1, 0])
    out = heads.loss(h, tids, y)
    assert set(out["per_task"]) == {0, 1}
    torch.testing.assert_close(
        out["loss"], out["per_task"][0] + out["per_task"][1], rtol=1e-5, atol=1e-6)


# -- §3.1: shared-parameter share -----------------------------------------
def test_most_parameters_are_shared(model):
    rep = model.parameter_report()
    assert rep["shared_fraction"] > 0.90
    assert rep["task_specific_parameters"] > 0
    assert (rep["shared_parameters"] + rep["task_specific_parameters"]
            == rep["total_parameters"])


def test_surgical_overhead_is_small_relative_to_the_encoder(model):
    rep = model.parameter_report()
    assert 0 < rep["surgical_overhead_fraction"] < 0.15


# -- end-to-end forward ----------------------------------------------------
def test_forward_shapes_and_loss(model):
    enc = model.tokenize(["a great film", "according to the report"])
    out = model(
        input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
        task_ids=torch.tensor([0, 1]), s=torch.rand(2, FD),
        labels=torch.tensor([1, 2]),
    )
    assert out["logits"].shape == (2, 3)
    assert out["loss"].ndim == 0 and torch.isfinite(out["loss"])
    out["loss"].backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all()
               for p in model.parameters() if p.requires_grad)


def test_gradients_reach_the_gate(model):
    enc = model.tokenize(["great", "terrible"])
    out = model(enc["input_ids"], enc["attention_mask"], torch.tensor([0, 0]),
                torch.rand(2, FD), labels=torch.tensor([0, 1]))
    out["loss"].backward()
    assert model.gate.feature_proj.weight.grad is not None
    assert model.gate.gate_proj.weight.grad is not None


def test_checkpoint_round_trip(model):
    payload = model.checkpoint_payload()
    clone = SurgeLLM.from_checkpoint(payload)
    enc = model.tokenize(["hello world"])
    args = (enc["input_ids"], enc["attention_mask"], torch.tensor([0]), torch.rand(1, FD))
    model.eval(); clone.eval()
    with torch.no_grad():
        torch.testing.assert_close(model(*args)["logits"], clone(*args)["logits"])


def test_baseline_variant_builds_without_the_gate():
    m = SurgeLLM(SurgeLLMConfig(variant="baseline", backbone="tiny",
                                offline_backbone=True, hidden_size_override=D))
    assert m.gate.enabled is False


@pytest.mark.parametrize("backbone", ["albert", "distilbert", "bert", "roberta", "t5"])
def test_every_demo_backbone_builds_and_differs(backbone):
    m = SurgeLLM(SurgeLLMConfig(variant="surgellm-iwn", backbone=backbone,
                                offline_backbone=True))
    assert m.hidden_size >= 64
    enc = m.tokenize(["x y z"])
    with torch.no_grad():
        h = m.backbone.encode(enc["input_ids"], enc["attention_mask"])
    assert h.shape == (1, m.hidden_size)
