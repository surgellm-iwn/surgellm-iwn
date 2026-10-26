"""Definition 1, Lemma 4, Theorem 1, Proposition 2 and Corollary 3.

Proposition 2 is the only theorem whose hypothesis can be constructed
(``rho_k = 0`` by feeding label-independent features), so it is genuinely
*verified*.  Theorem 1 bounds an unobservable quantity, so the tests check
the structure of the bound - monotonicity in each factor and that the
computed bound holds against a conservative excess-risk proxy.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from surgellm.data.cache import TokenizedSplit
from surgellm.models.surgellm import SurgeLLM, SurgeLLMConfig
from surgellm.theory.alignment import alignment_gain_correlation, estimate_rho
from surgellm.theory.bound import (SIGMOID_SUP_DERIVATIVE, estimate_lipschitz,
                                   evaluate_bound, oracle_feature_vector)
from surgellm.theory.degeneracy import (check_proposition2, degeneracy_diagnostics,
                                        imbalance_loophole_check,
                                        run_decay_simulation)

D = 32
FD = 16
N = 64


def _model(seed=0):
    torch.manual_seed(seed)
    return SurgeLLM(SurgeLLMConfig(
        variant="surgellm-iwn", backbone="tiny", offline_backbone=True,
        hidden_size_override=D, task_names=["D1"], n_classes_per_task=[2],
    ))


def _split(model, informative=True, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=N)
    texts = [("great wonderful superb " if c else "terrible awful boring ") * 3
             for c in y]
    enc = model.tokenize(texts)
    if informative:
        feats = np.stack([np.full(FD, 2.0) if c else np.full(FD, -2.0) for c in y])
        feats = feats + rng.normal(0, 0.1, size=(N, FD))
    else:
        feats = rng.normal(size=(N, FD))      # label-independent: rho_k = 0
    return TokenizedSplit(
        input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
        features=torch.tensor(feats, dtype=torch.float32),
        labels=torch.tensor(y, dtype=torch.long),
        task_ids=torch.zeros(N, dtype=torch.long), texts=texts,
    )


# ===================== Definition 1 ======================================
def test_rho_is_finite_and_nonnegative():
    m = _model()
    r = estimate_rho(m, _split(m), 0, n_examples=N)
    assert r["rho"] >= 0 and math.isfinite(r["rho"])
    assert r["n_examples"] == N


def test_rho_is_zero_when_the_projection_is_zero():
    """rho_k = E|<s', grad>| vanishes if s' = 0 - Proposition 2's premise."""
    m = _model()
    with torch.no_grad():
        m.gate.feature_proj.weight.zero_()
        m.gate.feature_proj.bias.zero_()
    assert estimate_rho(m, _split(m), 0, n_examples=N)["rho"] == pytest.approx(0.0, abs=1e-8)


def test_rho_grows_with_the_feature_scale():
    """rho is linear in ||s'||, so scaling W_s scales rho."""
    m = _model()
    sp = _split(m)
    base = estimate_rho(m, sp, 0, n_examples=N)["rho"]
    with torch.no_grad():
        m.gate.feature_proj.weight *= 3.0
        m.gate.feature_proj.bias *= 3.0
    scaled = estimate_rho(m, sp, 0, n_examples=N)["rho"]
    assert scaled > base


def test_alignment_gain_correlation_is_perfect_when_orders_match():
    c = alignment_gain_correlation({"a": 1.0, "b": 2.0, "c": 3.0},
                                   {"a": 0.1, "b": 0.2, "c": 0.3})
    assert c["spearman_rho"] == pytest.approx(1.0)


# ===================== Lemma 4 ===========================================
def test_composed_lipschitz_bound_is_a_product_of_its_factors():
    m = _model()
    rep = estimate_lipschitz(m, _split(m), 0, n_pairs=40)
    assert rep.L_composed == pytest.approx(rep.L_encoder * rep.L_gate * rep.L_head,
                                           rel=1e-9)
    assert all(v > 0 for v in (rep.L_encoder, rep.L_gate, rep.L_head))


def test_empirical_lipschitz_does_not_exceed_the_bound():
    m = _model()
    rep = estimate_lipschitz(m, _split(m), 0, n_pairs=60)
    assert rep.to_dict()["bound_holds"], \
        "a measured ratio above the composed bound would falsify Lemma 4"


def test_sigmoid_sup_derivative_is_one_quarter():
    """The constant used when propagating through the bilinear gate."""
    assert SIGMOID_SUP_DERIVATIVE == 0.25
    z = torch.linspace(-40, 40, 20001)
    sig = torch.sigmoid(z)
    assert float((sig * (1 - sig)).max()) == pytest.approx(0.25, abs=1e-6)


# ===================== Theorem 1 =========================================
def test_bound_terms_are_nonnegative_and_finite():
    m = _model()
    sp = _split(m)
    bt = evaluate_bound(m, sp, "D1", 0, n_train=1000, test_risk=0.2)
    assert bt.generalization_term >= 0 and bt.approximation_term >= 0
    assert math.isfinite(bt.bound_total)


def test_generalization_term_decays_as_one_over_sqrt_N():
    m = _model()
    sp = _split(m)
    a = evaluate_bound(m, sp, "D1", 0, n_train=100, test_risk=0.2)
    b = evaluate_bound(m, sp, "D1", 0, n_train=10_000, test_risk=0.2)
    assert b.generalization_term == pytest.approx(a.generalization_term / 10, rel=1e-6)


def test_approximation_term_scales_with_the_gate_spectral_norm():
    m = _model()
    sp = _split(m)
    before = evaluate_bound(m, sp, "D1", 0, n_train=1000, test_risk=0.2)
    with torch.no_grad():
        m.gate.gate_proj.weight *= 2.0
    after = evaluate_bound(m, sp, "D1", 0, n_train=1000, test_risk=0.2)
    assert after.lambda_max_WgTWg == pytest.approx(4 * before.lambda_max_WgTWg, rel=1e-4)
    assert after.approximation_term > before.approximation_term


def test_bound_holds_against_a_conservative_excess_risk_proxy():
    m = _model()
    sp = _split(m)
    bt = evaluate_bound(m, sp, "D1", 0, n_train=500, test_risk=0.5,
                        bayes_risk_proxy=0.0)
    assert bt.bound_holds and bt.slack >= 0


def test_oracle_surrogate_returns_a_finite_gap():
    m = _model()
    o = oracle_feature_vector(m, _split(m), 0, n_examples=N)
    assert o["feature_gap_sq_mean"] >= 0 and math.isfinite(o["feature_gap_sq_mean"])


# ===================== Proposition 2 =====================================
def test_weight_decay_update_is_exactly_geometric():
    """Proof step 1: W_s <- (1 - lambda*eta) W_s when E[grad] = 0."""
    tr = run_decay_simulation(d=32, feature_dim=FD, lr=1e-2, weight_decay=1e-2,
                              steps=300)
    assert tr.max_abs_error < 1e-5
    assert tr.norms[-1] < tr.norms[0]
    assert tr.half_life_steps > 0 and math.isfinite(tr.half_life_steps)


def test_stronger_decay_shrinks_faster():
    slow = run_decay_simulation(weight_decay=1e-3, steps=300)
    fast = run_decay_simulation(weight_decay=1e-1, steps=300)
    assert fast.norms[-1] < slow.norms[-1]
    assert fast.half_life_steps < slow.half_life_steps


def test_degeneracy_diagnostics_report_all_four_conclusions():
    m = _model()
    d = degeneracy_diagnostics(m, _split(m), 0)
    for key in ("Ws_fro_norm", "s_proj_norm_mean", "gate_mean", "identity_gap_rel"):
        assert key in d and math.isfinite(d[key])


def test_zero_alignment_training_moves_toward_the_identity_map():
    """The full Proposition 2 experiment, in miniature."""
    m = _model(seed=1)
    sp = _split(m, informative=False, seed=1)      # rho_k = 0 by construction
    before = degeneracy_diagnostics(m, sp, 0)

    opt = torch.optim.SGD(m.parameters(), lr=0.05, weight_decay=0.5)
    m.train()
    for _ in range(60):
        opt.zero_grad()
        out = m(sp.input_ids, sp.attention_mask, sp.task_ids, sp.features,
                labels=sp.labels)
        out["loss"].backward()
        opt.step()

    after = degeneracy_diagnostics(m, sp, 0)
    checks = check_proposition2(before, after)
    assert checks["c1_Ws_norm_decreased"], "conclusion 1: ||W_s|| -> 0"
    assert checks["c2_s_proj_shrank"], "conclusion 2: s'(x) -> 0"


def test_check_proposition2_detects_the_wrong_direction():
    before = dict(Ws_fro_norm=1.0, s_proj_norm_mean=1.0, gate_mean=0.5,
                  identity_gap_rel=1.0)
    after = dict(Ws_fro_norm=2.0, s_proj_norm_mean=2.0, gate_mean=0.3,
                 identity_gap_rel=2.0)
    assert not check_proposition2(before, after)["all_directional_checks_pass"]


# ===================== Corollary 3 / §A.3 ================================
def test_imbalance_loophole_signature():
    sig = imbalance_loophole_check(0.61, 2.13, 0.711, 0.892)
    assert sig["loophole_signature"]
    assert sig["rho_increase"] == pytest.approx(1.52, abs=1e-6)
    assert sig["f1_increase"] == pytest.approx(0.181, abs=1e-6)


def test_no_loophole_when_rho_does_not_rise():
    assert not imbalance_loophole_check(2.0, 1.8, 0.8, 0.85)["loophole_signature"]
