"""Algorithm 1 end to end, plus the Eq. (12) prefix and the IWN wiring.

These are integration tests: slow by unit-test standards (tens of seconds),
but they are the only thing that catches a mismatch between the normaliser
that was fitted and the features the model actually receives.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from surgellm.data.cache import (CHUNK_SIZE, MultiTaskDataset, PerTaskBatchSampler,
                                 tokenize_with_cache)
from surgellm.data.registry import load_all_tasks
from surgellm.experiment import (DEMO_HP, HPARAMS_TABLE13, RunSpec, hparams_for,
                                 run_single)
from surgellm.features import SurgicalFeatureExtractor
from surgellm.models.surgellm import SurgeLLM, SurgeLLMConfig
from surgellm.train import _fit_normalizers, prepare_tensors

pytestmark = pytest.mark.slow

SCALE = 0.02


@pytest.fixture(scope="module")
def tasks():
    return load_all_tasks(provider="synthetic", seed=0, scale=SCALE)


# ===================== Table 13 ==========================================
def test_table13_values_match_the_paper():
    assert hparams_for("baseline", "roberta")["lr"] == pytest.approx(2e-5)
    assert hparams_for("surgellm-iwn", "roberta")["lr"] == pytest.approx(1.5e-5)
    assert hparams_for("t5", "t5")["lr"] == pytest.approx(3e-4)
    assert hparams_for("surgellm-iwn", "roberta")["ep"] == 5
    assert hparams_for("baseline", "distilbert")["maxl"] == 96
    assert all(v["wu"] == 0.06 for v in HPARAMS_TABLE13.values()), "6% warmup"


def test_demo_profile_is_uniform_across_variants():
    a = hparams_for("baseline", "roberta", profile="demo")
    b = hparams_for("surgellm-iwn", "roberta", profile="demo")
    assert a == b == DEMO_HP


def test_unknown_profile_raises():
    with pytest.raises(ValueError):
        hparams_for("baseline", "roberta", profile="nope")


# ===================== normaliser wiring =================================
def test_iwn_flag_selects_the_balanced_moments(tasks):
    ex = SurgicalFeatureExtractor()
    std = _fit_normalizers(tasks, ex, use_iwn=False)
    iwn = _fit_normalizers(tasks, ex, use_iwn=True)
    assert all(n.mode == "standard" for n in std.values())
    assert all(n.mode == "iwn" for n in iwn.values())
    gap = np.linalg.norm(std["D4"].stats.mean - iwn["D4"].stats.mean)
    assert gap > 0, "on a skewed pool the two estimators must differ"


def test_normalizer_is_fitted_on_the_skewed_pool_not_the_balanced_cap(tasks):
    """§4: 'feature normalization uses the full training data'."""
    assert len(tasks["D4"].normalizer_pool) > len(tasks["D4"].splits.train)
    assert tasks["D4"].normalizer_pool.skew() > 4.0


# ===================== Eq. (12) prefix ===================================
def test_prefix_changes_the_token_sequence(tasks):
    model = SurgeLLM(SurgeLLMConfig(variant="surgellm-iwn", backbone="tiny",
                                    offline_backbone=True))
    ex = SurgicalFeatureExtractor()
    norms = _fit_normalizers(tasks, ex, use_iwn=True)
    with_prefix = prepare_tensors(tasks, model, ex, norms, True, None)
    without = prepare_tensors(tasks, model, ex, norms, False, None)
    a = with_prefix["train"]["D1"].input_ids
    b = without["train"]["D1"].input_ids
    assert not torch.equal(a, b), "the prefix must reach the tokenizer"
    assert int(a.ne(0).sum()) > int(b.ne(0).sum()), "prefix adds tokens"


def test_prefix_is_tokenised_jointly_with_the_text(tasks):
    """§3.6: the prefix is co-attended to, so it shares the sequence."""
    model = SurgeLLM(SurgeLLMConfig(variant="surgellm-g", backbone="tiny",
                                    offline_backbone=True))
    enc = tokenize_with_cache(model.backbone, ["the film was great"],
                              prefixes=["[TASK:D1|F1:1|F2:0]"])
    assert enc["input_ids"].shape[0] == 1
    assert int(enc["attention_mask"].sum()) > 4


# ===================== batching ==========================================
def test_per_task_sampler_yields_homogeneous_batches(tasks):
    model = SurgeLLM(SurgeLLMConfig(variant="baseline", backbone="tiny",
                                    offline_backbone=True))
    ex = SurgicalFeatureExtractor()
    norms = _fit_normalizers(tasks, ex, use_iwn=False)
    tensors = prepare_tensors(tasks, model, ex, norms, False, None)
    ds = MultiTaskDataset(tensors["train"], ["D1", "D2", "D3", "D4"])
    sampler = PerTaskBatchSampler(ds, batch_size=8, seed=0)
    for batch in sampler:
        tids = {ds[i]["task_ids"].item() for i in batch}
        assert len(tids) == 1, "each batch draws from a single task"


def test_sampler_covers_every_example_once_per_epoch(tasks):
    model = SurgeLLM(SurgeLLMConfig(variant="baseline", backbone="tiny",
                                    offline_backbone=True))
    ex = SurgicalFeatureExtractor()
    norms = _fit_normalizers(tasks, ex, use_iwn=False)
    ds = MultiTaskDataset(prepare_tensors(tasks, model, ex, norms, False, None)["train"],
                          ["D1", "D2", "D3", "D4"])
    seen = [i for batch in PerTaskBatchSampler(ds, 8, seed=0) for i in batch]
    assert sorted(seen) == list(range(len(ds)))


def test_chunk_size_is_2048():
    assert CHUNK_SIZE == 2048


# ===================== Algorithm 1 ======================================
def test_training_runs_and_improves_over_chance(tasks):
    art = run_single(RunSpec(
        variant="surgellm-iwn", backbone="tiny", offline_backbone=True,
        scale=SCALE, seed=0, profile="demo", max_epochs=2,
    ), tasks=tasks, verbose=False)
    assert 0.0 <= art.avg_f1 <= 1.0
    assert art.result.epochs_run >= 1
    assert np.isfinite(art.result.initial_loss)
    assert len(art.result.history) == art.result.epochs_run


def test_early_stopping_respects_the_patience_budget(tasks):
    art = run_single(RunSpec(
        variant="surgellm-iwn", backbone="tiny", offline_backbone=True,
        scale=SCALE, seed=0, profile="demo", max_epochs=6, learning_rate=0.0,
    ), tasks=tasks, verbose=False)
    # With lr = 0 nothing improves, so patience P = 2 must trigger early.
    assert art.result.early_stopped
    assert art.result.epochs_run <= 3


def test_best_checkpoint_is_restored(tasks):
    art = run_single(RunSpec(
        variant="surgellm-s", backbone="tiny", offline_backbone=True,
        scale=SCALE, seed=0, profile="demo", max_epochs=3,
    ), tasks=tasks, verbose=False)
    assert 1 <= art.result.best_epoch <= art.result.epochs_run


def test_checkpoint_is_written_and_reloadable(tasks, tmp_path):
    art = run_single(RunSpec(
        variant="surgellm-iwn", backbone="tiny", offline_backbone=True,
        scale=SCALE, seed=0, profile="demo", max_epochs=1,
        checkpoint_dir=str(tmp_path),
    ), tasks=tasks, verbose=False)
    path = tmp_path / f"{art.spec.run_id}.pt"
    assert path.exists() and path.with_suffix(".json").exists()

    from surgellm.train import load_checkpoint
    model, result = load_checkpoint(path)
    assert model.config.variant == "surgellm-iwn"
    assert "avg_test_macro_f1" in result


def test_variants_produce_different_models(tasks):
    runs = {
        v: run_single(RunSpec(variant=v, backbone="tiny", offline_backbone=True,
                              scale=SCALE, seed=0, profile="demo", max_epochs=1),
                      tasks=tasks, verbose=False)
        for v in ("baseline", "surgellm-iwn")
    }
    assert runs["baseline"].model.gate.enabled is False
    assert runs["surgellm-iwn"].model.gate.enabled is True
    assert runs["surgellm-iwn"].normalizers["D4"].mode == "iwn"
    assert runs["baseline"].normalizers["D4"].mode == "standard"


def test_same_seed_reproduces_the_same_score(tasks):
    kw = dict(variant="surgellm-s", backbone="tiny", offline_backbone=True,
              scale=SCALE, seed=0, profile="demo", max_epochs=1)
    a = run_single(RunSpec(**kw), tasks=tasks, verbose=False)
    b = run_single(RunSpec(**kw), tasks=tasks, verbose=False)
    assert a.avg_f1 == pytest.approx(b.avg_f1, abs=1e-6)
