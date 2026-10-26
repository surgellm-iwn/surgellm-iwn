"""§4-§5: corpora, splits, metrics, significance, auto-extraction, Alg. 1."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from surgellm.autoextract import (auto_extract_vocabulary, monroe_log_odds,
                                  random_control_vocabulary)
from surgellm.data.registry import load_all_tasks, load_task
from surgellm.data.schema import TASKS, TASK_ORDER, TOTAL_EXAMPLES, Corpus
from surgellm.data.splits import stratified_cap, stratified_split
from surgellm.eval.failure_analysis import (analyse_failures, efficiency_row,
                                            pareto_frontier)
from surgellm.eval.metrics import (aggregate_metrics, classification_metrics,
                                   macro_f1, per_class_metrics)
from surgellm.eval.significance import (benjamini_hochberg, bootstrap_ci,
                                        paired_comparison_table, welch_t_test)
from surgellm.utils.seeding import SEEDS, seed_everything
from surgellm.vocab import default_vocabulary


# ===================== Table 2 ===========================================
def test_task_table_matches_paper():
    assert TOTAL_EXAMPLES == 17_830
    assert TASKS["D1"].capped_n == 7_666 and TASKS["D1"].source == "SST-2"
    assert TASKS["D2"].capped_n == 2_000 and TASKS["D2"].source == "HotPotQA"
    assert TASKS["D3"].capped_n == 3_164 and TASKS["D3"].source == "LLM-7"
    assert TASKS["D4"].capped_n == 5_000 and TASKS["D4"].source == "HumLLM"
    assert TASKS["D4"].raw_skew == 9.3, "the authorship corpus is 9.3:1"
    assert all(t.n_classes == 2 for t in TASKS.values())


# ===================== splits ============================================
def _corpus(n=1000, skew=9.0, seed=0):
    rng = np.random.default_rng(seed)
    p = skew / (skew + 1)
    y = (rng.random(n) > p).astype(np.int64)
    return Corpus("D4", [f"doc {i}" for i in range(n)], y, 2)


def test_stratified_cap_hits_the_requested_balance():
    """D4 is capped 50/50 from a 9.3:1 pool, so the minority must suffice."""
    c = stratified_cap(_corpus(n=4000), 200, seed=0, target_priors=[0.5, 0.5])
    counts = c.class_counts()
    assert len(c) == 200
    assert abs(counts[0] - counts[1]) <= 1, "50/50 cap"


def test_stratified_cap_clips_when_the_minority_is_exhausted():
    """Asking for more minority examples than exist cannot invent them.

    The shortfall is reallocated to the available classes and the total is
    still honoured, rather than silently returning a short corpus.
    """
    src = _corpus(n=1000, skew=9.0)            # ~100 minority examples
    minority = int(src.class_counts().min())
    c = stratified_cap(src, 400, seed=0, target_priors=[0.5, 0.5])
    assert len(c) == 400
    assert int(c.class_counts().min()) == minority, "all minority examples used"


def test_stratified_cap_preserves_balance_when_unspecified():
    src = _corpus(skew=4.0)
    c = stratified_cap(src, 300, seed=0)
    np.testing.assert_allclose(c.class_priors(), src.class_priors(), atol=0.03)


def test_split_is_70_15_15_and_stratified():
    src = stratified_cap(_corpus(n=2000), 1000, seed=0, target_priors=[0.5, 0.5])
    b = stratified_split(src, seed=0)
    sizes = b.sizes()
    assert sizes["train"] + sizes["val"] + sizes["test"] == len(src)
    assert abs(sizes["train"] / len(src) - 0.70) < 0.02
    assert abs(sizes["val"] / len(src) - 0.15) < 0.02
    for part in (b.train, b.val, b.test):
        np.testing.assert_allclose(part.class_priors(), src.class_priors(), atol=0.05)


def test_splits_are_disjoint_and_reproducible():
    src = _corpus(n=500)
    a = stratified_split(src, seed=3)
    b = stratified_split(src, seed=3)
    assert a.train.texts == b.train.texts
    assert not (set(a.train.texts) & set(a.test.texts))


# ===================== synthetic provider ================================
def test_synthetic_d4_pool_reproduces_the_paper_skew():
    td = load_task("D4", provider="synthetic", seed=0, scale=0.05)
    assert 8.0 < td.normalizer_pool.skew() < 11.0, "pool should be ~9.3:1"
    assert abs(td.splits.train.minority_pct() - 50.0) < 6.0, "capped to ~50/50"


def test_all_tasks_load_and_are_nonempty():
    tasks = load_all_tasks(provider="synthetic", seed=0, scale=0.03)
    assert set(tasks) == set(TASK_ORDER)
    for td in tasks.values():
        assert len(td.splits.train) > 0 and len(td.splits.test) > 0


def test_surgical_features_carry_class_signal_in_the_surrogates():
    """Otherwise the ablations would be measuring nothing."""
    from surgellm.features import SurgicalFeatureExtractor
    ex = SurgicalFeatureExtractor()
    td = load_task("D4", provider="synthetic", seed=0, scale=0.05)
    s = ex.extract_batch(td.splits.train.texts)
    y = td.splits.train.labels
    gap = np.abs(s[y == 0].mean(0) - s[y == 1].mean(0))
    scale = s.std(0) + 1e-6
    assert float((gap / scale).max()) > 0.5


# ===================== metrics ===========================================
def test_macro_f1_on_a_perfect_classifier():
    y = np.array([0, 1, 0, 1])
    assert macro_f1(y, y, 2) == pytest.approx(1.0)


def test_macro_f1_penalises_majority_collapse():
    """Exactly the pathology IWN fixes: predicting only the majority."""
    y = np.array([0] * 90 + [1] * 10)
    pred = np.zeros(100, dtype=int)
    m = classification_metrics(y, pred, n_classes=2)
    assert m["accuracy"] == pytest.approx(0.90)
    assert m["macro_f1"] < 0.5, "accuracy hides it, macro-F1 does not"


def test_per_class_precision_recall_are_correct():
    y = np.array([0, 0, 1, 1])
    p = np.array([0, 1, 1, 1])
    pc = per_class_metrics(y, p, 2)
    assert pc["precision"][1] == pytest.approx(2 / 3)
    assert pc["recall"][1] == pytest.approx(1.0)
    assert pc["recall"][0] == pytest.approx(0.5)


def test_roc_auc_is_rank_based_and_handles_ties():
    y = np.array([0, 0, 1, 1])
    perfect = np.stack([1 - np.array([.1, .2, .8, .9]), [.1, .2, .8, .9]], 1)
    assert classification_metrics(y, y, perfect, 2)["roc_auc"] == pytest.approx(1.0)
    tied = np.full((4, 2), 0.5)
    assert classification_metrics(y, y, tied, 2)["roc_auc"] == pytest.approx(0.5)


def test_class_names_appear_in_the_metric_keys():
    y = np.array([0, 1])
    m = classification_metrics(y, y, n_classes=2, class_names=("human", "llm"))
    assert "precision_human" in m and "recall_llm" in m


def test_aggregate_is_the_unweighted_task_mean():
    assert aggregate_metrics({"a": {"macro_f1": 0.8}, "b": {"macro_f1": 0.9}}) \
        == pytest.approx(0.85)


# ===================== significance ======================================
def test_welch_detects_a_real_difference_and_ignores_a_null():
    t, df, p = welch_t_test([0.94, 0.937, 0.94], [0.76, 0.762, 0.758])
    assert p < 0.01 and t > 0
    _, _, p_null = welch_t_test([0.90, 0.91, 0.89], [0.90, 0.905, 0.895])
    assert p_null > 0.05


def test_welch_handles_identical_constant_samples():
    t, df, p = welch_t_test([0.9, 0.9, 0.9], [0.9, 0.9, 0.9])
    assert p == pytest.approx(1.0)


def test_benjamini_hochberg_is_monotone_and_at_least_the_raw_p():
    p = [0.001, 0.01, 0.03, 0.2, 0.5]
    adj, rej = benjamini_hochberg(p, q=0.05)
    assert list(adj) == sorted(adj), "adjusted p-values must be monotone"
    assert all(a >= r - 1e-12 for a, r in zip(adj, p))
    assert rej[0] and not rej[-1]


def test_bh_is_less_conservative_than_bonferroni():
    p = [0.01] * 5
    adj, _ = benjamini_hochberg(p, q=0.05)
    assert float(np.max(adj)) < 5 * 0.01 + 1e-9


def test_bootstrap_ci_brackets_the_mean():
    vals = [0.940, 0.937, 0.943]
    lo, hi, width = bootstrap_ci(vals, B=2000, seed=0)
    assert lo <= float(np.mean(vals)) <= hi and width > 0


def test_comparison_table_builds_one_bh_family():
    per_seed = {
        "A": {"D1": [0.9, 0.91, 0.89], "D4": [0.89, 0.892, 0.893]},
        "B": {"D1": [0.9, 0.905, 0.895], "D4": [0.76, 0.762, 0.758]},
    }
    rows = paired_comparison_table(per_seed, [("A", "B")], ["D1", "D4"])
    assert len(rows) == 2
    d4 = next(r for r in rows if r["task"] == "D4")
    assert d4["significant"] and d4["delta"] > 0.1


# ===================== auto-extraction (Appendix E) ======================
def test_log_odds_separates_class_marker_words():
    texts = (["wonderful superb excellent film story"] * 60
             + ["terrible awful dreadful film story"] * 60)
    y = np.array([1] * 60 + [0] * 60)
    scored = monroe_log_odds(texts, y, 2)
    top1 = {w for w, _ in scored[1][:4]}
    assert top1 & {"wonderful", "superb", "excellent"}
    assert "film" not in {w for w, _ in scored[1][:2]}, "shared words rank low"


def test_auto_extraction_returns_ten_groups():
    texts = (["wonderful superb excellent film"] * 40
             + ["terrible awful dreadful movie"] * 40)
    y = np.array([1] * 40 + [0] * 40)
    rep = auto_extract_vocabulary({"D1": texts}, {"D1": y}, {"D1": 2},
                                  seed=0, use_sbert=False)
    assert rep.vocabulary.n_groups == 10, "k-means with k = 10"
    assert rep.n_candidates > 0
    assert "ppmi" in rep.embedding_backend or "sbert" in rep.embedding_backend


def test_random_control_preserves_group_cardinality():
    curated = default_vocabulary()
    texts = ["alpha beta gamma delta epsilon zeta eta theta iota kappa"] * 50
    rand = random_control_vocabulary(texts, curated, seed=0)
    assert rand.n_groups == curated.n_groups
    for a, b in zip(rand.groups, curated.groups):
        assert a.cardinality() <= b.cardinality()
    curated_terms = {t for g in curated.groups for t in g.terms}
    random_terms = {t for g in rand.groups for t in g.terms}
    assert not (curated_terms & random_terms), "the control must not be curated"


# ===================== failure analysis / efficiency =====================
def test_failure_modes_are_detected_for_d4_short_texts():
    texts = ["short text"] * 10
    y = np.zeros(10, dtype=int)
    pred = np.ones(10, dtype=int)
    rep = analyse_failures("D4", texts, y, pred, n_inspect=10)
    assert rep["n_errors"] == 10
    assert "short_text_unreliable_counts" in rep["mode_counts"]


def test_negation_mode_detected_on_d1():
    rep = analyse_failures("D1", ["this was not bad at all"], np.array([1]),
                           np.array([0]), n_inspect=5)
    assert "negation_scope" in rep["mode_counts"]


def test_efficiency_formulas():
    r = efficiency_row("m", 125_000_000, 332.0, 0.940, baseline_seconds=82.0)
    # Values are rounded for the CSV, so compare at the emitted precision.
    assert r["f1_per_min"] == pytest.approx(round(0.940 * 60 / 332, 3))
    assert r["efficiency"] == pytest.approx(
        round(0.940 * 1e3 / math.log10(125_000_000), 1))
    assert r["overhead_x"] == pytest.approx(4.0, abs=0.05)


def test_pareto_frontier_marks_dominant_points():
    rows = [
        {"model": "cheap", "train_seconds": 80, "avg_f1": 0.886},
        {"model": "dominated", "train_seconds": 120, "avg_f1": 0.870},
        {"model": "best", "train_seconds": 330, "avg_f1": 0.940},
    ]
    out = {r["model"]: r["pareto_efficient"] for r in pareto_frontier(rows)}
    assert out["cheap"] and out["best"] and not out["dominated"]


# ===================== determinism =======================================
def test_seeds_are_zero_one_two():
    assert SEEDS == (0, 1, 2)


def test_seeding_makes_torch_reproducible():
    seed_everything(7)
    a = torch.randn(5)
    seed_everything(7)
    torch.testing.assert_close(a, torch.randn(5))
