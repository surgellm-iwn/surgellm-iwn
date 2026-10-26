"""§3.5: the four claimed properties of Instance-Weighted Normalization.

Each test corresponds to a sentence in the paper, so a failure here is a
failure of a stated property, not of an implementation detail.
"""

from __future__ import annotations

import numpy as np
import pytest

from surgellm.normalization import FeatureNormalizer, prior_contamination


def _skewed(n=4000, d=6, skew=9.3, seed=0):
    """A corpus with majority class 0 whose per-class moments differ."""
    rng = np.random.default_rng(seed)
    p = skew / (skew + 1.0)
    y = (rng.random(n) > p).astype(int)
    s = np.empty((n, d))
    s[y == 0] = rng.normal(1.0, 0.5, size=((y == 0).sum(), d))
    s[y == 1] = rng.normal(5.0, 2.0, size=((y == 1).sum(), d))
    return np.clip(s, 0, None), y


def _balanced(n=4000, d=6, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) > 0.5).astype(int)
    s = np.empty((n, d))
    s[y == 0] = rng.normal(2.0, 1.0, size=((y == 0).sum(), d))
    s[y == 1] = rng.normal(3.0, 1.0, size=((y == 1).sum(), d))
    return np.clip(s, 0, None), y


# -- property 1: parameter-free -------------------------------------------
def test_iwn_introduces_no_learnable_parameters():
    s, y = _skewed()
    n = FeatureNormalizer("iwn").fit(s, y, 2)
    assert n.n_parameters == 0
    assert FeatureNormalizer("standard").fit(s, y, 2).n_parameters == 0


# -- property 2: test-time class-agnostic ---------------------------------
def test_transform_requires_no_labels():
    s, y = _skewed()
    n = FeatureNormalizer("iwn").fit(s, y, 2)
    out = n.transform(s[:10])            # no labels passed
    assert out.shape == (10, s.shape[1])
    assert np.all(np.isfinite(out))


def test_transform_is_a_pure_affine_map_of_the_input():
    s, y = _skewed()
    n = FeatureNormalizer("iwn").fit(s, y, 2)
    # Same input -> same output, regardless of what else is in the batch.
    a = n.transform(s[:5])
    b = n.transform(s[:50])[:5]
    np.testing.assert_allclose(a, b, rtol=1e-6)


# -- property 3: reduces to standard normalization on balanced corpora -----
def test_iwn_matches_standard_on_a_balanced_corpus():
    s, y = _balanced()
    std = FeatureNormalizer("standard").fit(s, y, 2)
    iwn = FeatureNormalizer("iwn").fit(s, y, 2)
    # Means coincide closely; scales differ only by the weighted vs
    # unweighted variance estimator, as the paper states.
    np.testing.assert_allclose(std.stats.mean, iwn.stats.mean, atol=0.05)
    assert np.all(iwn.stats.std > 0)
    assert np.allclose(std.stats.std, iwn.stats.std, rtol=0.35)


def test_iwn_differs_from_standard_under_skew():
    s, y = _skewed()
    std = FeatureNormalizer("standard").fit(s, y, 2)
    iwn = FeatureNormalizer("iwn").fit(s, y, 2)
    assert np.linalg.norm(std.stats.mean - iwn.stats.mean) > 1.0, \
        "Eq. (9): the marginal mean is dragged to the majority class"


# -- Eq. (9): prior contamination -----------------------------------------
def test_marginal_mean_approaches_majority_mean_as_skew_grows():
    gaps = []
    for skew in (1.0, 3.0, 9.3, 50.0):
        s, y = _skewed(skew=skew, seed=1)
        n = FeatureNormalizer("standard").fit(s, y, 2)
        marg = n.stats.mean
        majority = n.stats.per_class_mean[0]
        gaps.append(float(np.linalg.norm(marg - majority)))
    assert gaps == sorted(gaps, reverse=True), \
        "the marginal mean should converge to the majority mean as pi -> 1"


def test_contamination_metric_grows_with_skew():
    low = prior_contamination(FeatureNormalizer("standard").fit(*_balanced(), 2))
    high = prior_contamination(FeatureNormalizer("standard").fit(*_skewed(), 2))
    assert high["normalized_mean_gap"] > low["normalized_mean_gap"]
    assert high["majority_prior"] > 0.85


# -- Eq. (10)/(11) correctness --------------------------------------------
def test_balanced_moments_are_the_unweighted_class_averages():
    s, y = _skewed()
    n = FeatureNormalizer("iwn").fit(s, y, 2)
    expected_mean = np.stack([s[y == 0].mean(0), s[y == 1].mean(0)]).mean(0)
    expected_std = np.stack([s[y == 0].std(0), s[y == 1].std(0)]).mean(0)
    np.testing.assert_allclose(n.stats.mean, expected_mean, rtol=1e-10)
    np.testing.assert_allclose(n.stats.std, expected_std, rtol=1e-10)


def test_standard_moments_are_the_marginal_ones():
    s, y = _skewed()
    n = FeatureNormalizer("standard").fit(s, y, 2)
    np.testing.assert_allclose(n.stats.mean, s.mean(0), rtol=1e-10)
    np.testing.assert_allclose(n.stats.std, s.std(0), rtol=1e-10)


def test_iwn_centres_the_minority_class_closer_to_zero():
    """The mechanism: under Eq. (8) the minority sits far out in the tail."""
    s, y = _skewed()
    z_std = FeatureNormalizer("standard").fit(s, y, 2).transform(s)
    z_iwn = FeatureNormalizer("iwn").fit(s, y, 2).transform(s)
    minority_std = float(np.abs(z_std[y == 1]).mean())
    minority_iwn = float(np.abs(z_iwn[y == 1]).mean())
    assert minority_iwn < minority_std


# -- robustness ------------------------------------------------------------
def test_iwn_without_labels_raises():
    s, _ = _skewed()
    with pytest.raises(ValueError):
        FeatureNormalizer("iwn").fit(s)


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        FeatureNormalizer("magic")


def test_zero_variance_dimension_is_safe():
    s, y = _skewed()
    s[:, 0] = 3.0                                 # constant column
    for mode in ("standard", "iwn"):
        z = FeatureNormalizer(mode).fit(s, y, 2).transform(s)
        assert np.all(np.isfinite(z))
        assert np.allclose(z[:, 0], 0.0, atol=1e-3)


def test_state_dict_round_trip():
    s, y = _skewed()
    n = FeatureNormalizer("iwn").fit(s, y, 2)
    m = FeatureNormalizer.from_state_dict(n.state_dict())
    np.testing.assert_allclose(n.transform(s[:20]), m.transform(s[:20]), rtol=1e-6)
