"""Eq. (4) and Appendix D: feature extraction and the surgical vocabulary."""

from __future__ import annotations

import numpy as np
import pytest

from surgellm.features import SurgicalFeatureExtractor
from surgellm.vocab import (DEFAULT_GROUPS, FEATURE_DIM, GROUP_ORDER,
                            IndicatorGroup, SurgicalVocabulary,
                            default_vocabulary)


def test_vocabulary_has_ten_groups_and_16_dims():
    v = default_vocabulary()
    assert v.n_groups == 10, "Appendix D specifies ten indicator groups"
    assert FEATURE_DIM == 16, "s in R^16"
    assert SurgicalFeatureExtractor(v).dim == 16


def test_group_names_match_paper():
    assert GROUP_ORDER == [
        "sst_pos", "sst_neg", "llm_stat", "llm_formal", "llm_list",
        "human_pers", "human_hedge", "human_emo", "retrieval", "prompt_cot",
    ]


def test_prefix_matching_covers_inflectional_family():
    """``oscillat*`` matches oscillation / oscillates / oscillating."""
    g = IndicatorGroup("test", ["oscillat*"])
    for form in ("oscillation", "oscillates", "oscillating", "oscillate"):
        assert g.count(form, [form]) == 1, form
    assert g.count("pendulum", ["pendulum"]) == 0


def test_count_is_an_indicator_per_entry_not_a_term_frequency():
    """Eq. (4) sums ``1[w in x]`` over group members, so repeats don't stack."""
    g = IndicatorGroup("t", ["great", "superb"])
    assert g.count("great great great", ["great"] * 3) == 1
    assert g.count("great superb", ["great", "superb"]) == 2


def test_multiword_and_symbolic_entries_match():
    v = default_vocabulary()
    ex = SurgicalFeatureExtractor(v)
    i = v.group_names.index("retrieval")
    assert ex.extract("According to the report, x holds.")[i] >= 1
    j = v.group_names.index("llm_list")
    assert ex.extract("Points: (1) a (2) b (3) c")[j] >= 3


def test_surface_features_are_computed_on_untruncated_text():
    """The truncation-loss argument (§7.2.3) requires full-document stats."""
    ex = SurgicalFeatureExtractor()
    names = ex.feature_names()
    short = "a b c."
    long = short + " " + " ".join(["word"] * 500)
    s_short, s_long = ex.extract(short), ex.extract(long)
    wc = names.index("word_count")
    assert s_long[wc] > s_short[wc] + 400


def test_surface_feature_values():
    ex = SurgicalFeatureExtractor()
    n = ex.feature_names()
    s = ex.extract("Is this ok? Yes! Really great. 42")
    assert s[n.index("question_marks")] == 1
    assert s[n.index("exclamation_marks")] == 1
    assert s[n.index("has_digit")] == 1
    # Splitting on .!? yields four non-empty fragments: the trailing "42"
    # is unterminated but is still a sentence for counting purposes.
    assert s[n.index("sentence_count")] == 4


def test_sentence_count_ignores_empty_fragments():
    ex = SurgicalFeatureExtractor()
    i = ex.feature_names().index("sentence_count")
    assert ex.extract("One. Two. Three.")[i] == 3
    assert ex.extract("Wow!!! Really???")[i] == 2


def test_features_are_nonnegative_and_deterministic():
    ex = SurgicalFeatureExtractor()
    texts = ["Great film!", "Terrible, boring.", "According to the article, x."]
    a = ex.extract_batch(texts)
    b = ex.extract_batch(texts)
    assert np.all(a >= 0), "s in R^16_{>=0}"
    np.testing.assert_array_equal(a, b)


def test_empty_text_does_not_crash():
    s = SurgicalFeatureExtractor().extract("")
    assert s.shape == (16,)
    assert np.all(np.isfinite(s))


def test_prefix_string_format_matches_eq12():
    ex = SurgicalFeatureExtractor()
    s = ex.extract("Great movie, according to the critics! 10/10")
    p = ex.prefix_string("D1", s)
    assert p.startswith("[TASK:D1|F1:")
    assert p.endswith("]")
    assert p.count("|F") == 16, "sixteen feature slots"
    # Values are integer counts floor(s_j).
    for part in p.strip("[]").split("|")[1:]:
        int(part.split(":")[1])


def test_drop_and_subset_change_dimension():
    v = default_vocabulary()
    assert SurgicalFeatureExtractor(v.drop("retrieval")).dim == 15
    assert SurgicalFeatureExtractor(v.subset(GROUP_ORDER[:5])).dim == 11


def test_surface_only_and_groups_only_configurations():
    v = default_vocabulary()
    assert SurgicalFeatureExtractor(v, use_groups=False).dim == 6
    assert SurgicalFeatureExtractor(v, use_surface=False).dim == 10


def test_case_insensitivity():
    ex = SurgicalFeatureExtractor()
    i = default_vocabulary().group_names.index("sst_pos")
    assert ex.extract("GREAT")[i] == ex.extract("great")[i] == 1
