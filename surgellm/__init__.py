"""SURGeLLM: task-aware feature gating with class-balanced normalization.

Reference implementation for

    Noor Islam S. Mohammad and Ulug Bayazit.
    "SURGeLLM: Rethinking Multi-Task Evaluation through Task-Aware Feature
    Gating with Class-Balanced Normalization."
    Proceedings of the 6th Workshop on Trustworthy NLP (TrustNLP 2026),
    pages 600-617.

Top-level entry points
----------------------
``surgellm.experiment.run_single``   train+evaluate one configuration
``surgellm.train.Trainer``           Algorithm 1
``surgellm.models.SurgeLLM``         the model (Eqs. 2-7, 12-14)
``surgellm.normalization``           IWN (Eqs. 8-11)
``surgellm.theory``                  Definition 1, Theorem 1, Proposition 2
"""

__version__ = "1.0.0"
__paper__ = "TrustNLP 2026, pages 600-617"

from .vocab import default_vocabulary, SurgicalVocabulary          # noqa: F401
from .features import SurgicalFeatureExtractor                      # noqa: F401
from .normalization import FeatureNormalizer                        # noqa: F401

__all__ = [
    "__version__", "default_vocabulary", "SurgicalVocabulary",
    "SurgicalFeatureExtractor", "FeatureNormalizer",
]
