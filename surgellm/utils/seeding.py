"""Determinism controls (Reproducibility Checklist, Appendix I).

"All results from seeds {0, 1, 2}; data splits, weight initialization,
dropout masks, and CUDA determinism are seeded."
"""

from __future__ import annotations

import os
import random
from typing import Dict

import numpy as np

__all__ = ["seed_everything", "SEEDS", "determinism_report"]

#: The three seeds used throughout the paper.
SEEDS = (0, 1, 2)


def seed_everything(seed: int, deterministic: bool = True) -> int:
    """Seed Python, NumPy and torch; optionally force deterministic kernels."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except ImportError:  # pragma: no cover - torch is a hard dep in practice
        pass
    return seed


def determinism_report() -> Dict[str, object]:
    report: Dict[str, object] = {
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "numpy_version": np.__version__,
    }
    try:
        import torch
        report.update({
            "torch_version": torch.__version__,
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cuda_available": bool(torch.cuda.is_available()),
            "num_threads": int(torch.get_num_threads()),
        })
    except ImportError:
        report["torch_version"] = None
    return report
