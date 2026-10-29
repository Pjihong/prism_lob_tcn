"""Reproducibility controls shared by paper training entry points."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def configure_determinism(seed: int) -> None:
    """Seed all RNGs and request deterministic PyTorch kernels.

    ``warn_only=True`` keeps the run usable if an installed PyTorch/backend has
    no deterministic implementation, while making the exception visible in
    logs.  Exact reproducibility still requires recording the environment.
    """
    seed = int(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)

