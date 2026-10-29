"""Configuration for FI-2010 training and attribution.

Config retains GAP defaults for compatibility with historical experiments.
The current manuscript uses positional readout and identity input normalization;
use configs/paper/{toxic,vacuum}.args with scripts/train_microstructure.py.
Inputs: data paths and model/training options.
Outputs: a configuration object; no data is loaded at import time.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List
import os

from .paper_protocol import DATA, TRAINING


# ───────────────────────────────────────────────────────────────────
# 출력 디렉토리
# ───────────────────────────────────────────────────────────────────
FIGURES_DIR = os.environ.get("LOB_FIGURES_DIR", "outputs/lob_figures")


# ───────────────────────────────────────────────────────────────────
# 메인 Config
# ───────────────────────────────────────────────────────────────────

@dataclass
class Config:
    """FI-2010 학습 + 분석 통합 설정."""

    # ── 데이터 ────────────────────────────────────────────────
    train_csv: str = "data/FI2010/FI2010_train.csv"
    test_csv:  str = "data/FI2010/FI2010_test.csv"
    n_features: int = DATA.n_features  # 0-39 raw LOB만 사용
    n_classes: int = 3              # up / flat / down

    # ── 라벨 / 윈도우 ─────────────────────────────────────────
    horizon_idx: int = 4            # k = 1,2,3,5,10 중 4번 인덱스 = k=10
    seq_len: int = DATA.seq_len      # FI-2010 표준 윈도우 길이

    # ── Branching scheme ──────────────────────────
    # one of: type4 (default, original 4-branch), topdeep8, topmiddeep12,
    # level10, feature40, custom
    branching_scheme: str = "type4"
    custom_branch_indices: Optional[Dict[str, List[int]]] = None

    # ── 모델 backbone ─────────────────────────────────────────
    backbone: str = "tcn"           # paper model: dilated temporal CNN
    pool: str = TRAINING.pool        # legacy GAP default; paper preset selects positional
    pool_windows: Tuple[int, ...] = (8, 32, 100)
    n_branches: int = 4             # ★ kept for backward compat; auto-overridden
                                    #    by len(branching_scheme) at build time
    cnn_channels: Tuple[int, ...] = TRAINING.channels
    cnn_kernel: int = TRAINING.kernel_size
    cnn_dropout: float = TRAINING.dropout
    fc_hidden: Tuple[int, ...] = TRAINING.fc_hidden
    tcn_dilation_base: int = TRAINING.dilation_base

    # ── Branch-aware input normalization ───────────────
    # one of: "identity" (no-op, default — preserves original behavior),
    #         "dain"     (per-branch adaptive shift/scale/gate, Passalis 2019),
    #         "revin_c"  (per-branch conditional RevIN normalization step;
    #                     denormalize omitted because task is classification).
    input_norm: str = "identity"

    # ── 학습 ──────────────────────────────────────────────────
    batch_size: int = TRAINING.batch_size
    epochs: int = TRAINING.max_epochs
    lr: float = TRAINING.learning_rate
    weight_decay: float = TRAINING.weight_decay
    val_ratio: float = DATA.validation_fraction_within_development_file
    early_stop_patience: int = TRAINING.early_stop_patience

    # ── 분석 (downstream) ─────────────────────────────────────
    target_class: str = "predicted"
    n_perm: int = 2000
    min_gap: int = 50
    n_subsample_analyze: int = 5000

    # ── 기타 ──────────────────────────────────────────────────
    out_dir: str = "outputs"
    seed: int = TRAINING.seeds[0]
    device: str = "auto"
    num_workers: int = 0
