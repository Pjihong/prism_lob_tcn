"""Shared experiment constants and compatibility defaults for Prism-LOB.

TRAINING.pool remains GAP for historical callers; current manuscript training
uses configs/paper/{toxic,vacuum}.args. The 70:15 weights define the split
inside the development file, not the proportions including the separate test
file. Inspect saved run configuration and checkpoint hashes when selecting
published results; these constants do not identify a particular trained model.
Inputs: protocol values and model dimensions.
Outputs: split checks, protocol metadata and analytic parameter counts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class DataProtocol:
    seq_len: int = 100
    n_features: int = 40
    train_fraction: float = 0.70
    valid_fraction: float = 0.15
    test_fraction: float = 0.15
    toxic_horizon: int = 10
    vacuum_horizon: int = 30
    detection_table_horizon: int = 10
    tick_size_z: float = 0.00098
    toxic_ticks: int = 3
    vacuum_volume_drop_z: float = 1.5
    vacuum_spread_ticks: int = 1

    @property
    def validation_fraction_within_development_file(self) -> float:
        """Normalize the 70:15 train/validation weights within the development file."""
        return self.valid_fraction / (self.train_fraction + self.valid_fraction)


@dataclass(frozen=True)
class TrainingProtocol:
    channels: tuple[int, ...] = (32, 32, 64)
    fc_hidden: tuple[int, ...] = (64,)
    kernel_size: int = 3
    dilation_base: int = 2
    dropout: float = 0.1
    pool: str = "avg"  # Compatibility default; paper presets select positional.
    loss: str = "ce_weighted"
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    max_epochs: int = 100
    early_stop_patience: int = 10
    checkpoint_metric: str = "val_f1"
    seeds: tuple[int, ...] = (1, 2, 3)

    @property
    def branch_temporal_receptive_field(self) -> int:
        """Local receptive field of one final branch timestep.

        Each TCN block contains two causal convolutions. Temporal readout subsequently
        aggregates temporal positions, but it does not enlarge this local
        receptive field.
        """
        dilations = (self.dilation_base ** i for i in range(len(self.channels)))
        return 1 + 2 * (self.kernel_size - 1) * sum(dilations)


@dataclass(frozen=True)
class CounterfactualProtocol:
    deletion_grid: tuple[float, ...] = tuple(i / 100 for i in range(5, 51, 5))
    pool_size: int = 30_000
    n_windows: int = 400
    mc_samples: int = 10
    # Deletion samples M donors from K context-nearest training windows.
    neighbor_candidates: int = 50
    cam_method: str = "hirescam"
    seeds: tuple[int, ...] = (1, 2, 3)

    @property
    def curve_grid(self) -> tuple[float, ...]:
        """Deletion grid plus the unmodified rho=0 anchor used for AUC."""
        return (0.0,) + self.deletion_grid


DATA = DataProtocol()
TRAINING = TrainingProtocol()
COUNTERFACTUAL = CounterfactualProtocol()


def ours_parameter_count(n_classes: int) -> int:
    """Analytic trainable-parameter count for the declared four-branch model."""
    if int(n_classes) <= 0:
        raise ValueError("n_classes must be positive")
    n_branches = 4
    in_channels = DATA.n_features // n_branches
    if in_channels * n_branches != DATA.n_features:
        raise ValueError("n_features must divide evenly across four branches")
    per_branch = 0
    prev = in_channels
    for out in TRAINING.channels:
        # Two biased temporal convolutions per residual block.
        per_branch += prev * out * TRAINING.kernel_size + out
        per_branch += out * out * TRAINING.kernel_size + out
        if prev != out:
            per_branch += prev * out + out  # biased 1x1 residual projection
        prev = out
    total = n_branches * per_branch
    head_in = n_branches * TRAINING.channels[-1]
    for hidden in TRAINING.fc_hidden:
        total += head_in * hidden + hidden
        head_in = hidden
    total += head_in * int(n_classes) + int(n_classes)
    return total


def branch_variant_parameter_count(
    n_classes: int,
    *,
    channels: Sequence[int] = TRAINING.channels,
    kernel_size: int = TRAINING.kernel_size,
    fc_hidden: Sequence[int] = TRAINING.fc_hidden,
    pool: str = TRAINING.pool,
    tcn_norm: str = "none",
    interaction_channels: int = 0,
    seq_len: int = DATA.seq_len,
    pool_windows: Sequence[int] = (8, 32, 100),
    n_features: int = DATA.n_features,
    n_branches: int = 4,
) -> int:
    """Analytic trainable-parameter count for opt-in branch-TCN variants.

    This covers identity input normalization, the residual TCN implemented in
    :mod:`lob_xai.models`, fixed/attention readouts, and the optional 1x1
    interaction stream.  DAIN/conditional-RevIN parameters are intentionally
    outside this contract and should be measured from the instantiated model.
    """
    n_classes = int(n_classes)
    channels = tuple(int(x) for x in channels)
    fc_hidden = tuple(int(x) for x in fc_hidden)
    n_features = int(n_features)
    n_branches = int(n_branches)
    kernel_size = int(kernel_size)
    interaction_channels = int(interaction_channels)
    seq_len = int(seq_len)
    pool_windows = tuple(int(x) for x in pool_windows)
    if n_classes < 2 or not channels or any(x <= 0 for x in channels):
        raise ValueError("invalid class count or channel configuration")
    if n_features <= 0 or n_branches <= 0 or n_features % n_branches:
        raise ValueError("n_features must divide evenly across branches")
    if kernel_size <= 0 or interaction_channels < 0 or seq_len <= 0:
        raise ValueError("invalid kernel or interaction width")
    if not pool_windows or any(x <= 0 for x in pool_windows):
        raise ValueError("pool windows must be positive")
    if len(set(pool_windows)) != len(pool_windows):
        raise ValueError("pool windows must be unique")
    norm = str(tcn_norm).lower()
    if norm not in {"none", "layer", "layernorm", "batch", "batchnorm"}:
        raise ValueError("unknown TCN normalization")

    prev = n_features // n_branches
    per_branch = 0
    for out in channels:
        per_branch += prev * out * kernel_size + out
        per_branch += out * out * kernel_size + out
        if prev != out:
            per_branch += prev * out + out
        if norm != "none":
            per_branch += 4 * out  # two norms, each affine weight + bias
        prev = out
    total = n_branches * per_branch

    readout_channels = n_branches * channels[-1]
    if interaction_channels:
        total += readout_channels * interaction_channels + interaction_channels
        if norm != "none":
            total += 2 * interaction_channels
        readout_channels += interaction_channels

    pool = str(pool).lower()
    if pool == "attention":
        total += readout_channels + 1
    elif pool in {"avg", "max"}:
        pass
    elif pool in {"avg_last", "avg+last"}:
        readout_channels *= 2
    elif pool == "multi_scale":
        if interaction_channels:
            raise ValueError("branch-local pooling forbids interaction_channels")
        total += n_branches * len(pool_windows)
    elif pool == "positional":
        if interaction_channels:
            raise ValueError("branch-local pooling forbids interaction_channels")
        total += n_branches * seq_len
    else:
        raise ValueError("unknown pooling strategy")

    for hidden in fc_hidden:
        if hidden <= 0:
            raise ValueError("hidden dimensions must be positive")
        total += readout_channels * hidden + hidden
        readout_channels = hidden
    total += readout_channels * n_classes + n_classes
    return total


def deeplob_parameter_count(n_classes: int) -> int:
    """Analytic count for ``models.deeplob.DeepLOB`` (40-feature stem)."""
    n_classes = int(n_classes)
    if n_classes < 2:
        raise ValueError("n_classes must be at least 2")
    convolution_bn_and_lstm = 143_712
    return convolution_bn_and_lstm + 65 * n_classes


def bin_ctabl_parameter_count(
    n_classes: int,
    *,
    seq_len: int = DATA.seq_len,
    n_features: int = DATA.n_features,
    d2: int = 60,
    t2: int = 10,
    d3: int = 120,
    t3: int = 5,
) -> int:
    """Analytic count for the default self-contained BiN-C(TABL)."""
    values = (n_classes, seq_len, n_features, d2, t2, d3, t3)
    if int(n_classes) < 2 or any(int(x) <= 0 for x in values[1:]):
        raise ValueError("invalid BiN-C(TABL) dimensions")
    n_classes = int(n_classes)
    seq_len, n_features, d2, t2, d3, t3 = map(
        int, (seq_len, n_features, d2, t2, d3, t3)
    )
    bin_params = 2 * n_features + 2 * seq_len + 2
    bl1 = d2 * n_features + seq_len * t2 + d2 * t2
    bl2 = d3 * d2 + t2 * t3 + d3 * t3
    tabl = n_classes * d3 + t3 * t3 + t3 + n_classes + 1
    return bin_params + bl1 + bl2 + tabl


def _tlob_layer_parameter_count(d: int, heads: int, final_dim: int) -> int:
    """Count one TLOB TransformerLayer, including its explicit Q/K/V maps."""
    d, heads, final_dim = int(d), int(heads), int(final_dim)
    return (
        4 * d * d * heads * heads
        + 4 * d * d * heads
        + 4 * d * d
        + 7 * d * heads
        + 7 * d
        + 4 * d * final_dim
        + final_dim
    )


def tlob_parameter_count(
    n_classes: int,
    *,
    seq_len: int = DATA.seq_len,
    n_features: int = DATA.n_features,
    hidden_dim: int = 128,
    num_layers: int = 4,
    num_heads: int = 1,
    sinusoidal_embedding: bool = True,
) -> int:
    """Analytic count for ``models.tlob.TLOB``."""
    n_classes, seq_len, n_features = map(
        int, (n_classes, seq_len, n_features)
    )
    hidden_dim, num_layers, num_heads = map(
        int, (hidden_dim, num_layers, num_heads)
    )
    if n_classes < 2 or min(seq_len, hidden_dim) < 4:
        raise ValueError("invalid TLOB class or representation dimensions")
    if num_layers < 1 or num_heads < 1:
        raise ValueError("TLOB layers and heads must be positive")
    if sinusoidal_embedding and hidden_dim % 2:
        raise ValueError("sinusoidal TLOB embedding requires even hidden_dim")

    total = 2 * n_features + 2 * seq_len + 2  # BiN
    total += n_features * hidden_dim + hidden_dim  # input embedding
    if not sinusoidal_embedding:
        total += seq_len * hidden_dim
    regular_pair = (
        _tlob_layer_parameter_count(hidden_dim, num_heads, hidden_dim)
        + _tlob_layer_parameter_count(seq_len, num_heads, seq_len)
    )
    total += (num_layers - 1) * regular_pair
    total += _tlob_layer_parameter_count(
        hidden_dim, num_heads, hidden_dim // 4
    )
    total += _tlob_layer_parameter_count(
        seq_len, num_heads, seq_len // 4
    )

    head_dim = (hidden_dim // 4) * (seq_len // 4)
    while head_dim > 128:
        next_dim = head_dim // 4
        total += head_dim * next_dim + next_dim
        head_dim = next_dim
    total += head_dim * n_classes + n_classes
    return total


def baseline_parameter_count(model: str, n_classes: int, **kwargs) -> int:
    """Dispatch the manuscript baseline parameter-count contract."""
    if model == "deeplob":
        return deeplob_parameter_count(n_classes)
    if model == "tlob":
        return tlob_parameter_count(n_classes, **kwargs)
    if model == "bin_ctabl":
        return bin_ctabl_parameter_count(n_classes, **kwargs)
    raise ValueError(f"unknown baseline model: {model!r}")


def parse_boundaries(value: str | Sequence[int] | None) -> list[int] | None:
    """Parse comma-separated internal segment boundaries used by both trainers."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        items: Iterable[str | int] = value.split(",")
    else:
        items = value
    result = sorted({int(x) for x in items if str(x).strip()})
    if any(x <= 0 for x in result):
        raise ValueError("segment boundaries must be positive row indices")
    return result


def split_fraction_report(n_train: int, n_valid: int, n_test: int) -> dict[str, float]:
    total = n_train + n_valid + n_test
    if total <= 0:
        raise ValueError("split sizes must contain at least one observation")
    return {
        "train": n_train / total,
        "valid": n_valid / total,
        "test": n_test / total,
    }


def assert_paper_split(n_train: int, n_valid: int, n_test: int, tolerance: float = 0.02) -> None:
    """Fail when realised split sizes contradict the declared file-based split.

    The provided FI-2010 files are a development CSV (362,400 rows) plus a
    separate test CSV (31,937 rows), so a combined-row 70/15/15 split is not
    reconstructible from them; the realized overall fractions are
    75.7/16.2/8.1.  Every frozen run in this project (devsel, poolab, donor
    pools) was produced under exactly that reality, so the enforceable
    contract is:

    1. the chronological within-development split matches
       ``DATA.validation_fraction_within_development_file`` (= 0.15/0.85);
    2. a non-empty held-out test file exists.

    The manuscript must therefore declare the split as "development file
    split chronologically into train/validation at 82.35/17.65, with the
    separate FI-2010 test file held out", NOT as a combined 70/15/15.
    """
    if min(n_train, n_valid, n_test) < 0 or (n_train + n_valid) <= 0:
        raise ValueError(
            "split sizes must be non-negative with a non-empty development part"
        )
    if n_test <= 0:
        raise ValueError(
            "realised split contradicts the paper protocol: the held-out "
            "test file is empty"
        )
    got_val = n_valid / (n_train + n_valid)
    expected_val = DATA.validation_fraction_within_development_file
    if abs(got_val - expected_val) > tolerance:
        raise ValueError(
            "realised split contradicts the paper protocol: "
            f"validation share within the development file is {got_val:.4f} "
            f"(expected {expected_val:.4f} ± {tolerance})"
        )


def as_dict() -> dict:
    training = asdict(TRAINING)
    training["branch_temporal_receptive_field"] = (
        TRAINING.branch_temporal_receptive_field
    )
    training["expected_parameter_counts"] = {
        "toxic_3class": ours_parameter_count(3),
        "vacuum_2class": ours_parameter_count(2),
        "deeplob_3class": deeplob_parameter_count(3),
        "deeplob_2class": deeplob_parameter_count(2),
        "tlob_3class": tlob_parameter_count(3),
        "tlob_2class": tlob_parameter_count(2),
        "bin_ctabl_3class": bin_ctabl_parameter_count(3),
        "bin_ctabl_2class": bin_ctabl_parameter_count(2),
    }
    return {
        "data": asdict(DATA),
        "training": training,
        "counterfactual": asdict(COUNTERFACTUAL),
    }
