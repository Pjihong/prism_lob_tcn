"""Branch classifiers with temporal encoders and configurable readouts.

The manuscript model uses four semantic TCN branches, input-independent
positional readout and a shared classification head. GAP remains the constructor
default for compatibility. Attention, max, avg_last and multi_scale support
historical experiments and controlled comparisons.
Inputs: tensors shaped (batch, time, features).
Outputs: class logits shaped (batch, n_classes).
Scope: branch identity is preserved before the head; the head can mix branches.
"""
from __future__ import annotations
from typing import List, Optional, Sequence, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .input_norm import make_input_norm
from .paper_protocol import TRAINING


# ───────────────────────────────────────────────────────────────────
# Building blocks
# ───────────────────────────────────────────────────────────────────


class ChannelLayerNorm1D(nn.Module):
    """LayerNorm over channels independently at every timestep."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


def make_tcn_norm(kind: str, channels: int) -> nn.Module:
    """Build an opt-in internal normalization layer for a TCN block."""
    kind = str(kind).lower()
    if kind == "none":
        return nn.Identity()
    if kind in {"layer", "layernorm"}:
        return ChannelLayerNorm1D(channels)
    if kind in {"batch", "batchnorm"}:
        return nn.BatchNorm1d(channels)
    raise ValueError(
        f"unknown TCN norm: {kind!r}. Use 'none'|'layer'|'batch'."
    )


class CNNBranch(nn.Module):
    """1D CNN block — preserves time length via asymmetric padding."""

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernel_size: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._left_pad = kernel_size // 2
        self._right_pad = (kernel_size - 1) - self._left_pad
        self.kernel_size = kernel_size

        self.convs = nn.ModuleList()
        prev = in_channels
        for c in channels:
            self.convs.append(nn.Conv1d(prev, c, kernel_size=kernel_size, padding=0))
            prev = c
        self.act = nn.LeakyReLU(0.01, inplace=False)
        self.dropout = nn.Dropout(dropout)
        self.out_channels = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = F.pad(x, (self._left_pad, self._right_pad))
            x = conv(x)
            x = self.act(x)
            x = self.dropout(x)
        return x


class TCNBlock(nn.Module):
    """
    Single Temporal Conv block (Bai et al. 2018 style).

    Causal-padded dilated conv → activation → dropout → conv → ... + residual.
    Receptive field doubles per block when dilation_base = 2.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
        norm: str = "none",
    ):
        super().__init__()
        # Causal pad: (k-1)*dilation on the left only — past-only conv
        self.causal_pad = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(in_channels, out_channels,
                                kernel_size=kernel_size, dilation=dilation, padding=0)
        self.conv2 = nn.Conv1d(out_channels, out_channels,
                                kernel_size=kernel_size, dilation=dilation, padding=0)
        self.norm1 = make_tcn_norm(norm, out_channels)
        self.norm2 = make_tcn_norm(norm, out_channels)
        self.act = nn.LeakyReLU(0.01, inplace=False)
        self.dropout = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels else None
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.downsample is None else self.downsample(x)
        h = F.pad(x, (self.causal_pad, 0))
        h = self.conv1(h)
        h = self.norm1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = F.pad(h, (self.causal_pad, 0))
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        return self.act(h + residual)


class TCNBranch(nn.Module):
    """
    Temporal Convolutional Network branch — exponentially growing receptive field.

    Each block has two causal convolutions. With the paper configuration
    ``channels=(32,32,64)``, ``k=3``, and dilations ``1,2,4``, a final
    timestep has receptive field ``1 + 2*(k-1)*(1+2+4) = 29``. Global
    average pooling still lets the classifier aggregate representations from
    every position in the 100-step window; it does not make any one temporal
    representation see all 100 steps.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernel_size: int = 3,
        dropout: float = 0.1,
        dilation_base: int = 2,
        norm: str = "none",
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        prev = in_channels
        for i, c in enumerate(channels):
            dilation = dilation_base ** i
            self.blocks.append(TCNBlock(prev, c, kernel_size=kernel_size,
                                         dilation=dilation, dropout=dropout,
                                         norm=norm))
            prev = c
        self.out_channels = prev

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


# ───────────────────────────────────────────────────────────────────
# Pool strategies
# ───────────────────────────────────────────────────────────────────


class AttentionPool1D(nn.Module):
    """Soft attention pool over time (LEARNED weights — can collapse)."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.out_channels = int(in_channels)
        self.score = nn.Conv1d(in_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.score(x)
        a = F.softmax(s, dim=-1)
        return (x * a).sum(dim=-1)


class AvgPool1D(nn.Module):
    """Global average pool — every timestep contributes equally (FIXED 1/T)."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.out_channels = int(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=-1)


class MaxPool1D(nn.Module):
    """Global max pool — picks the single most-activated timestep per channel."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.out_channels = int(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.max(dim=-1).values


class AvgLastPool1D(nn.Module):
    """Fixed dual readout: concatenate the temporal mean and final state.

    The mean retains the paper model's distributed evidence path, while the
    last state gives the classifier direct access to the representation aligned
    with the future-event label.  No learned temporal weights are introduced.
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.out_channels = 2 * int(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((x.mean(dim=-1), x[..., -1]), dim=1)


class MultiScaleTailPool1D(nn.Module):
    """Convex mixture of fixed tail averages.

    The default windows ``(8, 32, 100)`` expose short, medium and full-window
    evidence without making the weights input-dependent.  Only the mixture
    logits are learned.  Uniform-logit initialization gives every scale equal
    prior mass and adds exactly ``len(windows)`` parameters per branch.
    """

    def __init__(self, in_channels: int,
                 windows: Sequence[int] = (8, 32, 100)):
        super().__init__()
        parsed = tuple(int(w) for w in windows)
        if not parsed or any(w <= 0 for w in parsed):
            raise ValueError("multi-scale windows must be positive")
        if len(set(parsed)) != len(parsed):
            raise ValueError("multi-scale windows must be unique")
        self.out_channels = int(in_channels)
        self.windows = parsed
        self.logits = nn.Parameter(torch.zeros(len(parsed)))

    def temporal_weights(self, length: int) -> torch.Tensor:
        """Return the effective length-``T`` linear pooling weights."""
        length = int(length)
        if length <= 0:
            raise ValueError("temporal length must be positive")
        mixture = F.softmax(self.logits, dim=0)
        components = []
        for window in self.windows:
            width = min(int(window), length)
            prefix = mixture.new_zeros(length - width)
            tail = mixture.new_ones(width) / float(width)
            components.append(torch.cat((prefix, tail), dim=0))
        basis = torch.stack(components, dim=0)
        return (mixture.view(-1, 1) * basis).sum(dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.temporal_weights(x.shape[-1]).to(
            device=x.device, dtype=x.dtype
        )
        return (x * weights.view(1, 1, -1)).sum(dim=-1)


class LearnedPositionalPool1D(nn.Module):
    """Input-independent learned softmax weights over temporal positions.

    Zero initialization recovers exact global average pooling at epoch zero.
    The weights are parameters of the model, not per-sample attention scores,
    so the readout remains linear in the branch activations.
    """

    def __init__(self, in_channels: int, seq_len: int = 100):
        super().__init__()
        if int(seq_len) <= 0:
            raise ValueError("seq_len must be positive")
        self.out_channels = int(in_channels)
        self.seq_len = int(seq_len)
        self.logits = nn.Parameter(torch.zeros(self.seq_len))

    def temporal_weights(self, length: int) -> torch.Tensor:
        length = int(length)
        if length <= 0:
            raise ValueError("temporal length must be positive")
        logits = self.logits
        if length != self.seq_len:
            logits = F.interpolate(
                logits.view(1, 1, -1), size=length,
                mode="linear", align_corners=True,
            ).view(-1)
        return F.softmax(logits, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.temporal_weights(x.shape[-1]).to(
            device=x.device, dtype=x.dtype
        )
        return (x * weights.view(1, 1, -1)).sum(dim=-1)


class BranchLocalPool1D(nn.Module):
    """Apply one independent temporal pool to each semantic branch."""

    def __init__(self, kind: str, branch_channels: Sequence[int],
                 seq_len: int = 100,
                 windows: Sequence[int] = (8, 32, 100)):
        super().__init__()
        if kind not in {"multi_scale", "positional"}:
            raise ValueError("branch-local pool must be multi_scale or positional")
        pools = []
        for channels in branch_channels:
            if kind == "multi_scale":
                pools.append(MultiScaleTailPool1D(channels, windows=windows))
            else:
                pools.append(LearnedPositionalPool1D(channels, seq_len=seq_len))
        self.kind = kind
        self.pools = nn.ModuleList(pools)
        self.out_channels = sum(int(c) for c in branch_channels)

    def forward(self, branches: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(branches) != len(self.pools):
            raise ValueError("branch count does not match branch-local pools")
        return torch.cat(
            [pool(x) for pool, x in zip(self.pools, branches)], dim=1
        )


def make_pool(kind: str, in_channels: int, *, seq_len: int = 100,
              windows: Sequence[int] = (8, 32, 100)) -> nn.Module:
    if kind == "attention":
        return AttentionPool1D(in_channels)
    if kind == "avg":
        return AvgPool1D(in_channels)
    if kind == "max":
        return MaxPool1D(in_channels)
    if kind in {"avg_last", "avg+last"}:
        return AvgLastPool1D(in_channels)
    if kind == "multi_scale":
        return MultiScaleTailPool1D(in_channels, windows=windows)
    if kind == "positional":
        return LearnedPositionalPool1D(in_channels, seq_len=seq_len)
    raise ValueError(
        f"unknown pool kind: {kind!r}. "
        "Use 'attention'|'avg'|'max'|'avg_last'|'multi_scale'|'positional'."
    )


# ───────────────────────────────────────────────────────────────────
# Branch factory
# ───────────────────────────────────────────────────────────────────


def make_branch(
    backbone: str,
    in_channels: int,
    channels: Sequence[int],
    kernel_size: int,
    dropout: float,
    dilation_base: int = 2,
    tcn_norm: str = "none",
) -> nn.Module:
    if backbone == "cnn":
        return CNNBranch(in_channels, channels, kernel_size=kernel_size,
                          dropout=dropout)
    if backbone == "tcn":
        return TCNBranch(in_channels, channels, kernel_size=kernel_size,
                          dropout=dropout, dilation_base=dilation_base,
                          norm=tcn_norm)
    raise ValueError(f"unknown backbone: {backbone!r}. Use 'cnn'|'tcn'.")


# ───────────────────────────────────────────────────────────────────
# Full model
# ───────────────────────────────────────────────────────────────────


class LOBBranchModel(nn.Module):
    """
    Generic 4-branch FI-2010 LOB classifier.

    backbone : "cnn" | "tcn"
    pool     : "attention" | "avg" | "max" | "avg_last" |
               "multi_scale" | "positional"

    Input  : (B, T, F=40)
    Output : (B, n_classes) logits
    """

    def __init__(
        self,
        n_features: int = 40,
        n_classes: int = 3,
        backbone: str = "tcn",
        pool: str = TRAINING.pool,
        cnn_channels: Sequence[int] = TRAINING.channels,
        cnn_kernel: int = TRAINING.kernel_size,
        cnn_dropout: float = TRAINING.dropout,
        fc_hidden: Sequence[int] = TRAINING.fc_hidden,
        branch_indices: Optional[dict] = None,
        tcn_dilation_base: int = 2,
        input_norm: str = "identity",
        tcn_norm: str = "none",
        interaction_channels: int = 0,
        seq_len: int = 100,
        pool_windows: Sequence[int] = (8, 32, 100),
    ):
        super().__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.backbone = backbone
        self.pool_kind = pool
        self.input_norm_kind = input_norm
        self.tcn_norm_kind = tcn_norm
        self.interaction_channels = int(interaction_channels)
        self.seq_len = int(seq_len)
        self.pool_windows = tuple(int(w) for w in pool_windows)
        if self.interaction_channels < 0:
            raise ValueError("interaction_channels must be >= 0")

        if branch_indices is None:
            branch_indices = dict(
                ask_price=list(range(0, 40, 4)),
                ask_volume=list(range(1, 40, 4)),
                bid_price=list(range(2, 40, 4)),
                bid_volume=list(range(3, 40, 4)),
            )
        self.branch_names = list(branch_indices.keys())
        self.branch_indices = branch_indices
        for name, idx in branch_indices.items():
            self.register_buffer(
                f"_idx_{name}",
                torch.tensor(idx, dtype=torch.long),
                persistent=False,
            )

        # Branch-aware input normalization (no-op by default).
        # Built AFTER branch_indices so it knows which feature columns belong
        # to which branch. Operates on the raw (B,T,F) input BEFORE the
        # existing transpose+split. See lob_xai/input_norm.py for variants.
        self.input_norm = make_input_norm(
            input_norm, n_features=n_features, branch_indices=branch_indices,
        )

        self.cnns = nn.ModuleList([
            make_branch(
                backbone=backbone,
                in_channels=len(branch_indices[name]),
                channels=list(cnn_channels),
                kernel_size=cnn_kernel,
                dropout=cnn_dropout,
                dilation_base=tcn_dilation_base,
                tcn_norm=tcn_norm,
            )
            for name in self.branch_names
        ])

        concat_channels = sum(b.out_channels for b in self.cnns)

        # The semantic TCN encoders remain independent.  This optional 1x1
        # stream sees their time-aligned outputs and makes cross-branch effects
        # explicit instead of hiding them inside a monolithic backbone.
        if self.interaction_channels:
            self.interaction = nn.Sequential(
                nn.Conv1d(concat_channels, self.interaction_channels,
                          kernel_size=1),
                make_tcn_norm(tcn_norm, self.interaction_channels),
                nn.GELU(),
                nn.Dropout(cnn_dropout),
            )
        else:
            self.interaction = None

        self.branch_local_pool = pool in {"multi_scale", "positional"}
        if self.branch_local_pool:
            if self.interaction_channels:
                raise ValueError(
                    "branch-local pooling forbids an interaction stream before the head"
                )
            self.pool = BranchLocalPool1D(
                pool,
                [branch.out_channels for branch in self.cnns],
                seq_len=self.seq_len,
                windows=self.pool_windows,
            )
        else:
            readout_channels = concat_channels + self.interaction_channels
            self.pool = make_pool(
                pool, readout_channels, seq_len=self.seq_len,
                windows=self.pool_windows,
            )

        fc = []
        prev = self.pool.out_channels
        for h in fc_hidden:
            fc.append(nn.Linear(prev, h))
            fc.append(nn.LeakyReLU(0.01, inplace=False))
            fc.append(nn.Dropout(cnn_dropout))
            prev = h
        fc.append(nn.Linear(prev, n_classes))
        self.fc = nn.Sequential(*fc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, f"expected (B, T, F), got {x.shape}"
        # Branch-aware input normalization. For input_norm='identity' this is a
        # no-op and the model behaves exactly like the original LOBBranchModel.
        x = self.input_norm(x)
        x_bft = x.transpose(1, 2)
        outs = []
        for name, cnn in zip(self.branch_names, self.cnns):
            idx = getattr(self, f"_idx_{name}")
            xb = x_bft.index_select(dim=1, index=idx)
            outs.append(cnn(xb))
        if self.branch_local_pool:
            z = self.pool(outs)
            return self.fc(z)
        h = torch.cat(outs, dim=1)
        if self.interaction is not None:
            h_interaction = self.interaction(h)
            h = torch.cat((h, h_interaction), dim=1)
        z = self.pool(h)
        return self.fc(z)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)


# Backward-compat alias (so old code/checkpoints still work)
LOBBranchCNN = LOBBranchModel
