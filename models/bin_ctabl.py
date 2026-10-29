"""
models/bin_ctabl.py — BiN + C(TABL) model for LOB classification.

Based on Tran et al. (2018, 2019). The original code distributed by the
authors depends on `constants as cst` for device handling and uses an in-model
softmax. This implementation is self-contained:

  - No `constants` import (device inferred from tensor)
  - Returns LOGITS, not probabilities (compatible with CrossEntropy / Focal loss)
  - Clamps lambda / mixing weights in-graph (no `if cond: reassign Parameter`)
  - Compatible with our (B, T, F) input convention via initial permute

Architecture (defaults for FI-2010 with seq_len=100, n_features=40):
    Input (B, 40, 100)
    BiN                     → (B, 40, 100)
    BL  (60, 10)            → (B, 60, 10)
    Dropout
    BL2 (120, 5)            → (B, 120, 5)
    Dropout
    TABL (n_classes, 1)     → (B, n_classes, 1)
    squeeze(-1)              → (B, n_classes)  [logits]

The factory `build_bin_ctabl(n_classes, seq_len, n_features)` returns a
model that conforms to the baseline-trainer interface used elsewhere in
the repository.
"""
from __future__ import annotations

import torch
from torch import nn


# ────────────────────────────────────────────────────────────────────────
#  BiN — Bilinear Normalization layer
# ────────────────────────────────────────────────────────────────────────
class BiN(nn.Module):
    """Bilinear Normalization (Tran et al. 2018).

    Normalizes along BOTH temporal and feature axes, then learnably mixes:
        x_temporal = (x − μ_t) / σ_t * γ1 + β1
        x_feature  = (x − μ_f) / σ_f * γ2 + β2
        out = y1 · x_temporal + y2 · x_feature

    where y1, y2 are scalar mixing weights constrained to be ≥ 0.

    Input  : (B, D, T)
    Output : (B, D, T)
    """

    def __init__(self, d1: int, t1: int):
        super().__init__()
        self.d1 = d1
        self.t1 = t1
        # Per-feature scale/bias for the temporal-norm branch (broadcast over T)
        self.gamma1 = nn.Parameter(torch.ones(d1, 1))
        self.beta1 = nn.Parameter(torch.zeros(d1, 1))
        # Per-timestep scale/bias for the feature-norm branch (broadcast over D)
        self.gamma2 = nn.Parameter(torch.ones(1, t1))
        self.beta2 = nn.Parameter(torch.zeros(1, t1))
        # Mixing weights (scalars)
        self.y1 = nn.Parameter(torch.tensor(0.5))
        self.y2 = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-6
        # Temporal normalization (across T for each (B, d))
        mean_t = x.mean(dim=2, keepdim=True)
        std_t = x.std(dim=2, keepdim=True) + eps
        x_t = (x - mean_t) / std_t * self.gamma1 + self.beta1
        # Feature normalization (across D for each (B, t))
        mean_f = x.mean(dim=1, keepdim=True)
        std_f = x.std(dim=1, keepdim=True) + eps
        x_f = (x - mean_f) / std_f * self.gamma2 + self.beta2
        # Mix, clamping weights to be non-negative
        y1 = torch.clamp(self.y1, min=0.0)
        y2 = torch.clamp(self.y2, min=0.0)
        return y1 * x_t + y2 * x_f


# ────────────────────────────────────────────────────────────────────────
#  BL — Bilinear Layer
# ────────────────────────────────────────────────────────────────────────
class BL_layer(nn.Module):
    """Bilinear Layer:  Y = ReLU(W1 · X · W2 + B)

    Input  : (B, d_in,  t_in)
    Output : (B, d_out, t_out)
    """

    def __init__(self, d_out: int, d_in: int, t_in: int, t_out: int):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(d_out, d_in))
        self.W2 = nn.Parameter(torch.empty(t_in, t_out))
        self.B = nn.Parameter(torch.zeros(d_out, t_out))
        nn.init.kaiming_uniform_(self.W1, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.W2, nonlinearity="relu")
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.W1 @ x @ self.W2 + self.B)


# ────────────────────────────────────────────────────────────────────────
#  TABL — Temporal Attention Bilinear Layer
# ────────────────────────────────────────────────────────────────────────
class TABL_layer(nn.Module):
    """Temporal Attention Bilinear Layer.

    Implements eqs. (7)-(11) of Tran et al. (2018, "Temporal attention augmented
    bilinear network for financial time series prediction"):

       (7)  X' = W1 · X
       (8)  E  = X' · W̃            where W̃ has diagonal forced to 1/t_in
       (9)  A  = softmax(E)
       (10) X'' = λ · X' + (1−λ) · X' ⊙ A
       (11) Y  = X'' · W2 + B

    Input  : (B, d_in,  t_in)
    Output : (B, d_out, t_out)
    """

    def __init__(self, d_out: int, d_in: int, t_in: int, t_out: int):
        super().__init__()
        self.t_in = t_in
        self.W1 = nn.Parameter(torch.empty(d_out, d_in))
        # W initialised to 1/t_in (uniform attention prior)
        self.W = nn.Parameter(torch.full((t_in, t_in), 1.0 / t_in))
        self.W2 = nn.Parameter(torch.empty(t_in, t_out))
        self.B = nn.Parameter(torch.zeros(d_out, t_out))
        # Soft attention mixing weight λ ∈ [0, 1]
        self.lam = nn.Parameter(torch.tensor(0.5))
        nn.init.kaiming_uniform_(self.W1, nonlinearity="relu")
        nn.init.kaiming_uniform_(self.W2, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clamp λ to [0, 1] every forward (in-graph; no Parameter reassignment)
        lam = torch.clamp(self.lam, 0.0, 1.0)
        # Step (7): first-mode mapping
        x = self.W1 @ x                                          # (B, d_out, t_in)
        # Step (8): attention with diag forced to 1/t_in
        eye = torch.eye(self.t_in, dtype=x.dtype, device=x.device)
        W = self.W - self.W * eye + eye / self.t_in
        E = x @ W                                                # (B, d_out, t_in)
        # Step (9): softmax along temporal axis
        A = torch.softmax(E, dim=-1)
        # Step (10): mix raw + attention-weighted
        x = lam * x + (1.0 - lam) * x * A                        # (B, d_out, t_in)
        # Step (11): temporal mapping → bias
        return x @ self.W2 + self.B                              # (B, d_out, t_out)


# ────────────────────────────────────────────────────────────────────────
#  BiN_CTABL — full model
# ────────────────────────────────────────────────────────────────────────
class BiN_CTABL(nn.Module):
    """BiN + C(TABL) network.

    Conforms to the baseline-trainer convention:
        input  shape : (B, T, F=40)       (B = batch, T = seq_len, F = features)
        output shape : (B, n_classes)     LOGITS — loss applies softmax/log-softmax
    """

    def __init__(
        self,
        n_classes: int = 3,
        seq_len: int = 100,
        n_features: int = 40,
        # Intermediate dimensions (Tran et al. FI-2010 config, rescaled for seq_len=100)
        d2: int = 60, t2: int = 10,
        d3: int = 120, t3: int = 5,
        dropout: float = 0.1,
        max_norm: float = 10.0,
    ):
        super().__init__()
        dims = {
            "n_classes": n_classes, "seq_len": seq_len,
            "n_features": n_features, "d2": d2, "t2": t2,
            "d3": d3, "t3": t3,
        }
        if int(n_classes) < 2:
            raise ValueError("n_classes must be at least 2")
        if any(int(value) <= 0 for key, value in dims.items() if key != "n_classes"):
            raise ValueError(f"all architecture dimensions must be positive: {dims}")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0,1)")
        if float(max_norm) <= 0:
            raise ValueError("max_norm must be positive")
        self.n_classes = int(n_classes)
        self.seq_len = int(seq_len)
        self.n_features = int(n_features)
        self.max_norm = float(max_norm)

        d1, t1 = n_features, seq_len
        d4, t4 = n_classes, 1

        self.bin = BiN(d1, t1)
        self.bl1 = BL_layer(d2, d1, t1, t2)
        self.bl2 = BL_layer(d3, d2, t2, t3)
        self.tabl = TABL_layer(d4, d3, t3, t4)
        self.dropout = nn.Dropout(dropout)
        self.constrain_parameters_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected input (B,T,F), got shape={tuple(x.shape)}")
        if x.shape[1] != self.seq_len or x.shape[2] != self.n_features:
            raise ValueError(
                f"expected (B,{self.seq_len},{self.n_features}), "
                f"got shape={tuple(x.shape)}"
            )
        # (B, T, F) → (B, F, T)
        x = x.permute(0, 2, 1)
        x = self.bin(x)
        x = self.bl1(x)
        x = self.dropout(x)
        x = self.bl2(x)
        x = self.dropout(x)
        x = self.tabl(x)                # (B, n_classes, 1)
        # Squeeze ONLY the last dim — `torch.squeeze(x)` without dim would also
        # remove the batch dim when B=1 (e.g. unit-tests, single-sample inference),
        # which silently breaks downstream code. squeeze(-1) is safe.
        x = x.squeeze(-1)               # (B, n_classes)
        # Return logits — the training loop applies softmax / log-softmax via CE/Focal
        return x

    @torch.no_grad()
    def constrain_parameters_(self) -> None:
        """Project constrained matrices after an optimizer step.

        Mutating ``Parameter.data`` inside ``forward`` changes model state during
        evaluation and bypasses autograd.  The baseline trainer calls this hook
        immediately after every optimizer step instead.
        """
        matrices = (
            self.bl1.W1, self.bl1.W2, self.bl2.W1, self.bl2.W2,
            self.tabl.W1, self.tabl.W, self.tabl.W2,
        )
        for w in matrices:
            self._max_norm_(w)

    def _max_norm_(self, w: torch.Tensor) -> None:
        """In-place max-norm projection on a weight matrix."""
        with torch.no_grad():
            norm = torch.linalg.matrix_norm(w)
            if norm > self.max_norm:
                w.mul_(self.max_norm / (norm + 1e-8))


# ────────────────────────────────────────────────────────────────────────
#  Factory — matches build_deeplob / build_tlob convention
# ────────────────────────────────────────────────────────────────────────
def build_bin_ctabl(
    n_classes: int = 3,
    seq_len: int = 100,
    num_features: int | None = None,
    # backward-compat alias: some baseline calling code uses `n_features=`
    n_features: int | None = None,
    d2: int = 60, t2: int = 10,
    d3: int = 120, t3: int = 5,
    dropout: float = 0.1,
    max_norm: float = 10.0,
) -> BiN_CTABL:
    """Factory mirroring build_tlob's signature.

    Accepts either `num_features` (TLOB style) or `n_features` (DeepLOB style).
    """
    if num_features is None and n_features is None:
        n_features = 40
    elif n_features is None:
        n_features = int(num_features)
    elif num_features is not None and int(n_features) != int(num_features):
        raise ValueError(
            f"conflicting feature counts: num_features={num_features}, "
            f"n_features={n_features}"
        )
    return BiN_CTABL(
        n_classes=n_classes,
        seq_len=seq_len,
        n_features=n_features,
        d2=d2, t2=t2, d3=d3, t3=t3,
        dropout=dropout, max_norm=max_norm,
    )
