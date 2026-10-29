"""
lob_xai/input_norm.py — branch-aware input normalization layers.

Three options, all operate on raw input x with shape (B, T, F=40) BEFORE the
LOBBranchModel splits features into branches, and they all RESPECT the branch
decomposition: each variant treats the feature columns of a branch as one
homogeneous group (price-only or volume-only), so the price-volume scale gap
(~100x in raw FI-2010 scale) is never mixed.

Three variants:

1) "identity" : no-op. Preserves the paper's original LOBBranchModel behavior.
                Use as the seed-matched baseline in any ablation.

2) "dain"     : Deep Adaptive Input Normalization (Passalis et al. 2019), but
                applied PER BRANCH instead of across all 40 features. For each
                branch group g with feature index set I_g:
                    a_t = x[:, :, I_g]                       # (B, T, |I_g|)
                    mu_g = MLP_shift_g(mean_t(a_t))          # (B, |I_g|)
                    a    = a_t - mu_g.unsqueeze(1)
                    sd_g = relu(MLP_scale_g( std_t(a) )) + eps
                    a    = a / sd_g.unsqueeze(1)
                    g_g  = sigmoid(MLP_gate_g( mean_t(a) ))
                    a    = a * g_g.unsqueeze(1)
                    x[:, :, I_g] := a
                Each branch gets its own (shift, scale, gate) sub-net. This is
                the closest faithful adaptation of DAIN to a branched LOB model:
                the same adaptive shift/scale/gate concept that TLOB's BiN
                exploits (same author group), but kept inside branch semantics
                so it does not blur price/volume scales the way a global DAIN
                would.

3) "revin_c"  : Conditional, branch-wise Reversible Instance Norm — denormalize
                step omitted because the task is classification (logits have no
                scale to restore). For each branch g:
                    mu_g = mean over time of features in I_g     (B, 1, |I_g|)
                    sd_g = std over time of features in I_g + eps
                    z    = (x[..., I_g] - mu_g) / sd_g
                    z    = gamma_g * z + beta_g                  affine, learned
                    s_g  = sigmoid(alpha_g)        scalar per branch, learned
                    x[..., I_g] := s_g * z + (1 - s_g) * x[..., I_g]
                "Conditional" means: the per-branch gate s_g is learned, so the
                model can decide branch-by-branch whether within-window z-score
                actually helps or is bypassed. This is exactly the RevIN
                normalization step (Kim et al. 2022) made (a) branch-local and
                (b) softly toggleable. The reversible denormalize is intentionally
                dropped — there is no per-sample regression scale to restore.

Why this design:
  - LOBBranchModel.forward already splits x by branch; if normalization is
    branch-aware, gradient through that path is per-branch as well, so the
    faithfulness audit (gradient*input saliency on raw x) still measures
    "which raw timestep matters" — only now with a learned branch-local
    transform in between.
  - All variants are torch.nn.Module subclasses with state_dict, so they save
    into ckpt['model_state'] alongside the existing weights. Old checkpoints
    that have no norm-state keys still load with strict=False.
"""
from __future__ import annotations
from typing import Dict, List
import torch
import torch.nn as nn


_EPS = 1e-6


def _branch_idx_list(branch_indices: Dict[str, List[int]]) -> List[List[int]]:
    """Stable ordering of branch index lists (one int list per branch)."""
    return [list(branch_indices[name]) for name in branch_indices]


# ───────────────────────────────────────────────────────────────────
# Identity (default, no-op)
# ───────────────────────────────────────────────────────────────────
class InputIdentity(nn.Module):
    """No-op normalization. Use as the baseline arm in ablation."""

    def __init__(self, n_features: int, branch_indices: Dict[str, List[int]]):
        super().__init__()
        self.n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        return x


# ───────────────────────────────────────────────────────────────────
# Branch-wise DAIN
# ───────────────────────────────────────────────────────────────────
class InputDAIN(nn.Module):
    """
    Branch-wise Deep Adaptive Input Normalization.

    For each branch, an independent (shift, scale, gate) MLP triple learns to
    adapt the within-window statistics. All three sub-nets are simple linear
    layers over the feature dimension of the branch (i.e. no time mixing),
    matching the original DAIN formulation.
    """

    def __init__(self, n_features: int, branch_indices: Dict[str, List[int]]):
        super().__init__()
        self.n_features = n_features
        self.branch_names = list(branch_indices.keys())
        # register each branch's column index list as a buffer so it follows
        # .to(device) without being a learnable parameter
        self.branch_sizes: List[int] = []
        for i, name in enumerate(self.branch_names):
            idx = list(branch_indices[name])
            self.register_buffer(f"_idx_{i}",
                                 torch.tensor(idx, dtype=torch.long),
                                 persistent=False)
            self.branch_sizes.append(len(idx))

        # per-branch shift/scale/gate sub-networks (no bias on shift to match
        # the canonical DAIN; scale uses ReLU to keep std positive)
        self.shift_layers = nn.ModuleList([
            nn.Linear(s, s, bias=False) for s in self.branch_sizes
        ])
        self.scale_layers = nn.ModuleList([
            nn.Linear(s, s, bias=False) for s in self.branch_sizes
        ])
        self.gate_layers = nn.ModuleList([
            nn.Linear(s, s, bias=True) for s in self.branch_sizes
        ])

        # initialize shift/scale near identity so DAIN starts as approximately
        # "subtract empirical mean, divide by empirical std" — i.e. close to a
        # plain within-window z-score on epoch 0, then drifts as training proceeds
        for lin in self.shift_layers:
            nn.init.eye_(lin.weight)
        for lin in self.scale_layers:
            nn.init.eye_(lin.weight)
        for lin in self.gate_layers:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)   # sigmoid(0) = 0.5, neutral gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        B, T, F = x.shape
        out = x.clone()
        for i, name in enumerate(self.branch_names):
            idx = getattr(self, f"_idx_{i}")           # (G,) feature indices
            a = x.index_select(dim=2, index=idx)       # (B, T, G)

            # 1) adaptive shift: subtract a learned function of per-feature mean
            mu = a.mean(dim=1)                          # (B, G)
            mu_hat = self.shift_layers[i](mu)           # (B, G)
            a = a - mu_hat.unsqueeze(1)                 # broadcast over time

            # 2) adaptive scale: divide by a learned positive function of std
            sd = a.std(dim=1, unbiased=False)           # (B, G)
            sd_hat = torch.relu(self.scale_layers[i](sd)) + _EPS
            a = a / sd_hat.unsqueeze(1)

            # 3) adaptive gate: per-feature soft suppression based on residual mean
            mu2 = a.mean(dim=1)                         # (B, G)
            gate = torch.sigmoid(self.gate_layers[i](mu2))   # (B, G) in (0,1)
            a = a * gate.unsqueeze(1)

            # write back into the (B,T,F) tensor at this branch's columns
            out = out.index_copy(dim=2, index=idx, source=a)
        return out


# ───────────────────────────────────────────────────────────────────
# Conditional, branch-wise RevIN (no denormalize)
# ───────────────────────────────────────────────────────────────────
class InputRevIN_C(nn.Module):
    """
    Conditional branch-wise RevIN (normalization-only; no denormalize step).

    For each branch g:
        z_g  = (x_g - mu_g(x)) / (sd_g(x) + eps)         instance-norm over time
        z_g  = gamma_g * z_g + beta_g                    learned affine
        s_g  = sigmoid(alpha_g)                          scalar mixing gate per branch
        out_g = s_g * z_g + (1 - s_g) * x_g

    'Conditional' = per-branch gate alpha_g is a learnable scalar, so the model
    can shut off or partially apply within-window z-score on a branch-by-branch
    basis. This addresses the failure mode of vanilla RevIN on classification:
    if a branch already lives on a clean scale (e.g. volume features already
    log-normalized), forcing within-window z-score can erase amplitude info
    the classifier was using; the gate lets the model opt out per branch.

    Denormalize is intentionally omitted: classification outputs are logits
    (softmax-invariant), so there is no per-sample scale to restore — RevIN's
    reversibility property does not apply here. What remains is its strongest
    component for non-stationary inputs: per-window z-score with per-channel
    affine.
    """

    def __init__(self, n_features: int, branch_indices: Dict[str, List[int]]):
        super().__init__()
        self.n_features = n_features
        self.branch_names = list(branch_indices.keys())
        self.branch_sizes: List[int] = []
        for i, name in enumerate(self.branch_names):
            idx = list(branch_indices[name])
            self.register_buffer(f"_idx_{i}",
                                 torch.tensor(idx, dtype=torch.long),
                                 persistent=False)
            self.branch_sizes.append(len(idx))

        # per-branch learned affine (gamma, beta) on the normalized tensor.
        # Initialize gamma=1, beta=0 → identity affine at epoch 0.
        self.gamma = nn.ParameterList([
            nn.Parameter(torch.ones(s)) for s in self.branch_sizes
        ])
        self.beta = nn.ParameterList([
            nn.Parameter(torch.zeros(s)) for s in self.branch_sizes
        ])

        # per-branch gate logit alpha (scalar). Initialize alpha = +2 so
        # sigmoid(alpha) ~ 0.88, i.e. start ~mostly normalized but with room to
        # close off.
        self.alpha = nn.Parameter(torch.full((len(self.branch_names),),
                                              2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F = x.shape
        out = x.clone()
        for i, name in enumerate(self.branch_names):
            idx = getattr(self, f"_idx_{i}")
            a = x.index_select(dim=2, index=idx)       # (B, T, G)

            mu = a.mean(dim=1, keepdim=True)            # (B, 1, G)
            sd = a.std(dim=1, keepdim=True, unbiased=False) + _EPS
            z = (a - mu) / sd
            # learned per-feature affine
            z = z * self.gamma[i].view(1, 1, -1) + self.beta[i].view(1, 1, -1)
            # soft mix between normalized and raw, gated per branch
            s = torch.sigmoid(self.alpha[i])
            mixed = s * z + (1.0 - s) * a

            out = out.index_copy(dim=2, index=idx, source=mixed)
        return out


# ───────────────────────────────────────────────────────────────────
# Factory
# ───────────────────────────────────────────────────────────────────
def make_input_norm(kind: str, n_features: int,
                    branch_indices: Dict[str, List[int]]) -> nn.Module:
    kind = (kind or "identity").lower()
    if kind == "identity" or kind == "none":
        return InputIdentity(n_features, branch_indices)
    if kind == "dain":
        return InputDAIN(n_features, branch_indices)
    if kind == "revin_c" or kind == "revin-c" or kind == "revinc":
        return InputRevIN_C(n_features, branch_indices)
    raise ValueError(f"unknown input_norm kind: {kind!r}. "
                     "Use 'identity' | 'dain' | 'revin_c'.")
