"""Self-contained TLOB comparison model with BiN and temporal/feature attention.

Inputs: tensors shaped (batch, time, features).
Outputs: raw logits shaped (batch, n_classes).
The FI-2010 configuration uses hidden_dim=128, num_layers=4 and num_heads=1.
Scope: implementation and run settings are recorded separately; see
models/README.md and the manuscript's TLOB optimization exception.
"""
from __future__ import annotations
import torch
from torch import nn


# ───────────────────────── BiN (Bilinear Normalization) ─────────────
class BiN(nn.Module):
    """
    Bilinear input normalization (Tran et al. 2021).
    Normalizes across BOTH the temporal and feature axes with learnable
    per-axis scale/shift, then combines via learnable gates.

    Expects x of shape (batch, num_features, seq_size).
    """
    def __init__(self, num_features: int, seq_size: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.seq_size = seq_size
        self.eps = eps
        # temporal-axis affine (per feature)
        self.gamma_t = nn.Parameter(torch.ones(num_features, 1))
        self.beta_t = nn.Parameter(torch.zeros(num_features, 1))
        # feature-axis affine (per timestep)
        self.gamma_f = nn.Parameter(torch.ones(1, seq_size))
        self.beta_f = nn.Parameter(torch.zeros(1, seq_size))
        # learnable mixing gates
        self.lam_t = nn.Parameter(torch.tensor(0.5))
        self.lam_f = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, T)
        # temporal normalization: stats over T (dim=2)
        mu_t = x.mean(dim=2, keepdim=True)
        sd_t = x.std(dim=2, keepdim=True) + self.eps
        xt = (x - mu_t) / sd_t
        xt = xt * self.gamma_t + self.beta_t
        # feature normalization: stats over F (dim=1)
        mu_f = x.mean(dim=1, keepdim=True)
        sd_f = x.std(dim=1, keepdim=True) + self.eps
        xf = (x - mu_f) / sd_f
        xf = xf * self.gamma_f + self.beta_f
        # learnable convex-ish combination
        lt = torch.sigmoid(self.lam_t)
        lf = torch.sigmoid(self.lam_f)
        return (lt * xt + lf * xf) / (lt + lf + self.eps)


# ───────────────────────── MLP ─────────────
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


# ───────────────────────── Transformer pieces ─────────────
class ComputeQKV(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.k = nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.v = nn.Linear(hidden_dim, hidden_dim * num_heads)

    def forward(self, x):
        return self.q(x), self.k(x), self.v(x)


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, final_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.qkv = ComputeQKV(hidden_dim, num_heads)
        self.attention = nn.MultiheadAttention(
            hidden_dim * num_heads, num_heads, batch_first=True)
        self.mlp = MLP(hidden_dim, hidden_dim * 4, final_dim)
        self.w0 = nn.Linear(hidden_dim * num_heads, hidden_dim)

    def forward(self, x):
        res = x
        q, k, v = self.qkv(x)
        x, att = self.attention(q, k, v, average_attn_weights=False,
                                 need_weights=True)
        x = self.w0(x)
        x = x + res
        x = self.norm(x)
        x = self.mlp(x)
        if x.shape[-1] == res.shape[-1]:
            x = x + res
        return x, att


def sinusoidal_positional_embedding(T: int, d: int, n: float = 10000.0,
                                     device=None):
    if d % 2 != 0:
        raise ValueError(f"sinusoidal emb needs even dim (got {d})")
    pos = torch.arange(0, T).unsqueeze(1)
    emb = torch.zeros(T, d)
    denom = torch.pow(n, 2 * torch.arange(0, d // 2) / d)
    emb[:, 0::2] = torch.sin(pos / denom)
    emb[:, 1::2] = torch.cos(pos / denom)
    if device is not None:
        emb = emb.to(device)
    return emb


# ───────────────────────── TLOB ─────────────
class TLOB(nn.Module):
    """
    Dual-axis (temporal + feature) transformer for LOB. FI-2010 path only
    (dataset_type != 'LOBSTER'), so the order-type embedding branch is
    not used.

    Output: raw logits (batch, n_classes).
    """
    def __init__(self,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 seq_size: int = 100,
                 num_features: int = 40,
                 num_heads: int = 1,
                 n_classes: int = 3,
                 is_sin_emb: bool = True,
                 dataset_type: str = "FI2010"):
        super().__init__()
        if int(n_classes) < 2:
            raise ValueError("n_classes must be at least 2")
        if int(num_layers) < 1:
            raise ValueError("num_layers must be positive")
        if int(num_heads) < 1:
            raise ValueError("num_heads must be positive")
        if int(seq_size) < 4 or int(hidden_dim) < 4:
            raise ValueError("seq_size and hidden_dim must both be at least 4")
        if is_sin_emb and int(hidden_dim) % 2:
            raise ValueError("sinusoidal positional embeddings require even hidden_dim")
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_size = seq_size
        self.num_features = num_features
        self.num_heads = num_heads
        self.n_classes = int(n_classes)
        self.is_sin_emb = is_sin_emb
        self.dataset_type = dataset_type

        self.norm_layer = BiN(num_features, seq_size)
        self.emb_layer = nn.Linear(num_features, hidden_dim)
        if is_sin_emb:
            self.register_buffer(
                "pos_encoder",
                sinusoidal_positional_embedding(seq_size, hidden_dim))
        else:
            self.pos_encoder = nn.Parameter(
                torch.randn(1, seq_size, hidden_dim))

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i != num_layers - 1:
                self.layers.append(TransformerLayer(hidden_dim, num_heads, hidden_dim))
                self.layers.append(TransformerLayer(seq_size, num_heads, seq_size))
            else:
                self.layers.append(TransformerLayer(hidden_dim, num_heads, hidden_dim // 4))
                self.layers.append(TransformerLayer(seq_size, num_heads, seq_size // 4))

        total_dim = (hidden_dim // 4) * (seq_size // 4)
        self.final_layers = nn.ModuleList()
        while total_dim > 128:
            self.final_layers.append(nn.Linear(total_dim, total_dim // 4))
            self.final_layers.append(nn.GELU())
            total_dim = total_dim // 4
        self.final_layers.append(nn.Linear(total_dim, self.n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected input (B,T,F), got shape={tuple(x.shape)}")
        if x.shape[1] != self.seq_size or x.shape[2] != self.num_features:
            raise ValueError(
                f"expected (B,{self.seq_size},{self.num_features}), "
                f"got shape={tuple(x.shape)}"
            )
        # x: (B, T, F)
        x = x.transpose(1, 2)
        x = self.norm_layer(x)
        x = x.transpose(1, 2)
        x = self.emb_layer(x)
        x = x + self.pos_encoder
        for layer in self.layers:
            x, _att = layer(x)
            x = x.permute(0, 2, 1)  # alternate temporal/feature axis
        # Flatten feature-major, matching the reference layout without an
        # additional tensor-rearrangement dependency.
        x = x.transpose(1, 2).contiguous().reshape(x.shape[0], -1)
        for layer in self.final_layers:
            x = layer(x)
        return x  # raw logits


def build_tlob(seq_size: int = 100, num_features: int = 40,
               hidden_dim: int = 128, num_layers: int = 4,
               num_heads: int = 1, n_classes: int = 3,
               is_sin_emb: bool = True,
               dataset_type: str = "FI2010") -> TLOB:
    return TLOB(hidden_dim=hidden_dim, num_layers=num_layers,
                seq_size=seq_size, num_features=num_features,
                num_heads=num_heads, n_classes=n_classes,
                is_sin_emb=is_sin_emb, dataset_type=dataset_type)
