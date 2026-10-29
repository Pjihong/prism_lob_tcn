"""DeepLOB comparison model with an Inception stem and LSTM.

Inputs: FI-2010 windows in the layout accepted by forward().
Outputs: raw class logits for cross-entropy training.
Attribution hooks: inp1, inp2 and inp3.
Scope: this repository implementation is identified by source hashes; it is
not a claim of byte-identical equivalence to an upstream release.
"""
from __future__ import annotations
import torch
from torch import nn


class DeepLOB(nn.Module):
    """
    Input : (batch, T, 40)  — FI-2010 window, T timesteps × 40 LOB features
    Output: (batch, n_classes) — raw logits

    Native capacity (~140k params). Do NOT shrink for the rebuttal.
    """

    def __init__(self, n_classes: int = 3, n_features: int = 40):
        super().__init__()
        if int(n_classes) < 2:
            raise ValueError("n_classes must be at least 2")
        if int(n_features) != 40:
            raise ValueError(
                "this DeepLOB stem is defined for exactly 40 interleaved LOB features"
            )
        self.n_classes = int(n_classes)
        self.n_features = int(n_features)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 2), stride=(1, 2)),
            nn.Tanh(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.Tanh(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.Tanh(), nn.BatchNorm2d(32),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 10)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=(4, 1)),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(32),
        )
        self.inp1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 1), padding="same"),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
        )
        self.inp2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(5, 1), padding="same"),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
        )
        self.inp3 = nn.Sequential(
            nn.MaxPool2d((3, 1), stride=(1, 1), padding=(1, 0)),
            nn.Conv2d(32, 64, kernel_size=(1, 1), padding="same"),
            nn.LeakyReLU(0.01), nn.BatchNorm2d(64),
        )
        self.lstm = nn.LSTM(input_size=192, hidden_size=64,
                            num_layers=1, batch_first=True)
        self.fc1 = nn.Linear(64, self.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected input (B,T,F), got shape={tuple(x.shape)}")
        if x.shape[-1] != self.n_features:
            raise ValueError(
                f"expected F={self.n_features}, got shape={tuple(x.shape)}"
            )
        if x.shape[1] < 19:
            raise ValueError("DeepLOB convolutional stem requires at least 19 timesteps")
        # x: (batch, T, 40) → add channel dim
        x = x[:, None, :, :]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        a, b, c = self.inp1(x), self.inp2(x), self.inp3(x)
        x = torch.cat((a, b, c), dim=1)
        if x.shape[-1] != 1:
            raise RuntimeError(
                "DeepLOB feature-axis collapse failed; expected width=1, "
                f"got shape={tuple(x.shape)}"
            )
        x = x.squeeze(-1).transpose(1, 2).contiguous()  # (B,T_eff,192)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc1(out)  # raw logits


def build_deeplob(n_classes: int = 3, n_features: int = 40) -> DeepLOB:
    return DeepLOB(n_classes=n_classes, n_features=n_features)
