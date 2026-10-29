"""Compute signed temporal attribution at branch outputs before pooling.

HiResCAM uses activation times gradient; Grad-CAM averages gradients over time.
Both target a class logit. The default hook is the post-residual branch output.
Inputs: model, (batch, time, features) tensor, branch and target class.
Outputs: signed maps and their absolute values, each shaped (batch, time).
Scope: normalization and perturbation evaluation are separate operations.
"""
from __future__ import annotations
from typing import Tuple, Optional, Union, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_METHOD = "hirescam"
SUPPORTED_METHODS = ("hirescam", "tsr", "gradcam")


# ───────────────────────────────────────────────────────────────────
# CAM core for a SINGLE branch
# ───────────────────────────────────────────────────────────────────


class TimeSeriesGradCAM:
    """
    Per-branch CAM engine for LOBBranchCNN.

    Usage:
        eng = TimeSeriesGradCAM(model, device, method="hirescam")
        cam_signed, cam_abs = eng.generate_batch(X, branch_idx=0,
                                                  target_class="predicted")
        eng.cleanup()
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str,
        method: str = DEFAULT_METHOD,
        tsr_alpha: float = 0.5,
        target_layer: str = "branch_output",
    ):
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"method must be in {SUPPORTED_METHODS}, got {method!r}")
        # init internal state BEFORE any validation that could fail
        self._handles: List = []
        self._a = None  # last activation
        self._g = None  # last gradient
        self.model = model
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.method = method
        self.tsr_alpha = float(tsr_alpha)
        if target_layer not in ("branch_output", "last_dilated_conv"):
            raise ValueError("target_layer must be 'branch_output' or 'last_dilated_conv'")
        self.target_layer = target_layer

    # ── Hook management ───────────────────────────────────────────
    def _attach_hooks(self, module: nn.Module) -> None:
        self._a = None
        self._g = None

        def fwd_hook(_m, _inp, out):
            # out: (B, C, T)  — keep ref for backward
            if not isinstance(out, torch.Tensor) or out.ndim != 3:
                raise RuntimeError(
                    f"CAM target must emit a (B,C,T) tensor, got {type(out)!r} "
                    f"with shape {getattr(out, 'shape', None)}"
                )
            self._a = out
            if out.requires_grad:
                # use tensor.register_hook (LeakyReLU(inplace=True) safe)
                def _g_hook(grad):
                    self._g = grad
                out.register_hook(_g_hook)

        h = module.register_forward_hook(fwd_hook)
        self._handles.append(h)

    def cleanup(self) -> None:
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()
        self._a = None
        self._g = None

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass

    # ── Locate target temporal representation ────────────────────
    def _find_target_layer(self, branch_idx: int) -> nn.Module:
        """Return the paper-defined pre-GAP branch representation.

        The old implementation selected the last registered Conv1d.  In a
        residual TCN that can be the 1x1 shortcut rather than the temporal
        feature used by the classifier.  The default now hooks the branch
        module itself, after residual addition and activation.  The legacy
        main-convolution representation remains available for sensitivity
        analysis via ``target_layer='last_dilated_conv'``.
        """
        cnns = getattr(self.model, "cnns", None)
        interaction = getattr(self.model, "interaction", None)
        if cnns is not None and branch_idx == len(cnns) and interaction is not None:
            if self.target_layer != "branch_output":
                raise ValueError(
                    "the explicit interaction stream supports branch_output CAM only"
                )
            return interaction
        if cnns is None or branch_idx < 0 or branch_idx >= len(cnns):
            raise ValueError(f"model has no cnns[{branch_idx}]")
        branch = cnns[branch_idx]
        if self.target_layer == "branch_output":
            return branch
        candidates = [
            m for m in branch.modules()
            if isinstance(m, nn.Conv1d) and int(m.kernel_size[0]) > 1
        ]
        if not candidates:
            raise ValueError(f"no dilated Conv1d in branch {branch_idx}")
        return candidates[-1]

    # ── target_class resolution ───────────────────────────────────
    def _resolve_target_class(
        self,
        target_class: Union[str, int, None],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns (B,) tensor of class indices.

        target_class:
          "predicted" | "up" (=0) | "flat" (=1) | "down" (=2) | int | None (= argmax = same as predicted)
        """
        B = logits.shape[0]
        if target_class is None or target_class == "predicted":
            return logits.argmax(dim=-1)
        if isinstance(target_class, str):
            mapping = dict(
                # Native FI-2010 direction labels (legacy path)
                up=0, flat=1, down=2,
                # Paper microstructure labels
                stationary=0, toxic_up=1, toxic_down=2,
                normal=0, vacuum=1,
            )
            if target_class not in mapping:
                raise ValueError(f"unknown target_class string: {target_class!r}")
            target_idx = mapping[target_class]
            if target_idx >= logits.shape[1]:
                raise ValueError(
                    f"target_class={target_class!r} maps to {target_idx}, but the "
                    f"model has only {logits.shape[1]} classes"
                )
            return torch.full((B,), target_idx,
                              dtype=torch.long, device=logits.device)
        if isinstance(target_class, int):
            if not 0 <= target_class < logits.shape[1]:
                raise ValueError(
                    f"target_class={target_class} outside [0, {logits.shape[1]})"
                )
            return torch.full((B,), int(target_class),
                              dtype=torch.long, device=logits.device)
        raise TypeError(f"invalid target_class type: {type(target_class)}")

    # ── CAM computation ───────────────────────────────────────────
    def _compute_cam(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute CAM from self._a, self._g.

        Returns
        -------
        cam_signed : (B, T) — signed CAM
        cam_abs    : (B, T) — |CAM|
        """
        a = self._a  # (B, C, T)
        g = self._g  # (B, C, T)
        if a is None or g is None:
            raise RuntimeError("forward/backward not run, no activations cached")

        if self.method == "hirescam":
            # element-wise grad × activation, sum over channels
            cam = (g * a).sum(dim=1)                               # (B, T)
        elif self.method == "tsr":
            # Temporal Saliency Rescaling: per-timestep relevance |G·A|
            rel = (g * a).abs().sum(dim=1)                          # (B, T)
            # threshold: keep top alpha fraction of timesteps
            B, T = rel.shape
            k = max(1, int(T * self.tsr_alpha))
            topk = torch.topk(rel, k=k, dim=-1).values[:, -1:]      # (B, 1) thresholds
            mask = (rel >= topk).float()
            # signed rescale
            sgn = (g * a).sum(dim=1).sign()
            cam = mask * rel * sgn
        elif self.method == "gradcam":
            # legacy: average grad over time per channel, then weight activations
            w = g.mean(dim=2, keepdim=True)                          # (B, C, 1)
            cam = (w * a).sum(dim=1)                                 # (B, T)
        else:
            raise RuntimeError(f"unhandled method {self.method}")

        cam_np = cam.detach().cpu().numpy()
        return cam_np, np.abs(cam_np)

    # ── Public API ────────────────────────────────────────────────
    def generate_batch(
        self,
        X: torch.Tensor,
        branch_idx: int,
        target_class: Union[str, int, None] = "predicted",
        smooth_steps: int = 0,
        noise_sigma: float = 0.1,
        desc: str = "",
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run CAM over entire batch X for one branch.

        Parameters
        ----------
        X : (N, T, F) tensor
        branch_idx : int in [0, n_branches-1]
        target_class : "predicted" | "up" | "flat" | "down" | int | None
        smooth_steps : SmoothGrad steps (0 = no smooth)
        noise_sigma : SmoothGrad noise std
        seed : optional seed for SmoothGrad noise (reproducibility)

        Returns
        -------
        cam_signed : (N, T)
        cam_abs    : (N, T)
        """
        self.model.eval()
        target_layer = self._find_target_layer(branch_idx)
        self.cleanup()
        self._attach_hooks(target_layer)

        X = X.to(self.device)
        N, T, F_ = X.shape

        if smooth_steps <= 0:
            cam_s, cam_a = self._cam_pass(X, target_class)
            self.cleanup()
            return cam_s, cam_a

        # SmoothGrad: average over noisy passes
        gen = torch.Generator(device=X.device)
        if seed is not None:
            gen.manual_seed(int(seed))
        cam_s_acc = np.zeros((N, T), dtype=np.float32)
        cam_a_acc = np.zeros((N, T), dtype=np.float32)
        for _ in range(smooth_steps):
            noise = torch.empty_like(X).normal_(generator=gen) * noise_sigma
            X_noisy = X + noise
            s, a = self._cam_pass(X_noisy, target_class)
            cam_s_acc += s
            cam_a_acc += a
        cam_s_acc /= smooth_steps
        cam_a_acc /= smooth_steps

        self.cleanup()
        return cam_s_acc, cam_a_acc

    def _cam_pass(
        self,
        X: torch.Tensor,
        target_class: Union[str, int, None],
    ) -> Tuple[np.ndarray, np.ndarray]:
        X = X.detach().requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        logits = self.model(X)                                       # (N, n_classes)
        tc = self._resolve_target_class(target_class, logits)        # (N,)
        # gather logits at target class per row, sum, backward
        gathered = logits.gather(1, tc.unsqueeze(1)).squeeze(1)      # (N,)
        gathered.sum().backward()
        return self._compute_cam()


# Aliases (VIX 코드와 호환성 유지)
TimeSeriesHiResCAM = TimeSeriesGradCAM
TimeSeriesTSR = TimeSeriesGradCAM


# ───────────────────────────────────────────────────────────────────
# Convenience: all branches at once
# ───────────────────────────────────────────────────────────────────


def compute_cam_all_branches(
    model: nn.Module,
    X: torch.Tensor,
    device: torch.device | str,
    target_class: Union[str, int, None] = "predicted",
    method: str = DEFAULT_METHOD,
    smooth_steps: int = 0,
    target_layer: str = "branch_output",
    include_interaction: bool = False,
) -> dict:
    """
    Run CAM on each branch separately. Returns dict[branch_name] = (cam_s, cam_a).
    """
    n_branches = len(model.cnns)
    branch_names = getattr(model, "branch_names", [f"b{i}" for i in range(n_branches)])
    branch_names = list(branch_names)
    if include_interaction and getattr(model, "interaction", None) is not None:
        branch_names.append("interaction")
    out = {}
    for i, name in enumerate(branch_names):
        eng = TimeSeriesGradCAM(
            model, device, method=method, target_layer=target_layer
        )
        try:
            s, a = eng.generate_batch(
                X, branch_idx=i,
                target_class=target_class,
                smooth_steps=smooth_steps,
                desc=f"CAM-{name}",
            )
            out[name] = (s, a)
        finally:
            eng.cleanup()
    return out
