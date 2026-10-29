"""Audit temporal attribution with donor-based deletion and insertion.

Select correct, high-confidence alerts and compare CAM, random and anti-CAM
contiguous spans. Deletion samples 10 donors from 50 context-nearest training
windows; insertion restores original spans into shared full-window donors.
Run: python -m lob_xai.counterfactual_test --help
Inputs: checkpoint, indexed FI-2010 development/test CSVs and task configuration.
Outputs: PNG, summary CSV, per-window CSV and NPZ curves under the --out prefix.
Scope: results depend on the model, cohort and replacement rule; spans need not
be nested across intervention ratios. See docs/REPRODUCE.md for full commands.
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

import torch
import torch.nn.functional as F

THIS = Path(__file__).resolve()
for cand in (THIS.parent, THIS.parent.parent):
    if (cand / "lob_xai").is_dir():
        sys.path.insert(0, str(cand))
        break

from lob_xai.models import LOBBranchModel
from lob_xai.branching import make_branching_scheme
from lob_xai.gradcam import SUPPORTED_METHODS, TimeSeriesGradCAM
from lob_xai.paper_protocol import COUNTERFACTUAL, DATA
from lob_xai.evaluation_protocol import select_top_confidence_alerts
from lob_xai.provenance import sha256_file, verify_file_identity

TASK_CLASSES = {"toxic": ["stationary", "toxic_up", "toxic_down"],
                "vacuum": ["normal", "vacuum"]}
TASK_FLAT = {"toxic": 0, "vacuum": 0}

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
    "figure.dpi": 120, "savefig.dpi": 300, "font.family": "DejaVu Sans",
})


# ====================================================================
# 1. Saliency providers — 모델마다 다르지만 (N,T) abs saliency 로 통일
# ====================================================================
class OursCAMProvider:
    """ours(LOBBranchModel): 4 branch |CAM| 합 → (N,T)."""
    def __init__(self, model, device, method="hirescam"):
        self.model, self.device, self.method = model, device, method
        self.n_branches = len(model.cnns)

    @torch.enable_grad()
    def saliency(self, X, target_class):
        T = X.shape[1]
        acc = np.zeros((X.shape[0], T), dtype=np.float64)
        for bi in range(self.n_branches):
            eng = TimeSeriesGradCAM(self.model, self.device, method=self.method)
            try:
                _, cam_abs = eng.generate_batch(X, branch_idx=bi,
                                                target_class=target_class)
                if cam_abs.shape[1] != T:
                    cam_abs = _interp_to_T(cam_abs, T)
                acc += cam_abs
            finally:
                eng.cleanup()
        return acc


class DeepLOBGradCAMProvider:
    """deeplob(DeepLOB): inception 출력(LSTM 직전)에 CAM → (N,T).

    method: "gradcam" | "hirescam" | "tsr"  (gradcam.py 의 _compute_cam 과 동일 공식)
    inception 출력 시간축은 T-18 (conv valid-padding). 입력 길이 T로 보간하고
    conv가 못 본 앞쪽 (T - T_eff) tick 은 saliency 0 으로 채운다 (causal 외삽 금지).
    """
    def __init__(self, model, device, method="gradcam", tsr_alpha=0.5):
        self.model, self.device, self.method = model, device, method
        self.tsr_alpha = float(tsr_alpha)
        self._acts, self._grads = {}, {}
        self._handles = []

    def _attach(self):
        self._acts, self._grads = {}, {}
        for name in ("inp1", "inp2", "inp3"):
            mod = getattr(self.model, name)
            def fwd(_m, _i, o, n=name):
                self._acts[n] = o
                if o.requires_grad:
                    o.register_hook(lambda g, n=n: self._grads.__setitem__(n, g))
            self._handles.append(mod.register_forward_hook(fwd))

    def _detach(self):
        for h in self._handles:
            try: h.remove()
            except Exception: pass
        self._handles = []

    def _cam_from(self, A, G):
        """A,G:(N,C,T_eff) → (N,T_eff) signed CAM. gradcam.py 와 동일 공식."""
        if self.method == "hirescam":
            return (G * A).sum(dim=1)                            # element-wise
        if self.method == "gradcam":
            w = G.mean(dim=2, keepdim=True)                      # 시간평균 가중치
            # Keep the signed map here, exactly as in lob_xai.gradcam.  All
            # paper saliency consumers take |CAM| afterwards.  Applying ReLU
            # only for DeepLOB would otherwise make the cross-model comparison
            # depend on two different Grad-CAM definitions.
            return (w * A).sum(dim=1)
        if self.method == "tsr":
            rel = (G * A).abs().sum(dim=1)                       # (N, T_eff)
            T_eff = rel.shape[1]
            k = max(1, int(T_eff * self.tsr_alpha))
            thr = torch.topk(rel, k=k, dim=-1).values[:, -1:]
            mask = (rel >= thr).float()
            sgn = (G * A).sum(dim=1).sign()
            return mask * rel * sgn
        raise ValueError(f"unknown method {self.method!r}")

    @torch.enable_grad()
    def saliency(self, X, target_class):
        self.model.eval()
        self._attach()
        # cuDNN RNN 은 eval 모드 backward 를 막는다. saliency 동안만 cuDNN 을 꺼서
        # native LSTM 구현으로 떨어뜨리면 eval 상태에서도 backward 가 된다.
        # (BatchNorm/Dropout 통계는 eval 그대로 유지 — 예측과 동일 조건)
        try:
            with torch.backends.cudnn.flags(enabled=False):
                X = X.to(self.device).detach().requires_grad_(True)
                self.model.zero_grad(set_to_none=True)
                logits = self.model(X)                          # (N, C)
                tc = _resolve_tc(target_class, logits)
                logits.gather(1, tc.unsqueeze(1)).squeeze(1).sum().backward()
            # concat 3 inception branches on channel axis: (N, 192, T_eff, 1)
            A = torch.cat([self._acts[n] for n in ("inp1", "inp2", "inp3")], dim=1)
            G = torch.cat([self._grads[n] for n in ("inp1", "inp2", "inp3")], dim=1)
            A = A.squeeze(-1); G = G.squeeze(-1)                # (N, C, T_eff)
            cam = self._cam_from(A, G).detach().cpu().numpy()   # (N, T_eff)
        finally:
            self._detach()
        T = X.shape[1]; T_eff = cam.shape[1]
        sal = np.zeros((cam.shape[0], T), dtype=np.float64)
        if T_eff >= T:
            sal = _interp_to_T(cam, T)
        else:
            # conv 가 본 영역만 채우고 앞쪽은 0
            sal[:, T - T_eff:] = cam
        return np.abs(sal)


def _interp_to_T(arr, T):
    return np.stack([np.interp(np.linspace(0, 1, T),
                               np.linspace(0, 1, arr.shape[1]), r) for r in arr])


def _trapz(curves, x):
    """행별 사다리꼴 적분. numpy 1.x(trapz)/2.x(trapezoid) 모두 지원."""
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    return fn(curves, x, axis=1)


def _resolve_tc(target_class, logits):
    B = logits.shape[0]
    if target_class is None or target_class == "predicted":
        return logits.argmax(-1)
    if isinstance(target_class, str):
        names = {
            "stationary": 0, "toxic_up": 1, "toxic_down": 2,
            "normal": 0, "vacuum": 1,
        }
        if target_class not in names:
            raise ValueError(
                "microstructure target must be predicted, stationary, "
                "toxic_up, toxic_down, normal, vacuum, or an integer"
            )
        target_class = names[target_class]
    c = int(target_class)
    if not 0 <= c < logits.shape[1]:
        raise ValueError(f"target class {c} outside model output [0,{logits.shape[1]})")
    return torch.full((B,), c, dtype=torch.long, device=logits.device)


# ====================================================================
# 2. Neighbor-conditional MC 대체기
# ====================================================================
def summary_stats(window, mask):
    """window:(T,40), mask:(T,) bool — True=X_U(유지구간). 3D 요약통계.
       feat0=mid추세 기울기, feat1=평균 spread, feat2=평균 L1 볼륨. (X_U 에서만)"""
    ask1p, ask1v, bid1p, bid1v = window[:, 0], window[:, 1], window[:, 2], window[:, 3]
    mid = 0.5 * (ask1p + bid1p)
    spread = ask1p - bid1p
    l1vol = ask1v + bid1v
    u = np.where(mask)[0]
    if len(u) < 2:
        return np.array([0.0, spread.mean(), l1vol.mean()])
    slope = np.polyfit(u.astype(float), mid[u], 1)[0]
    return np.array([slope, spread[u].mean(), l1vol[u].mean()])


def summary_stats_batch(windows, mask):
    """Vectorised ``summary_stats`` for a candidate pool using the same mask.

    Comparing a masked query with full-window candidate statistics is not a
    conditional-neighbour search.  Both sides must be summarized on the
    preserved context X_U, as stated in the manuscript.
    """
    windows = np.asarray(windows)
    u = np.flatnonzero(np.asarray(mask, dtype=bool))
    if len(u) == 0:
        raise ValueError("preserved context is empty")
    ask1p = windows[:, :, 0]
    ask1v = windows[:, :, 1]
    bid1p = windows[:, :, 2]
    bid1v = windows[:, :, 3]
    mid = 0.5 * (ask1p + bid1p)
    spread = ask1p - bid1p
    l1vol = ask1v + bid1v
    if len(u) < 2:
        slope = np.zeros(len(windows), dtype=np.float64)
    else:
        t = u.astype(np.float64)
        tc = t - t.mean()
        denom = float(np.dot(tc, tc))
        slope = (mid[:, u] * tc[None, :]).sum(axis=1) / max(denom, 1e-12)
    return np.column_stack((
        slope,
        spread[:, u].mean(axis=1),
        l1vol[:, u].mean(axis=1),
    ))


class NeighborConditionalSampler:
    """train 윈도우 풀에서 X_U 요약통계 kNN → 이웃의 X_S 구간을 빌려줌."""
    def __init__(self, pool_X, seed=0, n_neighbors=50):
        self.pool = np.asarray(pool_X)                        # (P, T, 40)
        if self.pool.ndim != 3 or self.pool.shape[2] < 4:
            raise ValueError("pool_X must have shape (P,T,F>=4)")
        if len(self.pool) == 0:
            raise ValueError("pool_X must not be empty")
        if int(n_neighbors) <= 0:
            raise ValueError("n_neighbors must be positive")
        if int(n_neighbors) > len(self.pool):
            raise ValueError(
                f"requested K={int(n_neighbors)} neighbours from a pool of "
                f"only {len(self.pool)} windows"
            )
        self.n_neighbors = int(n_neighbors)
        self.rng = np.random.default_rng(seed)
        # Prefix sums make same-mask candidate statistics O(P), not O(P*T),
        # for every deletion query.  They are computed in float64 because the
        # normalized price increments can be very small.
        mid = 0.5 * (self.pool[:, :, 0] + self.pool[:, :, 2])
        spread = self.pool[:, :, 0] - self.pool[:, :, 2]
        l1vol = self.pool[:, :, 1] + self.pool[:, :, 3]
        t = np.arange(self.pool.shape[1], dtype=np.float64)

        def prefix(a):
            z = np.zeros((len(a), 1), dtype=np.float64)
            return np.concatenate((z, np.cumsum(a, axis=1, dtype=np.float64)), axis=1)

        self._p_mid = prefix(mid)
        self._p_tmid = prefix(mid * t[None, :])
        self._p_spread = prefix(spread)
        self._p_l1vol = prefix(l1vol)
        self._t = t

    @staticmethod
    def _kept_sum(prefix, s, e):
        return prefix[:, -1] - (prefix[:, e] - prefix[:, s])

    def _pool_context_stats(self, s, e):
        T = self.pool.shape[1]
        n = T - (e - s)
        if n <= 0:
            raise ValueError("deletion span leaves no preserved context")
        sum_y = self._kept_sum(self._p_mid, s, e)
        sum_ty = self._kept_sum(self._p_tmid, s, e)
        sum_spread = self._kept_sum(self._p_spread, s, e)
        sum_l1vol = self._kept_sum(self._p_l1vol, s, e)
        keep = np.ones(T, dtype=bool)
        keep[s:e] = False
        tk = self._t[keep]
        sum_t = float(tk.sum())
        sum_t2 = float(np.dot(tk, tk))
        denom = n * sum_t2 - sum_t * sum_t
        slope = ((n * sum_ty - sum_t * sum_y) / denom
                 if n >= 2 and denom > 0 else np.zeros(len(self.pool)))
        return np.column_stack((slope, sum_spread / n, sum_l1vol / n))

    def sample(self, window, s, e, M):
        """window 의 [s,e) 를 이웃 M개의 같은 구간으로 대체한 M개 변형 반환."""
        T = window.shape[0]
        if not (0 <= int(s) <= int(e) <= T):
            raise ValueError(f"invalid replacement span [{s},{e}) for T={T}")
        if int(M) <= 0:
            raise ValueError("M must be positive")
        if s == e:
            return np.repeat(window[None], int(M), axis=0)
        if s == 0 and e == T:
            # Insertion starts from an unconditional on-manifold baseline: with
            # no preserved context, conditioning is undefined, so draw complete
            # real training windows uniformly from the declared donor pool.
            idx = self.rng.choice(len(self.pool), size=int(M),
                                  replace=(M > len(self.pool)))
            return self.pool[idx].copy()
        mask = np.ones(window.shape[0], dtype=bool); mask[s:e] = False
        pool_stats = self._pool_context_stats(s, e)
        q_stats = summary_stats(window, mask)
        mu = pool_stats.mean(axis=0)
        sd = pool_stats.std(axis=0) + 1e-8
        pool_z = (pool_stats - mu) / sd
        q = (q_stats - mu) / sd
        d = np.sqrt(((pool_z - q) ** 2).sum(1))              # (P,)
        k = self.n_neighbors
        near = np.argpartition(d, k - 1)[:k]
        nn_idx = self.rng.choice(near, size=int(M), replace=(M > k))
        variants = np.repeat(window[None], len(nn_idx), axis=0)  # (M, T, 40)
        for j, pi in enumerate(nn_idx):
            variants[j, s:e, :] = self.pool[pi, s:e, :]
        return variants


# ====================================================================
# 3. 연속구간 deletion sweep
# ====================================================================
def best_span(sal_row, L, mode, rng=None):
    """길이 L 연속구간 [s,s+L) 선택. mode: cam(합최대)/anti(합최소)/random."""
    T = len(sal_row)
    if L <= 0:
        return 0, 0
    L = min(L, T)
    sums = np.array([sal_row[s:s+L].sum() for s in range(T - L + 1)])
    if mode == "cam":
        s = int(sums.argmax())
    elif mode == "anti":
        s = int(sums.argmin())
    elif mode == "random":
        if rng is None:
            raise ValueError("random span selection requires an explicit seeded RNG")
        s = int(rng.integers(0, T - L + 1))
    else:
        raise ValueError("mode must be 'cam', 'anti', or 'random'")
    return s, s + L


@torch.no_grad()
def mc_mean_probs(model, variants, device):
    """variants:(M,T,40) → MC 평균 클래스 확률 벡터 (n_classes,)."""
    X = torch.from_numpy(variants).float().to(device)
    p = F.softmax(model(X), dim=-1).mean(0)          # (n_classes,)
    return p.cpu().numpy()


def deletion_curve(model, sampler, provider, device, window, target_c,
                   q_grid, mode, M, saliency=None):
    """한 윈도우의 deletion: q마다 타깃확률 곡선 + 마지막 q에서 argmax-flip 여부.
       flip = (q_max 까지 지운 뒤 MC 평균 예측의 argmax 가 target_c 를 벗어남)."""
    T = window.shape[0]
    Xt = torch.from_numpy(window[None]).float()
    if saliency is None:
        sal = (np.zeros(T, dtype=float) if mode == "random" else
               provider.saliency(Xt, int(target_c))[0])
    else:
        sal = np.asarray(saliency)
    if sal.shape != (T,):
        raise ValueError(f"saliency must have shape ({T},), got {sal.shape}")
    probs = []
    last_argmax = target_c
    for q in q_grid:
        if q <= 0:
            with torch.no_grad():
                p0 = F.softmax(model(Xt.to(device)), -1)[0].cpu().numpy()
            probs.append(float(p0[target_c]))
            last_argmax = int(p0.argmax())
            continue
        L = max(1, int(np.ceil(q * T)))
        s, e = best_span(sal, L, mode, rng=sampler.rng)
        variants = sampler.sample(window, s, e, M)
        pvec = mc_mean_probs(model, variants, device)        # (n_classes,)
        probs.append(float(pvec[target_c]))
        last_argmax = int(pvec.argmax())
    flipped = (last_argmax != target_c)
    return np.array(probs), flipped


def insertion_curve(model, sampler, provider, device, window, target_c,
                    q_grid, mode, M, saliency=None, base_variants=None):
    """한 윈도우의 insertion(충분성) 곡선.

    rho=0 기준선은 보존 context가 없어 이웃 조건화가 정의되지 않으므로,
    선언된 training donor pool에서의 **무조건부 전체-창 추출**이다
    (``sampler.sample(window, 0, T, M)`` 경로).  각 rho>0 에서는 길이
    ceil(rho*T)의 연속 구간을 mode 규칙(cam/random/anti)으로 선택해 그
    구간만 원본으로 복원한다.  deletion과 동일하게 rho마다 구간을 독립
    재선택하는 **비중첩 곡선**이며 단조 회복 경로로 해석하면 안 된다.

    base_variants가 주어지면 세 mode가 동일한 donor 기준선을 공유하여
    mode 간 비교가 짝지어진다(권장).
    recovery = (q_max에서 MC 평균 예측의 argmax가 target_c로 복귀)."""
    T = window.shape[0]
    Xt = torch.from_numpy(window[None]).float()
    if saliency is None:
        sal = (np.zeros(T, dtype=float) if mode == "random" else
               provider.saliency(Xt, int(target_c))[0])
    else:
        sal = np.asarray(saliency)
    if sal.shape != (T,):
        raise ValueError(f"saliency must have shape ({T},), got {sal.shape}")
    if base_variants is None:
        base_variants = sampler.sample(window, 0, T, M)
    base_variants = np.asarray(base_variants, dtype=window.dtype)
    if base_variants.shape[1:] != window.shape:
        raise ValueError("base_variants must match the window shape")
    probs = []
    last_argmax = None
    for q in q_grid:
        if q <= 0:
            pvec = mc_mean_probs(model, base_variants, device)
            probs.append(float(pvec[target_c]))
            last_argmax = int(pvec.argmax())
            continue
        L = max(1, int(np.ceil(q * T)))
        s, e = best_span(sal, L, mode, rng=sampler.rng)
        variants = base_variants.copy()
        variants[:, s:e, :] = window[s:e, :]
        pvec = mc_mean_probs(model, variants, device)
        probs.append(float(pvec[target_c]))
        last_argmax = int(pvec.argmax())
    recovered = (last_argmax == target_c)
    return np.array(probs), recovered


# ====================================================================
# 4. 데이터 로딩 / demo
# ====================================================================
def build_ours(ckpt, device, task):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = ck.get("config", {}) or {}
    meta = ck.get("meta", {}) or {}
    ck_task = cfg.get("task") or meta.get("task")
    if ck_task and ck_task != task:
        raise ValueError(f"checkpoint task={ck_task!r}, requested task={task!r}")
    nc = cfg.get("n_classes") or meta.get("n_classes") or len(TASK_CLASSES[task])
    if int(nc) != len(TASK_CLASSES[task]):
        raise ValueError(
            f"checkpoint n_classes={nc} does not match task={task!r}"
        )
    bidx = meta.get("branch_indices") or make_branching_scheme(
        cfg.get("branching_scheme", "type4"), n_features=40)
    m = LOBBranchModel(n_features=40, n_classes=int(nc),
                       backbone=cfg.get("backbone", "tcn"), pool=cfg.get("pool", "avg"),
                       cnn_channels=tuple(cfg.get("cnn_channels", (32, 32, 64))),
                       cnn_kernel=cfg.get("cnn_kernel", 3),
                       cnn_dropout=cfg.get("cnn_dropout", 0.1),
                       fc_hidden=tuple(cfg.get("fc_hidden", (64,))),
                       branch_indices=bidx,
                       tcn_dilation_base=cfg.get("tcn_dilation_base", 2),
                       input_norm=cfg.get("input_norm", "identity"),
                       tcn_norm=cfg.get("tcn_norm", "none"),
                       interaction_channels=int(
                           cfg.get("interaction_channels", 0)
                       ),
                       seq_len=int(cfg.get("seq_len", 100)),
                       pool_windows=tuple(
                           cfg.get("pool_windows", (8, 32, 100))
                       )).to(device)
    m.load_state_dict(ck["model_state"], strict=True)
    m.eval()
    return m, int(nc)


def build_dl(ckpt, device, task):
    from models.deeplob import build_deeplob
    from lob_xai.paper_protocol import deeplob_parameter_count
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = ck.get("config", {}) or {}
    meta = ck.get("meta", {}) or {}
    stored_arch = ck.get("arch") or cfg.get("model") or meta.get("model")
    if stored_arch is not None and stored_arch != "deeplob":
        raise ValueError(
            f"checkpoint architecture={stored_arch!r}, expected 'deeplob'"
        )
    ck_task = cfg.get("task") or meta.get("task")
    if ck_task and ck_task != task:
        raise ValueError(f"checkpoint task={ck_task!r}, requested task={task!r}")
    nc = int(cfg.get("n_classes") or meta.get("n_classes") or len(TASK_CLASSES[task]))
    if nc != len(TASK_CLASSES[task]):
        raise ValueError(
            f"checkpoint n_classes={nc} does not match task={task!r}"
        )
    model_cfg = cfg.get("model_config") or {}
    n_features = int(
        model_cfg.get("n_features", cfg.get("n_features", meta.get("n_features", 40)))
    )
    m = build_deeplob(n_classes=nc, n_features=n_features).to(device)
    actual_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    expected_params = deeplob_parameter_count(nc)
    if actual_params != expected_params:
        raise RuntimeError(
            f"DeepLOB parameter count={actual_params} != declared {expected_params}"
        )
    state = ck.get("model_state", ck)
    m.load_state_dict(state, strict=True)
    m.eval()
    return m, nc


def load_windows(csv, task, seq_len, k, tick, limit=None, seed=0,
                 boundaries=None, subset="all", val_ratio=DATA.validation_fraction_within_development_file):
    from lob_xai.data import load_fi2010_features_only, FI2010MicrostructureDataset
    from lob_xai.microstructure_labels import make_segment_id, split_within_segments
    X = load_fi2010_features_only(csv, n_features=40)
    seg = make_segment_id(len(X), boundaries or [])
    if subset not in {"all", "train", "validation"}:
        raise ValueError("subset must be all, train, or validation")
    if subset in {"train", "validation"}:
        Xtr, str_, Xva, sva = split_within_segments(X, seg, val_ratio)
        X, seg = (Xtr, str_) if subset == "train" else (Xva, sva)
    ds = FI2010MicrostructureDataset(
        X, segment_id=seg, task=task, seq_len=seq_len, k=k, tick_size=tick
    )
    n = len(ds); idx = np.arange(n)
    if limit and limit < n:
        idx = np.random.default_rng(seed).choice(n, limit, replace=False)
    Xs = np.stack([ds[i][0].numpy() for i in idx])
    ys = np.array([int(ds[i][1]) for i in idx])
    return Xs, ys


def make_demo(task, seq_len=100, n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, seq_len, 40), np.float32)
    base = np.cumsum(rng.normal(0, 0.02, (n, seq_len)), axis=1)
    for i in range(n):
        j = rng.integers(55, 85); base[i, j:] += rng.choice([1, -1]) * rng.uniform(.1, .3)
    for lvl in range(10):
        X[:, :, 4*lvl] = base + .01*(lvl+1); X[:, :, 4*lvl+2] = base - .01*(lvl+1)
        X[:, :, 4*lvl+1] = rng.normal(0, 1, (n, seq_len))
        X[:, :, 4*lvl+3] = rng.normal(0, 1, (n, seq_len))
    y = rng.integers(0, len(TASK_CLASSES[task]), n)
    return X.astype(np.float32), y


# ====================================================================
# 4b. method 비교 (Grad-CAM / HiResCAM / TSR)
# ====================================================================
COMPARE_METHODS = ["gradcam", "hirescam", "tsr"]
METHOD_STYLE = {
    "gradcam":  dict(color="#27ae60", lw=2.4, ls="-",  label="Grad-CAM"),
    "hirescam": dict(color="#c0392b", lw=2.6, ls="-",  label="HiResCAM"),
    "tsr":      dict(color="#8e44ad", lw=2.2, ls="-.", label="TSR"),
}


def _make_provider(model_kind, model, device, method):
    return (OursCAMProvider(model, device, method) if model_kind == "ours"
            else DeepLOBGradCAMProvider(model, device, method))


def run_compare_methods(args, model, sampler, device, classes, flat,
                        q_grid, test_X, test_y, probs):
    """세 method 각각의 CAM-order deletion 을 겹쳐 비교.
       random 은 method 무관이라 한 번만 계산해 공통 기준선으로 둔다."""
    target_classes = [c for c in range(len(classes)) if c != flat]
    # class -> dict(method -> (W,Q) cam-curve, 'random' -> (W,Q), '_auc', '_flip')
    results = {}
    for c in target_classes:
        sel = select_top_confidence_alerts(
            probs, test_y, c, args.n_windows,
            require_correct=not args.pred_based,
        )
        if len(sel) == 0:
            preds = probs.argmax(1)
            n_true = int((test_y == c).sum()); n_pred = int((preds == c).sum())
            print(f"[skip] class {classes[c]}: 정타 0개 "
                  f"(true={n_true}, pred={n_pred}). --pred-based 사용 가능")
            continue

        providers = {m: _make_provider(args.model, model, device, m)
                     for m in COMPARE_METHODS}
        cam_curves = {m: [] for m in COMPARE_METHODS}
        cam_flips = {m: [] for m in COMPARE_METHODS}
        rand_curves = []
        rand_flips = []
        for wi in sel:
            # random 기준선 (method 무관) — 윈도우마다 한 번
            cur_r, flipped_r = deletion_curve(
                model, sampler, providers["hirescam"], device,
                test_X[wi], c, q_grid, "random", args.mc,
            )
            rand_curves.append(cur_r)
            rand_flips.append(flipped_r)
            for m in COMPARE_METHODS:
                sal = providers[m].saliency(
                    torch.from_numpy(test_X[wi][None]).float(), int(c)
                )[0]
                cur, flipped = deletion_curve(model, sampler, providers[m], device,
                                              test_X[wi], c, q_grid, "cam", args.mc,
                                              saliency=sal)
                cam_curves[m].append(cur); cam_flips[m].append(flipped)
        entry = {m: np.stack(cam_curves[m]) for m in COMPARE_METHODS}
        entry["random"] = np.stack(rand_curves)
        entry["_flip"] = {m: np.array(cam_flips[m]) for m in COMPARE_METHODS}
        entry["_flip_random"] = np.asarray(rand_flips, dtype=bool)
        entry["_indices"] = np.asarray(sel)
        entry["_confidence"] = probs[sel, c]
        results[c] = entry
        print(f"[done] class {classes[c]}: {len(sel)} windows × {len(COMPARE_METHODS)} methods")

    if not results:
        print("[warn] 그릴 클래스가 없음"); return

    # --- 집계: method별 ΔAUC(=random-method) + Wilcoxon + flip ---
    rows = []
    window_rows = []
    for c, e in results.items():
        auc_rand = _trapz(e["random"], q_grid)
        for m in COMPARE_METHODS:
            auc_m = _trapz(e[m], q_grid)
            dAUC = auc_rand - auc_m
            try:
                _, pval = wilcoxon(dAUC, alternative="greater")
            except ValueError:
                pval = np.nan
            rows.append(dict(model=args.model, task=args.task, cls=classes[c],
                             method=m, n=e[m].shape[0],
                             selection=("top_confidence_predicted" if args.pred_based
                                        else "top_confidence_correct"),
                             checkpoint_sha256=args.checkpoint_sha256,
                             seed=args.seed, neighbor_candidates=args.neighbors_k,
                             mc_samples=args.mc,
                             auc_method=auc_m.mean(), auc_random=auc_rand.mean(),
                             dAUC_mean=dAUC.mean(), wilcoxon_p=pval,
                             flip_method=float(e["_flip"][m].mean()),
                             flip_random=float(e["_flip_random"].mean())))
            for rank, (wi, conf, am, ar, fl, flr) in enumerate(zip(
                    e["_indices"], e["_confidence"], auc_m, auc_rand,
                    e["_flip"][m], e["_flip_random"]), 1):
                window_rows.append(dict(
                    model=args.model, task=args.task, cls=classes[c], method=m,
                    selection_rank=rank, window_index=int(wi),
                    confidence=float(conf), auc_method=float(am),
                    auc_random=float(ar), dAUC=float(ar - am),
                    flipped_method=bool(fl), flipped_random=bool(flr), seed=args.seed,
                    checkpoint_sha256=args.checkpoint_sha256,
                ))
            print(f"  [{classes[c]}/{m}] ΔAUC={dAUC.mean():.4f} "
                  f"p={pval:.2e} flip(method)={float(e['_flip'][m].mean()):.2f} "
                  f"flip(random)={float(e['_flip_random'].mean()):.2f}")

    # --- figure: 클래스별 패널, 세 method cam-curve + random 기준선 ---
    nC = len(results)
    fig, axes = plt.subplots(1, nC, figsize=(6.2*nC, 4.6), squeeze=False)
    for ax, (c, e) in zip(axes[0], results.items()):
        for m in COMPARE_METHODS:
            mean = e[m].mean(0); se = e[m].std(0) / np.sqrt(len(e[m]))
            ax.plot(q_grid*100, mean, **METHOD_STYLE[m])
            ax.fill_between(q_grid*100, mean-se, mean+se,
                            color=METHOD_STYLE[m]["color"], alpha=0.10)
        rmean = e["random"].mean(0)
        ax.plot(q_grid*100, rmean, color="#7f8c8d", lw=2.0, ls="--",
                label="random (baseline)")
        ax.set_title(f"{classes[c]}  (n={e['gradcam'].shape[0]})", fontweight="bold")
        ax.set_xlabel("deleted fraction of window  (%)")
        ax.set_ylabel("mean target-class probability")
        ax.grid(alpha=0.2); ax.legend(fontsize=10)
    mname = {"ours": "4-branch TCN", "deeplob": "DeepLOB"}[args.model]
    fig.suptitle(f"CAM method comparison — {mname}, {args.task}\n"
                 "lower curve vs random  =  more faithful  (Grad-CAM / HiResCAM / TSR)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = args.out or f"figures/cf_methods_{args.model}_{args.task}.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"[wrote] {out}")
    plt.close(fig)
    _write_csv(out.replace(".png", ".csv"), rows)
    _write_csv(str(Path(out).with_suffix("")) + ".windows.csv", window_rows)


# ====================================================================
# 5. CLI workflow
# ====================================================================
def _parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["ours", "deeplob"])
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--train-csv", default="data/FI2010/FI2010_train.csv")
    ap.add_argument("--test-csv", default="data/FI2010/FI2010_test.csv")
    ap.add_argument("--task", default="toxic", choices=["toxic", "vacuum"])
    ap.add_argument("--method", default=COUNTERFACTUAL.cam_method,
                    choices=SUPPORTED_METHODS,
                    help="CAM method (paper default: hirescam)")
    ap.add_argument("--pool-size", type=int, default=COUNTERFACTUAL.pool_size,
                    help="train 이웃 풀 크기")
    ap.add_argument("--n-windows", type=int, default=COUNTERFACTUAL.n_windows,
                    help="클래스당 검증 윈도우 수")
    ap.add_argument("--mc", type=int, default=COUNTERFACTUAL.mc_samples,
                    help="윈도우당 MC 샘플 수 M")
    ap.add_argument("--neighbors-k", type=int, default=COUNTERFACTUAL.neighbor_candidates,
                    help="보존 컨텍스트 기준 후보 이웃 수 K")
    ap.add_argument("--q-max", type=float, default=0.5)
    ap.add_argument("--q-step", type=float, default=0.05)
    ap.add_argument("--seq-len", type=int, default=DATA.seq_len)
    ap.add_argument("--k", type=int, default=None,
                    help="default: toxic=10, vacuum=30; checkpoint metadata takes precedence")
    ap.add_argument("--tick-size", type=float, default=DATA.tick_size_z)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--seed", type=int, default=COUNTERFACTUAL.seeds[0])
    ap.add_argument("--pred-based", action="store_true",
                    help="정타(pred==true) 대신 '모델이 그 클래스로 예측한' 윈도우 사용 "
                         "(정타가 너무 적은 클래스 검증용)")
    ap.add_argument("--allow-unverified-data", action="store_true",
                    help="diagnostic only: allow legacy checkpoints without data SHA-256")
    ap.add_argument("--compare-methods", action="store_true",
                    help="Grad-CAM / HiResCAM / TSR 세 method 의 CAM-deletion 을 "
                         "한 번에 비교 (random 공통 기준선)")
    ap.add_argument("--direction", default="deletion",
                    choices=["deletion", "insertion", "both"],
                    help="deletion=필요성(기본, 동결 프로토콜), insertion=충분성 "
                         "(무조건부 donor 기준선에서 구간 복원), both=둘 다")
    args = ap.parse_args(argv)

    if not args.demo and args.ckpt is None:
        ap.error("--ckpt is required for empirical evaluation; use --demo explicitly for synthetic data")
    if (not args.demo and
            Path(args.train_csv).expanduser().resolve() == Path(args.test_csv).expanduser().resolve()):
        ap.error("train-csv and test-csv must be different files")
    if args.k is None:
        args.k = DATA.toxic_horizon if args.task == "toxic" else DATA.vacuum_horizon
    if args.pool_size <= 0 or args.n_windows <= 0 or args.mc <= 0 or args.neighbors_k <= 0:
        ap.error("pool-size, n-windows, mc, and neighbors-k must all be positive")

    if args.q_max <= 0 or args.q_max > 1 or args.q_step <= 0:
        ap.error("require 0 < q-max <= 1 and q-step > 0")
    if np.isclose(args.q_max, 0.5) and np.isclose(args.q_step, 0.05):
        q_grid = np.asarray(COUNTERFACTUAL.curve_grid)
    else:
        q_grid = np.arange(0, args.q_max + 1e-9, args.q_step)

    return ap, args, q_grid


def _load_model_and_data(args, ap, device):
    """Verify provenance before loading the model and donor/test windows."""
    classes = TASK_CLASSES[args.task]
    if args.demo:
        print("[demo] 랜덤 모델 + 합성 윈도우")
        test_X, test_y = make_demo(args.task, args.seq_len, n=max(args.n_windows, 200))
        pool_X, _ = make_demo(args.task, args.seq_len,
                              n=min(args.pool_size, 2000), seed=args.seed + 1)
        if args.model == "ours":
            bidx = make_branching_scheme("type4", 40)
            model = LOBBranchModel(n_features=40, n_classes=len(classes),
                                   backbone="tcn", pool="avg",
                                   cnn_channels=(32, 32, 64), cnn_kernel=3,
                                   branch_indices=bidx).to(device).eval()
        else:
            from models.deeplob import build_deeplob
            model = build_deeplob(n_classes=len(classes)).to(device).eval()
    else:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        ck_cfg = {**dict(ck.get("config", {}) or {}), **dict(ck.get("meta", {}) or {})}
        identities = ck_cfg.get("data_files") or {}
        if not identities:
            if not args.allow_unverified_data:
                ap.error(
                    "checkpoint has no data SHA-256 provenance; retrain with the "
                    "paper trainer or use --allow-unverified-data for diagnostics only"
                )
        else:
            verify_file_identity(args.train_csv, identities.get("development", {}))
            verify_file_identity(args.test_csv, identities.get("test", {}))
        checkpoint_sha256 = sha256_file(args.ckpt)
        args.seq_len = int(ck_cfg.get("seq_len", args.seq_len))
        args.k = int(ck_cfg.get("k", args.k))
        args.tick_size = float(ck_cfg.get("tick_size", args.tick_size))
        val_ratio = float(ck_cfg.get(
            "val_ratio", DATA.validation_fraction_within_development_file
        ))
        train_boundaries = list(ck_cfg.get("train_boundaries", []) or [])
        test_boundaries = list(ck_cfg.get("test_boundaries", []) or [])
        if args.model == "ours":
            model, _ = build_ours(args.ckpt, device, args.task)
        else:
            model, _ = build_dl(args.ckpt, device, args.task)
        test_X, test_y = load_windows(args.test_csv, args.task, args.seq_len,
                                      args.k, args.tick_size,
                                      boundaries=test_boundaries)
        pool_X, _ = load_windows(args.train_csv, args.task, args.seq_len,
                                 args.k, args.tick_size, limit=args.pool_size,
                                 seed=args.seed, boundaries=train_boundaries,
                                 subset="train", val_ratio=val_ratio)
        if len(pool_X) != args.pool_size:
            ap.error(
                f"requested donor pool size {args.pool_size}, but only "
                f"{len(pool_X)} eligible training windows exist"
            )

    args.checkpoint_sha256 = "DEMO" if args.demo else checkpoint_sha256
    return model, test_X, test_y, pool_X


def _predict_probabilities(model, test_X, device):
    with torch.no_grad():
        probs_parts = []
        for i in range(0, len(test_X), 512):
            xb = torch.from_numpy(test_X[i:i+512]).float().to(device)
            probs_parts.append(F.softmax(model(xb), dim=-1).cpu().numpy())
    return np.concatenate(probs_parts)


def _collect_curves(args, model, sampler, device, q_grid, test_X, test_y, probs):
    """Keep donor sampling and mode order paired exactly as in the protocol."""
    classes = TASK_CLASSES[args.task]
    flat = TASK_FLAT[args.task]
    preds = probs.argmax(1)
    provider = _make_provider(args.model, model, device, args.method)
    # --- 클래스별 deletion/insertion sweep ---
    do_del = args.direction in ("deletion", "both")
    do_ins = args.direction in ("insertion", "both")
    results = {}     # class_idx -> dict(cam/random/anti -> (W, len(q)) curves)
    ins_results = {}
    target_classes = [c for c in range(len(classes)) if c != flat]  # non-flat
    for c in target_classes:
        sel = select_top_confidence_alerts(
            probs, test_y, c, args.n_windows,
            require_correct=not args.pred_based,
        )
        if len(sel) == 0:
            n_true = int((test_y == c).sum())
            n_pred = int((preds == c).sum())
            n_corr = int(((preds == c) & (test_y == c)).sum())
            print(f"[skip] class {classes[c]}: 정타 윈도우 0개 "
                  f"(test에 true={n_true}, 모델이 이 클래스로 예측={n_pred}, "
                  f"그중 정타={n_corr}). "
                  f"--pred-based 로 '예측 기준' 윈도우 사용 가능")
            continue
        curves = {m: [] for m in ("cam", "random", "anti")}
        flips = {m: [] for m in ("cam", "random", "anti")}
        ins_curves = {m: [] for m in ("cam", "random", "anti")}
        recoveries = {m: [] for m in ("cam", "random", "anti")}
        for wi in sel:
            sal = provider.saliency(
                torch.from_numpy(test_X[wi][None]).float(), int(c)
            )[0]
            if do_del:
                for mode in ("cam", "random", "anti"):
                    cur, flipped = deletion_curve(model, sampler, provider, device,
                                                  test_X[wi], c, q_grid, mode, args.mc,
                                                  saliency=(None if mode == "random" else sal))
                    curves[mode].append(cur)
                    flips[mode].append(flipped)
            if do_ins:
                # One shared unconditional donor baseline per window keeps the
                # three modes paired on identical rho=0 states.
                T_w = test_X[wi].shape[0]
                base = sampler.sample(test_X[wi], 0, T_w, args.mc)
                for mode in ("cam", "random", "anti"):
                    cur, recovered = insertion_curve(
                        model, sampler, provider, device,
                        test_X[wi], c, q_grid, mode, args.mc,
                        saliency=(None if mode == "random" else sal),
                        base_variants=base,
                    )
                    ins_curves[mode].append(cur)
                    recoveries[mode].append(recovered)
        if do_del:
            results[c] = {m: np.stack(v) for m, v in curves.items()}
            results[c]["_flip"] = {m: np.array(flips[m]) for m in flips}
            results[c]["_indices"] = np.asarray(sel)
            results[c]["_confidence"] = probs[sel, c]
        if do_ins:
            ins_results[c] = {m: np.stack(v) for m, v in ins_curves.items()}
            ins_results[c]["_recovered"] = {m: np.array(recoveries[m])
                                            for m in recoveries}
            ins_results[c]["_indices"] = np.asarray(sel)
            ins_results[c]["_confidence"] = probs[sel, c]
        print(f"[done] class {classes[c]}: {len(sel)} windows "
              f"(direction={args.direction})")

    return results, ins_results


def _summarize_deletion(args, classes, q_grid, results, pool_size):
    """Deletion gain is AUC(random) - AUC(CAM); positive is more faithful."""
    # --- 집계 + 통계 ---
    rows = []
    window_rows = []
    for c, cur in results.items():
        modes = ("cam", "random", "anti")
        # per-window deletion AUC (작을수록 빨리 떨어짐 = faithful)
        auc = {m: _trapz(cur[m], q_grid) for m in modes}
        dAUC = auc["random"] - auc["cam"]                 # >0 이면 cam 이 더 가파름
        try:
            stat, pval = wilcoxon(dAUC, alternative="greater")
        except ValueError:
            stat, pval = np.nan, np.nan
        # flip rate (argmax 기반): q_max 까지 지운 뒤 MC 평균 예측의 argmax 가
        # 타깃 클래스를 벗어난 윈도우 비율. 3-class 에서도 정확. CAM/random 둘 다 보고.
        flip_cam = float(cur["_flip"]["cam"].mean())
        flip_rand = float(cur["_flip"]["random"].mean())
        rows.append(dict(model=args.model, task=args.task, cls=classes[c],
                         n=cur["cam"].shape[0],
                         selection=("top_confidence_predicted" if args.pred_based
                                    else "top_confidence_correct"),
                         checkpoint_sha256=args.checkpoint_sha256,
                         seed=args.seed, neighbor_candidates=args.neighbors_k,
                         mc_samples=args.mc, pool_size=pool_size,
                         auc_cam=auc["cam"].mean(), auc_rand=auc["random"].mean(),
                         auc_anti=auc["anti"].mean(),
                         dAUC_mean=dAUC.mean(), wilcoxon_p=pval,
                         flip_cam=flip_cam, flip_random=flip_rand))
        for rank, wi in enumerate(cur["_indices"], 1):
            j = rank - 1
            window_rows.append(dict(
                model=args.model, task=args.task, cls=classes[c],
                selection_rank=rank, window_index=int(wi),
                confidence=float(cur["_confidence"][j]), seed=args.seed,
                checkpoint_sha256=args.checkpoint_sha256,
                auc_cam=float(auc["cam"][j]),
                auc_random=float(auc["random"][j]),
                auc_anti=float(auc["anti"][j]),
                dAUC=float(dAUC[j]),
                flip_cam=bool(cur["_flip"]["cam"][j]),
                flip_random=bool(cur["_flip"]["random"][j]),
                flip_anti=bool(cur["_flip"]["anti"][j]),
            ))
        print(f"  [{classes[c]}] ΔAUC={dAUC.mean():.4f} p={pval:.2e} "
              f"flip(cam)={flip_cam:.2f} flip(rand)={flip_rand:.2f}")

    return rows, window_rows


def _summarize_insertion(args, classes, q_grid, ins_results, pool_size):
    """Insertion gain is AUC(CAM) - AUC(random); positive is more sufficient."""
    ins_rows = []
    ins_window_rows = []
    for c, cur in ins_results.items():
        modes = ("cam", "random", "anti")
        auc = {m: _trapz(cur[m], q_grid) for m in modes}
        dAUC = auc["cam"] - auc["random"]        # >0 이면 cam 복원이 더 빠름
        try:
            stat, pval = wilcoxon(dAUC, alternative="greater")
        except ValueError:
            stat, pval = np.nan, np.nan
        rec = {m: float(cur["_recovered"][m].mean()) for m in modes}
        ins_rows.append(dict(
            model=args.model, task=args.task, cls=classes[c],
            direction="insertion",
            n=cur["cam"].shape[0],
            selection=("top_confidence_predicted" if args.pred_based
                       else "top_confidence_correct"),
            baseline="unconditional_full_window_train_donor",
            checkpoint_sha256=args.checkpoint_sha256,
            seed=args.seed, neighbor_candidates=args.neighbors_k,
            mc_samples=args.mc, pool_size=pool_size,
            ins_auc_cam=auc["cam"].mean(), ins_auc_rand=auc["random"].mean(),
            ins_auc_anti=auc["anti"].mean(),
            ins_dAUC_mean=dAUC.mean(), wilcoxon_p=pval,
            recovery_cam=rec["cam"], recovery_random=rec["random"],
            recovery_anti=rec["anti"]))
        for rank, wi in enumerate(cur["_indices"], 1):
            j = rank - 1
            ins_window_rows.append(dict(
                model=args.model, task=args.task, cls=classes[c],
                direction="insertion",
                selection_rank=rank, window_index=int(wi),
                confidence=float(cur["_confidence"][j]), seed=args.seed,
                checkpoint_sha256=args.checkpoint_sha256,
                ins_auc_cam=float(auc["cam"][j]),
                ins_auc_random=float(auc["random"][j]),
                ins_auc_anti=float(auc["anti"][j]),
                ins_dAUC=float(dAUC[j]),
                recovered_cam=bool(cur["_recovered"]["cam"][j]),
                recovered_random=bool(cur["_recovered"]["random"][j]),
                recovered_anti=bool(cur["_recovered"]["anti"][j]),
            ))
        print(f"  [ins {classes[c]}] ΔAUC={dAUC.mean():.4f} p={pval:.2e} "
              f"recover(cam)={rec['cam']:.2f} recover(rand)={rec['random']:.2f}")

    return ins_rows, ins_window_rows


def _write_csv(path, rows):
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[wrote] {path}")


def _write_curve_outputs(args, classes, q_grid, results, rows, window_rows, direction):
    """Write the existing figure, summary, per-window table, and curve schemas."""
    n_classes = len(results)
    fig, axes = plt.subplots(1, n_classes, figsize=(6.2*n_classes, 4.6), squeeze=False)
    styles = dict(cam=dict(color="#c0392b", lw=2.6, label="CAM order"),
                  random=dict(color="#7f8c8d", lw=2.2, ls="--", label="random"),
                  anti=dict(color="#2980b9", lw=2.0, ls=":", label="anti-CAM"))
    action = "deleted" if direction == "deletion" else "restored"
    for ax, (c, curves) in zip(axes[0], results.items()):
        for mode in ("cam", "random", "anti"):
            mean = curves[mode].mean(0)
            se = curves[mode].std(0) / np.sqrt(len(curves[mode]))
            ax.plot(q_grid*100, mean, **styles[mode])
            ax.fill_between(q_grid*100, mean-se, mean+se,
                            color=styles[mode]["color"], alpha=0.12)
        ax.set_title(f"{classes[c]}  (n={curves['cam'].shape[0]})", fontweight="bold")
        ax.set_xlabel(f"{action} fraction of window  (%)")
        ax.set_ylabel("mean target-class probability")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=11)
    model_name = {"ours": "4-branch TCN", "deeplob": "DeepLOB"}[args.model]
    interpretation = (
        "lower CAM curve vs random = more faithful" if direction == "deletion"
        else "baseline = unconditional train-donor windows; "
             "higher CAM curve vs random = sufficient"
    )
    fig.suptitle(
        f"Counterfactual {direction} — {model_name}, {args.task}\n"
        f"{args.method}; {interpretation}", fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = args.out or f"figures/counterfactual_{args.model}_{args.task}.png"
    if direction == "insertion":
        out = str(Path(out).with_suffix("")) + "_insertion.png"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[wrote] {out}")
    _write_csv(out.replace(".png", ".csv"), rows)
    stem = str(Path(out).with_suffix(""))
    _write_csv(stem + ".windows.csv", window_rows)
    arrays = dict(
        direction=np.asarray(direction),
        model=np.asarray(args.model), task=np.asarray(args.task),
        seed=np.asarray(args.seed), method=np.asarray(args.method),
        checkpoint_sha256=np.asarray(args.checkpoint_sha256 or ""),
        q_grid=q_grid, classes=np.asarray([classes[c] for c in results]),
    )
    for c, curves in results.items():
        for mode in ("cam", "random", "anti"):
            arrays[f"{classes[c]}__{mode}"] = curves[mode].astype(np.float32)
        arrays[f"{classes[c]}__indices"] = curves["_indices"].astype(np.int64)
        arrays[f"{classes[c]}__confidence"] = np.asarray(
            curves["_confidence"], dtype=np.float32,
        )
    curves_out = stem + ".curves.npz"
    np.savez_compressed(curves_out, **arrays)
    print(f"[wrote] {curves_out}")


def main():
    parser, args, q_grid = _parse_args()
    device = torch.device(args.device)
    classes = TASK_CLASSES[args.task]
    model, test_X, test_y, pool_X = _load_model_and_data(args, parser, device)
    sampler = NeighborConditionalSampler(
        pool_X, seed=args.seed, n_neighbors=args.neighbors_k,
    )
    probs = _predict_probabilities(model, test_X, device)
    if args.compare_methods:
        run_compare_methods(
            args, model, sampler, device, classes, TASK_FLAT[args.task],
            q_grid, test_X, test_y, probs,
        )
        return
    deletion, insertion = _collect_curves(
        args, model, sampler, device, q_grid, test_X, test_y, probs,
    )
    if deletion:
        rows, windows = _summarize_deletion(args, classes, q_grid, deletion, len(pool_X))
        _write_curve_outputs(args, classes, q_grid, deletion, rows, windows, "deletion")
    if insertion:
        rows, windows = _summarize_insertion(args, classes, q_grid, insertion, len(pool_X))
        _write_curve_outputs(args, classes, q_grid, insertion, rows, windows, "insertion")


if __name__ == "__main__":
    main()
