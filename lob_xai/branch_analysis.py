#!/usr/bin/env python3
"""Feature replacement and branch attribution for the revised manuscript.

BranchAdapter uses a training-mean internal feature map (or diagnostic zero
baseline), keeping the input unchanged. Axis A compares drop-one logit changes
with absolute HiResCAM shares on the same correct-alert cohort.

probe_branch_structure.py verifies the branch module names and architecture
signature before Axis A or branch-importance analysis uses this adapter.
"""
from __future__ import annotations
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn as nn

BRANCH_NAMES = ["ask_price", "ask_volume", "bid_price", "bid_volume"]


def architecture_signature(model) -> str:
    """Hash module types and state shapes, independent of learned weights."""
    payload = {
        "modules": [(name, type(module).__module__, type(module).__qualname__)
                    for name, module in model.named_modules()],
        "state_shapes": [(name, list(value.shape), str(value.dtype))
                         for name, value in model.state_dict().items()],
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def validate_run_protocol(config, *, task, k, seed=None, label="checkpoint"):
    """Prevent an explicit checkpoint from being mislabeled by CLI arguments."""
    expected = {"task": task, "k": int(k)}
    if seed is not None:
        expected["seed"] = int(seed)
    missing = [key for key in expected if config.get(key) is None]
    if missing:
        raise ValueError(f"{label} lacks protocol fields: {missing}")
    mismatch = {
        key: {"stored": config[key], "requested": value}
        for key, value in expected.items()
        if (int(config[key]) != value if key in {"k", "seed"}
            else config[key] != value)
    }
    if mismatch:
        raise ValueError(f"{label} protocol mismatch: {mismatch}")


# ---------------------------------------------------------------------------
# module helpers
# ---------------------------------------------------------------------------
def get_module(model, dotted):
    """Resolve a module by its dotted name from named_modules()."""
    table = dict(model.named_modules())
    if dotted not in table:
        raise KeyError(f"module '{dotted}' not found. Available (sample): "
                       f"{list(table)[:20]} ...")
    return table[dotted]


def _capture_shapes(model, X_sample, device):
    """Run one forward and record the output shape of every named module."""
    shapes, handles = {}, []

    def mk(name):
        def h(mod, inp, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(o):
                shapes[name] = tuple(o.shape)
        return h

    for name, mod in model.named_modules():
        if name == "":
            continue
        handles.append(mod.register_forward_hook(mk(name)))
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(np.asarray(X_sample)).float().to(device)
        model(xb)
    for h in handles:
        h.remove()
    return shapes


def discover_branches(model, X_sample, device, n_branches=4, seq_len=None):
    """Best-effort auto-detection of the branch structure.

    Strategy:
      1. find an nn.ModuleList of length n_branches -> those are the branch
         encoders (fusion_modules).
      2. else find n_branches sibling submodules whose names share a prefix and
         contain a branch-ish keyword.
    For temporal_modules we pick, per branch, the *deepest* submodule under that
    branch whose output still has an axis equal to seq_len (the time axis); if the
    branch output itself is temporal we use the branch module directly.
    Returns a config dict (and never raises -- if nothing is found, returns a
    config with empty lists so the caller can print the tree and ask the user).
    """
    shapes = _capture_shapes(model, X_sample, device)
    if seq_len is None:
        # infer T as the largest axis that shows up a lot among 3D module outputs
        cand = [s[-1] for s in shapes.values() if len(s) >= 3]
        seq_len = max(set(cand), key=cand.count) if cand else X_sample.shape[1]

    fusion = []
    # (1) ModuleList of the right length
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) == n_branches:
            fusion = [f"{name}.{i}" for i in range(n_branches)]
            break
    # (2) sibling keyword modules
    if not fusion:
        kw = ("branch", "enc", "tcn", "stream", "tower")
        children = [n for n, _ in model.named_children()]
        hits = [n for n in children if any(k in n.lower() for k in kw)]
        if len(hits) >= n_branches:
            fusion = sorted(hits)[:n_branches]

    # temporal module per branch: prefer the branch's OWN final feature when it
    # already carries the time axis (the canonical pre-GAP last-layer activation,
    # which is also the fusion point). Only descend if the fusion output is NOT
    # temporal (e.g. a post-GAP vector). This avoids accidentally grabbing an
    # early block's projection just because its module name is longer.
    temporal = []
    for fm in fusion:
        shp = shapes.get(fm)
        if shp is not None and len(shp) >= 3 and seq_len in shp[1:]:
            temporal.append(fm)
            continue
        best, prefix = None, fm + "."
        for name, s in shapes.items():
            if name.startswith(prefix) and len(s) >= 3 and seq_len in s[1:]:
                if best is None or len(name) > len(best):
                    best = name
        temporal.append(best or fm)

    # head: last nn.Linear in the model (heuristic, optional)
    head = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            head = name
    return {
        "n_branches": n_branches,
        "branch_names": BRANCH_NAMES[:n_branches],
        "seq_len": int(seq_len),
        "fusion_modules": fusion,
        "temporal_modules": temporal,
        "head_module": head,
        "module_shapes": {k: list(v) for k, v in shapes.items()},
    }


# ---------------------------------------------------------------------------
# the adapter
# ---------------------------------------------------------------------------
class BranchAdapter:
    """Wraps an ours model to allow (i) coalition forwards with feature-level
    branch ablation and (ii) per-branch HiResCAM time series extraction."""

    def __init__(self, model, config, device):
        self.model = model.eval()
        self.device = device
        self.cfg = config
        self.n = config["n_branches"]
        self.fusion_names = config["fusion_modules"]
        self.temporal_names = config["temporal_modules"]
        if len(self.fusion_names) != self.n:
            raise ValueError("config['fusion_modules'] must list n_branches "
                             f"modules; got {self.fusion_names}")
        self.fusion_mods = [get_module(model, n) for n in self.fusion_names]
        self.temporal_mods = [get_module(model, n) for n in self.temporal_names]

        self._active = False          # ablation only fires when True
        self._ablate = {}             # idx -> bool
        self._mode = "mean"           # 'mean' | 'zero'
        self._baseline = {}           # idx -> tensor [C(,T)]
        self._last = {}               # idx -> detached last output (for baseline)
        self._handles = []
        for i, mod in enumerate(self.fusion_mods):
            self._handles.append(mod.register_forward_hook(self._mk_ablation(i)))

    # -- ablation hook on each fusion module --------------------------------
    def _mk_ablation(self, idx):
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            self._last[idx] = o.detach()
            if self._active and self._ablate.get(idx, False):
                if self._mode == "zero":
                    repl = torch.zeros_like(o)
                else:
                    b = self._baseline[idx].to(o.dtype).to(o.device)
                    repl = b.unsqueeze(0).expand(o.shape[0], *b.shape).contiguous()
                if isinstance(out, (tuple, list)):
                    return (repl,) + tuple(out[1:])
                return repl
            return None
        return hook

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    # -- baseline (mean branch feature over a pool) -------------------------
    def fit_baseline(self, X, bs=256):
        if len(X) == 0:
            raise ValueError("cannot fit a branch baseline on an empty training pool")
        self._active = False
        acc = {i: None for i in range(self.n)}
        cnt = 0
        with torch.no_grad():
            for s in range(0, len(X), bs):
                xb = torch.from_numpy(np.asarray(X[s:s + bs])).float().to(self.device)
                self.model(xb)
                for i in range(self.n):
                    o = self._last[i]                # [B, C(,T)]
                    summed = o.sum(dim=0)            # [C(,T)]
                    acc[i] = summed if acc[i] is None else acc[i] + summed
                cnt += xb.shape[0]
        self._baseline = {i: (acc[i] / cnt) for i in range(self.n)}
        return self._baseline

    # -- coalition value: target logit (or prob) with a subset present ------
    def value(self, X, target, present_set, use_prob=False, bs=256):
        self._active = True
        self._ablate = {i: (i not in present_set) for i in range(self.n)}
        out = []
        with torch.no_grad():
            for s in range(0, len(X), bs):
                xb = torch.from_numpy(np.asarray(X[s:s + bs])).float().to(self.device)
                logit = self.model(xb)
                v = torch.softmax(logit, -1)[:, target] if use_prob else logit[:, target]
                out.append(v.detach().cpu().numpy())
        self._active = False
        return np.concatenate(out)

    # -- per-branch HiResCAM time series s_b(t) (signed), coalition-aware ----
    def branch_temporal(self, X, target, present_set=None, bs=128):
        """Return dict idx -> array [N, T] of signed per-branch HiResCAM.
        If present_set is given, branches outside it are ablated during the pass
        (so s_j for kept branches reflects the intervention)."""
        if present_set is None:
            present_set = set(range(self.n))
        self._active = True
        self._ablate = {i: (i not in present_set) for i in range(self.n)}

        cap = {}                                   # idx -> live activation tensor
        handles = []

        def mk(idx):
            def h(mod, inp, out):
                o = out[0] if isinstance(out, (tuple, list)) else out
                cap[idx] = o
            return h
        for i in [j for j in range(self.n) if j in present_set]:
            handles.append(self.temporal_mods[i].register_forward_hook(mk(i)))

        present_order = sorted(present_set)
        series = {i: [] for i in present_order}
        for s in range(0, len(X), bs):
            xb = torch.from_numpy(np.asarray(X[s:s + bs])).float().to(self.device)
            xb.requires_grad_(False)
            cap.clear()
            logit = self.model(xb)
            score = logit[:, target].sum()
            active_order = [i for i in present_order if i in cap]
            acts = [cap[i] for i in active_order]
            grads = torch.autograd.grad(score, acts, retain_graph=False,
                                        allow_unused=True)
            for k, i in enumerate(active_order):
                a = acts[k]
                g = grads[k]
                if g is None:
                    T = self.cfg["seq_len"]
                    series[i].append(np.zeros((xb.shape[0], T)))
                    continue
                gi = g * a                          # [B, C, T] or [B, T, C]
                # reduce over the channel axis (the non-batch axis != time)
                t_axis = _time_axis(gi.shape, self.cfg["seq_len"])
                ch = [d for d in range(1, gi.dim()) if d != t_axis]
                st = gi.sum(dim=tuple(ch)) if ch else gi   # [B, T]
                series[i].append(st.detach().cpu().numpy())
        for h in handles:
            h.remove()
        self._active = False
        return {i: np.concatenate(series[i], axis=0) for i in present_order}


def _time_axis(shape, seq_len):
    dims = list(range(1, len(shape)))
    cand = [d for d in dims if shape[d] == seq_len]
    return cand[0] if cand else dims[-1]


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_ckpt(root, model, task, seed, k):
    import glob
    hits = sorted(glob.glob(f"{root}/{model}_{task}_seed{seed}_k{k}_*/model.pt"))
    if not hits:
        return None
    if len(hits) > 1:
        raise RuntimeError(
            f"ambiguous checkpoint for {model}/{task}/seed{seed}/k{k}: "
            f"{hits}. Remove stale runs or pass an explicit checkpoint."
        )
    return hits[0]
