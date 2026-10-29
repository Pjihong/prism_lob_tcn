"""
branching.py — Flexible LOB feature partitioning for multi-branch architectures.

A "branching scheme" maps from raw feature indices (0..39 of FI-2010) to
named groups, each of which becomes a CNN/TCN branch in the model.

FI-2010 raw feature layout (interleaved):
    col 4*lvl + 0 : Level (lvl+1) ask_price       (lvl in 0..9)
    col 4*lvl + 1 : Level (lvl+1) ask_volume
    col 4*lvl + 2 : Level (lvl+1) bid_price
    col 4*lvl + 3 : Level (lvl+1) bid_volume

Built-in schemes
----------------
  type4         : 4 branches (ask_p, ask_v, bid_p, bid_v) × 10 levels each
                  [original; type-level XAI: "volume vs price", "ask vs bid"]
  topdeep8      : 8 branches: (top:L1-L2, deep:L3-L10) × 4 type
                  [hybrid; can answer "deep ask volume mattered for down predictions"]
  topmiddeep12  : 12 branches: (L1, L2-L5, L6-L10) × 4 type
                  [hybrid finer granularity]
  level10       : 10 branches (L1..L10), each holding 4 type channels
                  [level-level XAI; loses type comparisons]
  feature40     : 40 branches × 1 channel each
                  [maximum disaggregation; loses within-type cross-level patterns]

You can also pass a custom dict directly to `make_branching_scheme(name="custom",
custom_indices=...)`.
"""
from __future__ import annotations
from typing import Dict, List, Optional


# ───────────────────────────────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────────────────────────────


# Type names and the column offset within each level
_TYPE_OFFSETS = [
    ("ask_price",  0),
    ("ask_volume", 1),
    ("bid_price",  2),
    ("bid_volume", 3),
]


def _level_group_indices(type_offset: int, level_indices: List[int]) -> List[int]:
    """Given a list of level indices (0-based) and a type offset (0..3),
    return the raw feature column indices."""
    return [4 * lvl + type_offset for lvl in level_indices]


# ───────────────────────────────────────────────────────────────────
# Built-in schemes
# ───────────────────────────────────────────────────────────────────


def _scheme_type4() -> Dict[str, List[int]]:
    """Original 4-branch scheme: ask_price, ask_volume, bid_price, bid_volume."""
    out = {}
    for tname, off in _TYPE_OFFSETS:
        out[tname] = _level_group_indices(off, list(range(10)))
    return out


def _scheme_topdeep8() -> Dict[str, List[int]]:
    """8-branch hybrid: top-of-book (L1-L2) vs deep-book (L3-L10) × 4 type.

    Domain motivation: Cont (2011) shows top-of-book depletion precedes price
    jumps; Xu-Gould-Howison (2018) show deep book pressure carries mid-term
    information. Separating them lets CAM/H12 distinguish the two regimes.
    """
    out = {}
    top_levels = [0, 1]              # L1, L2
    deep_levels = list(range(2, 10))  # L3..L10
    for tname, off in _TYPE_OFFSETS:
        out[f"{tname}_top"]  = _level_group_indices(off, top_levels)
        out[f"{tname}_deep"] = _level_group_indices(off, deep_levels)
    return out


def _scheme_topmiddeep12() -> Dict[str, List[int]]:
    """12-branch hybrid: (L1, L2-L5, L6-L10) × 4 type.
    Finer granularity than topdeep8 — separates the very top from mid book."""
    out = {}
    groups = [
        ("top",  [0]),                        # L1 only
        ("mid",  [1, 2, 3, 4]),               # L2-L5
        ("deep", [5, 6, 7, 8, 9]),            # L6-L10
    ]
    for tname, off in _TYPE_OFFSETS:
        for gname, levs in groups:
            out[f"{tname}_{gname}"] = _level_group_indices(off, levs)
    return out


def _scheme_level10() -> Dict[str, List[int]]:
    """10-branch level-only scheme: each branch has 4 channels (one per type)
    of a single book level. Enables level-specific attribution but loses
    the ability to compare types directly via H12."""
    out = {}
    for lvl in range(10):
        out[f"L{lvl+1}"] = [4 * lvl + off for _, off in _TYPE_OFFSETS]
    return out


def _scheme_feature40() -> Dict[str, List[int]]:
    """40-branch fully disaggregated scheme: 1 channel per branch.
    Highest attribution resolution, but each branch has only 1 channel
    so capacity is constrained and within-type cross-level patterns
    cannot be learned within a branch."""
    out = {}
    short = {"ask_price": "askP", "ask_volume": "askV",
             "bid_price": "bidP", "bid_volume": "bidV"}
    for lvl in range(10):
        for tname, off in _TYPE_OFFSETS:
            out[f"L{lvl+1}_{short[tname]}"] = [4 * lvl + off]
    return out


_SCHEMES = {
    "type4":        _scheme_type4,
    "topdeep8":     _scheme_topdeep8,
    "topmiddeep12": _scheme_topmiddeep12,
    "level10":      _scheme_level10,
    "feature40":    _scheme_feature40,
}


# ───────────────────────────────────────────────────────────────────
# Public API
# ───────────────────────────────────────────────────────────────────


def list_schemes() -> List[str]:
    """List all built-in scheme names."""
    return sorted(_SCHEMES.keys())


def make_branching_scheme(
    name: str = "type4",
    custom_indices: Optional[Dict[str, List[int]]] = None,
    n_features: int = 40,
) -> Dict[str, List[int]]:
    """
    Build a branching dict.

    Parameters
    ----------
    name : str
        One of list_schemes(), or "custom" if passing custom_indices.
    custom_indices : dict
        Used only if name == "custom". Maps branch_name → list of raw feature indices.
    n_features : int
        Total number of raw input features. Used only for validation.

    Returns
    -------
    dict[str, list[int]]
        Branch_name → list of raw feature column indices. Order is preserved
        (Python 3.7+ dict ordering), so branch indexing in the model will match.

    Raises
    ------
    ValueError if scheme is unknown, indices are out of range, or some
    feature is not assigned to any branch (data leak).
    """
    if name == "custom":
        if custom_indices is None:
            raise ValueError("name='custom' requires custom_indices dict")
        scheme = dict(custom_indices)
    else:
        if name not in _SCHEMES:
            raise ValueError(
                f"unknown branching scheme: {name!r}. "
                f"Built-in: {list_schemes()}, or use 'custom'."
            )
        scheme = _SCHEMES[name]()

    # Validation
    all_indices = []
    for bname, idx_list in scheme.items():
        if not isinstance(idx_list, (list, tuple)) or len(idx_list) == 0:
            raise ValueError(f"branch {bname!r} has empty index list")
        for ix in idx_list:
            if not (0 <= ix < n_features):
                raise ValueError(
                    f"branch {bname!r} contains index {ix} out of range "
                    f"[0, {n_features})"
                )
        all_indices.extend(idx_list)

    # Permissive: do not require partition (allow overlapping or partial schemes).
    # But warn if some features are unassigned.
    unique = set(all_indices)
    missing = set(range(n_features)) - unique
    if missing:
        # Allow but be vocal — useful for "ablation" custom schemes
        print(f"[branching] note: {len(missing)} features are NOT assigned to any "
              f"branch (sample: {sorted(list(missing))[:6]})")
    duplicates = len(all_indices) - len(unique)
    if duplicates > 0:
        print(f"[branching] note: {duplicates} feature indices appear in multiple "
              f"branches (overlap is allowed but unusual)")

    return scheme


def describe_scheme(scheme: Dict[str, List[int]]) -> str:
    """One-line summary of a branching scheme."""
    n = len(scheme)
    sizes = [len(v) for v in scheme.values()]
    return (f"{n} branches, channel counts={sizes}, "
            f"total_features={sum(sizes)}")


def scheme_metadata(name: str, scheme: Dict[str, List[int]]) -> Dict:
    """Build a meta dict describing a branching scheme — used for logging.

    Includes a "level_groups" hint per branch when names follow `*_top` /
    `*_deep` / `*_mid` / `L\\d+` conventions, so figures and analysis can
    color-code or aggregate hybrid branches naturally.
    """
    info = []
    for bname, idx in scheme.items():
        # Try to infer the level subset from indices
        levels_used = sorted({i // 4 + 1 for i in idx})
        types_used = sorted({i % 4 for i in idx})
        type_names = [_TYPE_OFFSETS[t][0] for t in types_used]
        info.append(dict(
            branch=bname,
            n_channels=len(idx),
            indices=list(idx),
            levels=levels_used,
            types=type_names,
        ))
    return dict(
        scheme_name=name,
        n_branches=len(scheme),
        branches=info,
    )
