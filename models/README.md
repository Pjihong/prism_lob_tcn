# Baseline sources

The three supplied baseline implementations are integrated here:

- `deeplob.py` exposes `build_deeplob`
- `tlob.py` exposes `build_tlob`
- `bin_ctabl.py` exposes `build_bin_ctabl`

All factories accept `n_classes`, return logits, and use the paper input
convention `(batch, time, 40 features)`. The trainer checks the forward output
and analytic parameter count before optimization and stores the source SHA-256
in each checkpoint.

These are self-contained source files supplied with the project; no upstream
repository commit/tag accompanied them. Record that revision before describing
them as byte-identical to an authors' reference implementation.
