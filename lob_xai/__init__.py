"""LOB-XAI package.

Public symbols are imported lazily.  This keeps lightweight utilities such as
label construction and manuscript auditing usable without importing PyTorch,
and avoids expensive import-time side effects in command-line tools.
"""
from __future__ import annotations

from importlib import import_module


_MODULE_EXPORTS = {
    "config": ["Config", "FIGURES_DIR"],
    "paper_protocol": [
        "DATA", "TRAINING", "COUNTERFACTUAL", "ours_parameter_count",
        "branch_variant_parameter_count",
        "deeplob_parameter_count", "tlob_parameter_count",
        "bin_ctabl_parameter_count", "baseline_parameter_count",
    ],
    "branching": [
        "make_branching_scheme", "list_schemes", "describe_scheme", "scheme_metadata",
    ],
    "data": [
        "load_fi2010_csv", "FI2010WindowDataset", "build_dataloaders",
        "stratified_subsample", "FI2010MicrostructureDataset",
        "build_microstructure_dataloaders", "load_fi2010_features_only",
    ],
    "models": [
        "LOBBranchModel", "LOBBranchCNN", "CNNBranch", "TCNBranch", "TCNBlock",
        "AttentionPool1D", "AvgPool1D", "MaxPool1D", "AvgLastPool1D",
        "MultiScaleTailPool1D", "LearnedPositionalPool1D", "BranchLocalPool1D",
        "ChannelLayerNorm1D", "make_branch", "make_pool",
    ],
    "gradcam": [
        "TimeSeriesGradCAM", "TimeSeriesHiResCAM", "TimeSeriesTSR",
        "compute_cam_all_branches", "DEFAULT_METHOD", "SUPPORTED_METHODS",
    ],
    "training": [
        "train", "evaluate", "compute_class_weights", "FocalLoss",
        "compute_alpha_inverse_freq", "find_threshold_high_precision",
        "find_threshold_youden", "find_threshold_cost",
    ],
    "loggers": ["generate_id", "find_save_path", "logger", "read_log"],
    "reporting": [
        "generate_full_report", "save_history_csv", "save_predictions_csv",
        "save_confusion_matrix", "save_learning_curves", "cam_diagnostic",
        "save_summary_json",
    ],
    "stats": [
        "permutation_test", "cliffs_delta", "cohens_d", "bh_fdr",
        "bootstrap_ci", "auc_roc", "wilcoxon_signed_rank",
    ],
    "microstructure_labels": [
        "compute_toxic_labels", "compute_vacuum_labels",
        "compute_microstructure_labels", "mask_valid_labels", "audit_tick_size",
        "TICK_SIZE_DEFAULT", "K_DEFAULT",
    ],
}

_LAZY = {
    symbol: (f"{__name__}.{module}", symbol)
    for module, symbols in _MODULE_EXPORTS.items()
    for symbol in symbols
}
__all__ = sorted(_LAZY)


def __getattr__(name: str):
    try:
        module_name, symbol = _LAZY[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), symbol)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
