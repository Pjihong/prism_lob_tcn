"""Create experiment identifiers and persist YAML run metadata.

Inputs: run names, configuration and an optional output root.
Outputs: run directories and YAML logs.
Scope: this module records runs; it does not select manuscript checkpoints.
"""
from __future__ import annotations
import os
import sys
import yaml
from datetime import datetime


def _root_path() -> str:
    """프로젝트 루트 — 환경변수 LOB_XAI_ROOT 우선, 없으면 sys.path[0]."""
    return os.environ.get("LOB_XAI_ROOT", sys.path[0])


def generate_id(name: str) -> str:
    """
    실험 ID 생성 + 결과 디렉토리 생성.

    Returns
    -------
    model_id : str  e.g. 'lobcnn_2026-05-07_06:30:00'
    """
    init_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_id = f"{name}_{init_time}"
    dir_path = os.path.join(_root_path(), "loggers", "results", model_id)
    os.makedirs(dir_path, exist_ok=True)
    return model_id


def find_save_path(model_id: str) -> str:
    """model_id에 해당하는 결과 디렉토리 경로 반환."""
    return os.path.join(_root_path(), "loggers", "results", model_id)


def _to_yaml_safe(obj):
    """Recursively convert tuples → lists, numpy types → Python scalars.

    yaml.safe_load can't read python/tuple tags, and numpy scalars dump as
    binary blobs.  This makes the YAML round-trippable.
    """
    import numpy as _np
    if isinstance(obj, tuple):
        return [_to_yaml_safe(x) for x in obj]
    if isinstance(obj, list):
        return [_to_yaml_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    return obj


def logger(model_id: str, header: str, contents) -> None:
    """
    log.yaml에 {header: contents} append (기존 키 있으면 update).
    """
    file_path = os.path.join(_root_path(), "loggers", "results", model_id, "log.yaml")
    payload = {header: _to_yaml_safe(contents)}
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            current = yaml.safe_load(f) or {}
            current.update(payload)
    else:
        current = payload
    with open(file_path, "w") as f:
        yaml.dump(current, f, default_flow_style=False, allow_unicode=True)


def read_log(model_id: str, header: str):
    """log.yaml에서 header 키 값 읽기."""
    file_path = os.path.join(_root_path(), "loggers", "results", model_id, "log.yaml")
    with open(file_path, "r") as f:
        log = yaml.safe_load(f)
    return log[header]
