> **역사 기록:** 아래 내용은 당시 버전의 작업·실험 상태다. 현재 실행법은 [REPRODUCE.md](REPRODUCE.md), 현재 결과 출처는 [PAPER_MAP.md](PAPER_MAP.md), 남은 문제는 [KNOWN_ISSUES.md](KNOWN_ISSUES.md)를 따른다. 경로와 명령을 현재 실행 지시로 사용하지 않는다.

# Excluded legacy paths

정리된 논문 재현 패키지는 아래 코드를 의도적으로 포함하지 않는다. 삭제가 아니라 원 압축본에 보존된 비정규 경로의 격리다.

| 제외 범주 | 이유 | 대체 경로 |
|---|---|---|
| `.ipynb_checkpoints`, `__pycache__`, `*.bak*`, `*.pyc` | 실행 이력·중복본이며 source revision을 모호하게 함 | 정리본의 단일 active source |
| `lob_xai/analysis.py`, TCAV/concept/regime 모듈 | 현 원고의 핵심 gate·branch 결과 계약과 직접 연결되지 않음 | 원고에 다시 포함할 때 별도 protocol·test 추가 |
| `lob_xai/viz_cam_timeline.py`, `viz_xai_method_agreement.py` | 부분 checkpoint loading과 오래된 sample 선택 규칙 | `paper/paper_figures*.py`, canonical CAM/selection 모듈 |
| 대부분의 `lob_xai_viz/fig*.py` | 중복 loader, `strict=False`, hard-coded/legacy 결과 경로 | `lob_xai_viz/viz_micro_loader.py`와 `paper/` 렌더러 |
| rebuttal·hypothesis·normalization-ablation 스크립트 | 별도 연구 질문·별도 protocol이며 현재 원고 표를 생성하지 않음 | 필요 시 독립 부록 패키지로 검증 |
| scattered timestep / temporal-mean deletion | 원고의 contiguous neighbor-conditional intervention과 다름 | `lob_xai.counterfactual_test` |
| input-level branch replacement / test-set branch mean | 원고의 내부 branch GAP training-mean 절제와 다름 | `scripts/common/axis_a_min.py`, `axis_b_branch_structure.py` |
| 기존 `viz_fig/*.png` | 수정 전 코드·입력 provenance를 확인할 수 없음 | 수정 코드와 실제 checkpoint/data로 재생성 |

제외된 코드를 논문 결과 생성에 다시 사용하려면 canonical data identity, strict checkpoint loading, top-confidence correct cohort, training-only donor/baseline, 창 단위 산출물과 run manifest를 먼저 연결해야 한다.

## 통합 정리(2026-07-27)에서 추가로 제외·정리한 항목

| 항목 | 처리 | 이유 |
|---|---|---|
| `cf_response_carpet_package.zip` (저장소 안에 포함돼 있던 배포 zip) | 삭제 | 내용물은 이미 `scripts/common/run_cf_response_carpet.py`와 `scripts/run_response_carpet.sh`로 편입됨 |
| `scripts/common/make_saliency_shape.py.pre_tensor_fix` | 삭제 | 수정 전 백업본. active source를 모호하게 함 |
| `README_FIRST_KO.md` | 삭제 | positional 패키지 우선순위 안내문. 내용은 `README.md`와 `docs/POOLING_EXPERIMENT.md`로 흡수 |
| 루트 `run_experiment.sh` (EXPERIMENT_POOL=positional 고정 wrapper) | 삭제 | `EXPERIMENT_POOL=... bash scripts/run_pooling_experiment.sh`로 직접 실행 |
| 루트 `README.md` | 재작성 | carpet 패키지 unzip 시 원 README가 덮어써져 있었음 |
| `run_response_carpet.sh`, `run_positional_visualization.sh` | `scripts/`로 이동 | 루트에는 문서와 설정만 유지 |
| 1세대 v3.x 저장소 (`lob_xai` zip: mid-price 방향 예측, TCAV/concept/regime, fig00–17/figa–j, rebuttal 스크립트) | 이 저장소에 미포함 (별도 보존) | 현 원고의 과제·결과 계약과 무관. 재사용하려면 canonical loader·strict loading·protocol 연결 필요 |
| `lob_xai_viz` 작업본 스냅샷 (`lob_xai_viz_20260615_0816`: .ipynb_checkpoints, __pycache__ 포함) | 이 저장소에 미포함 (별도 보존) | v3.x figure 체계의 작업 스냅샷 |

sibling 실험 저장소 `lob_xai_multiscale_pooling`은 코드가 이 저장소와 동일 계열이며(모든 pool 변형이 여기 포함됨), 그쪽에는 multi_scale run 산출물(loggers/results, figures)만 남는다. 신규 실험은 이 통합 저장소에서 실행하고, 기존 multi_scale checkpoint는 `frozen/task_specific_selected_6ckpt_manifest.json`의 SHA-256으로 참조·검증한다.
