# 실행 코드 읽는 순서

공통 데이터·모델·분석 함수는 `lob_xai`, 비교 모델은 `models`, 렌더러는 `lob_xai_viz`에 있습니다.
이 폴더는 이 기능들을 호출해 실험을 실행합니다. 저장소 루트에서 설치한 뒤 실행하세요.

| 순서 | 실행 파일 | 사용하는 구현 / 만들어지는 자료 |
|---|---|---|
| 1. 데이터 확인 | `audit_label_prevalence.py`, `audit_tick_size.py` | `lob_xai.data`, 위험 라벨과 데이터 범위 점검 |
| 2. 학습 | `train_microstructure.py`, `train_baseline_microstructure.py` | 논문 `.args` → `lob_xai` / `models` → checkpoint |
| 3. 평가 | `evaluate_ours_checkpoint.py`, `aggregate_final_test.py` | `lob_xai.checkpoints` → 예측·지표 → seed별 집계 |
| 4. 분기 분석 | `probe_branch_structure.py` → `analyze_branch_importance.py` | `lob_xai.branch_analysis` → 검증된 분기 설정 → signed 대체 효과 |
| 5. 시간 구간 개입 | `run_cf_response_carpet.py`, `cf_effect_stats.py` | `lob_xai.counterfactual_test` → 창별 개입 결과 → 통계 |
| 6. 설명 점검 | `cam_sanity_checks.py`, `inspect_pool_weights.py` | CAM 안정성·readout 가중치 |
| 7. 그림 | `python -m lob_xai_viz.render` | [그림 코드 안내](../lob_xai_viz/README.md) |
| 실행 전후 확인 | `smoke_pooling.py`, `verify_baseline_models.py`, `verify_positional_checkpoint.py`, `audit_paper_contract.py` | 모델 구조·프로토콜 점검 |
| 자료 식별·문서 확인 | `make_run_manifest.py`, `verify_preservation.py`, `check_docs.py` | 현재 소스 해시, 보존 기록, 문서 링크 |

`run_pooling_experiment.sh`와 `run_response_carpet.sh`는 여러 실험을 순차 실행하는 Bash 도구입니다.
개별 Python 명령과 필요한 입력은 [REPRODUCE.md](../docs/REPRODUCE.md)에 있습니다.
고신뢰 CF와 신뢰도 층화 CF는 표본군이 다르므로 결과를 섞지 않습니다.
