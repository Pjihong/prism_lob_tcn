# 실행 코드 읽는 순서

공통 데이터·모델·분석 함수는 `lob_xai`, 비교 모델은 `models`, 렌더러는 `lob_xai_viz`에 있습니다.

| 단계 | 실행 파일 | 현재 지원 범위 |
|---|---|---|
| 데이터 확인 | `audit_label_prevalence.py`, `audit_tick_size.py` | 위험 라벨과 데이터 범위 점검 |
| 학습 | `train_microstructure.py`, `train_baseline_microstructure.py` | 학습 및 로컬 결과 저장. Ours는 YAML 실행 기록을 생성하지 않음 |
| 저장된 결과 집계 | `aggregate_final_test.py`, `collect_branch_dev_results.py` | 별도로 준비한 결과의 집계 |
| 시간 구간 개입 | `run_cf_response_carpet.py`, `cf_effect_stats.py` | 독립 CF 구현과 결과 통계 |
| 그림 | `python -m lob_xai_viz.render` | [그림 코드 안내](../lob_xai_viz/README.md) |
| 구조 점검 | `smoke_pooling.py`, `verify_baseline_models.py`, `audit_paper_contract.py` | 남아 있는 모델·프로토콜 점검 |
| 자료 식별·문서 확인 | `make_run_manifest.py`, `verify_preservation.py`, `check_docs.py` | 소스 해시, 보존 기록, 문서 링크 |

공통 checkpoint 로더와 의존 평가·분기 분석·검증 명령은 제거했습니다. 정확한 파일 목록과 독립 도구의 유지 범위는 [공개 범위](../docs/PUBLICATION_SCOPE.md)에 있습니다.

`run_pooling_experiment.sh`와 `run_response_carpet.sh`는 여러 실험을 순차 실행하는 Bash 도구이며 stdout을 `.log` 파일로 저장하는 단계는 제거했습니다. 개별 Python 명령과 입력은 [실행 안내](../docs/REPRODUCE.md)를 따릅니다.
