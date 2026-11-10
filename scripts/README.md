# 실행 코드 안내

| 단계 | 명령·파일 | 입력과 출력 |
|---|---|---|
| 데이터 확인 | `audit_label_prevalence.py`, `audit_tick_size.py` | 로컬 데이터의 라벨·범위 점검 |
| 모델 학습 | `train_microstructure.py`, `train_baseline_microstructure.py` | 데이터·설정 → 새 로컬 모델·예측·요약 |
| 결과 집계 | `aggregate_final_test.py`, `collect_branch_dev_results.py` | 사용자가 준비한 평가·validation 결과 |
| 시간 구간 개입 | `run_cf_response_carpet.py`, `cf_effect_stats.py` | 데이터·호환 모델 → 개입 결과·통계 |
| 구조 검사 | `smoke_pooling.py --pool positional`, `verify_baseline_models.py` | 합성 입력의 모델 동작 검사 |
| 자료 식별·문서 | `make_run_manifest.py`, `check_docs.py` | 새 실행의 출처 기록·현재 문서 링크 검사 |

각 Python 명령은 `--help`로 인자를 확인할 수 있습니다. `run_response_carpet.sh`는 사용자가 지정한 모델 경로로 여러 CF 실행을 연결합니다. 과거 결과 파일이나 원고 감사 기록은 이 폴더에 포함하지 않습니다.

[설치·실행](../docs/REPRODUCE.md) · [그림 도구](../lob_xai_viz/README.md) · [공개 범위](../docs/PUBLICATION_SCOPE.md)
