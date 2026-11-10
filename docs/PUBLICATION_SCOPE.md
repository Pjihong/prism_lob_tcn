# 공개 범위와 제거된 기능

2026-11-10 사용자 지정에 따라 로그 산출물과 두 공통 모듈 및 직접 의존 기능을 제거했다.

## 이번에 제거한 항목

- 추적된 `.log` 파일 22개와 결과 압축파일 두 개 안의 `.log` 항목 12건.
- `lob_xai/loggers.py`: YAML 실행 기록 생성·읽기 기능과 패키지 export.
- `lob_xai/checkpoints.py`: 공통 checkpoint 모델·데이터 로더.
- 이 로더에 의존한 `scripts/evaluate_ours_checkpoint.py`, `scripts/verify_positional_checkpoint.py`, `scripts/probe_branch_structure.py`, `scripts/analyze_branch_importance.py`, `scripts/cam_sanity_checks.py`, `scripts/inspect_pool_weights.py`.
- 해당 로더 전용 테스트와 `docs/validation/regression_probe.py`.
- `plot_branch_importance.py`의 checkpoint 기반 CAM 캐시 계산. 기존 CSV·NPZ 캐시 렌더링은 유지한다.

학습 본체는 유지하며 공통 YAML logger 호출을 제거했다. Ours 학습 출력은 `runs/NAME/` 또는 `--output-dir`에 저장한다. 기존 출력 폴더를 재사용하지 않는다. 같은 이름을 다시 쓰려면 새 출력 경로를 지정한다. 화면 진행 출력과 학습 결과의 메모리상 계산은 남아 있다.

이는 모든 모델 로딩·로깅 코드를 없앴다는 뜻은 아니다. 별도 구현을 사용하는 CF·카드·factorization·gallery·positional-readout 도구와 baseline 학습은 남아 있다. 미래에 로컬로 생성되는 모델·예측 산출물도 공개 파일과 구분한다.

## 아직 검토 중인 자료

기존 `figures/` PNG 6개와 압축 안의 비로그 파일은 이번 삭제 대상이 아니다. 압축 안의 CSV·JSON·NPZ·과거 그림 및 일부 추출 가중치는 그대로 남아 있으며, 공개 제외 여부는 별도 검토 중이다. 상세 reference JSON과 개인 경로도 아직 일괄 정제하지 않았다.

압축파일을 다시 쓸 때 남겨 둔 모든 멤버의 경로·종류·크기·내용 해시가 동일한지 확인했다. 압축파일 자체의 해시는 바뀐다. 보존 원장은 원래 source 해시를 유지하고 현재 파일의 해시를 별도로 기록한다.

Git의 현재 파일 삭제는 과거 커밋의 삭제가 아니다. 이번 변경은 이력을 재작성하지 않는다. 과거 검증 로그가 사라져도 그 검증이 새로 실행되거나 과거 과학적 문제가 해결된 것으로 해석하지 않는다.

현재 사용법은 [실행 안내](REPRODUCE.md), 수치·출처의 한계는 [알려진 이슈](KNOWN_ISSUES.md)를 따른다.
