> **역사 기록:** 아래 내용은 당시 버전의 작업·실험 상태다. 현재 실행법은 [REPRODUCE.md](REPRODUCE.md), 현재 결과 출처는 [PAPER_MAP.md](PAPER_MAP.md), 남은 문제는 [KNOWN_ISSUES.md](KNOWN_ISSUES.md)를 따른다. 경로와 명령을 현재 실행 지시로 사용하지 않는다.

# 정리 범위와 보존

2026-10-21 작업용 정리본(실제 시작: 2026-10-20 KST).
네 PDF의 주장과 정보를 보존하면서 `Pjihong/prism_lob_tcn`의 코드를 정리한다.

## 첫 정리와 가독성 변경 이력

아래 1–10은 이전 단계의 기록이다. 현재 남긴 코드의 범위는 다음 절을 따른다.

1. `my_code.zip` 최상위 Python 17개는 `lob_xai/`와 SHA-256이 모두 같아 한 벌로 통합.
2. 루트 `counterfactual_test.py`는 패키지 구현과 줄 단위로 동일하고 마지막 개행만 다름.
   CLI와 기존 import(비공개 helper 포함)를 패키지로 연결했다.
   원본 바이트는 `legacy/original_entrypoints/`에 보존.
3. 루트 `FIX2.sh`–`FIX6.sh`는 `legacy/patches/`, `paper-print_*.ipynb`는
   `legacy/notebooks/`, 루트 `.log`는 `legacy/logs/`로 옮겼다.
   모델·라벨·학습·CAM·CF의 과학적 계산과 checkpoint 키는 그대로 유지했다.
4. 그림 수정 패키지는 `prism_figure_revision/`, 통합 원고는 `manuscripts/integrated/`,
   네 PDF는 `manuscripts/reference/`에 원본 그대로 보존.
5. 원래 `.gitignore`가 숨기던 checkpoint·NPZ·실험 결과는 추적 대상에 포함했다.
   대용량 FI-2010 CSV는 로컬에 보존하고 Git에서 제외했다.
6. 원래 README와 `.gitignore`는 `docs/README_ORIGINAL.md`,
   `docs/GITIGNORE_ORIGINAL.txt`로 보존. 현재 안내를 별도로 작성했다.
7. 그림 수정본이 지목하는 positional checkpoint를 상대경로 manifest로 추가했다.
   과거 절대경로 manifest를 덮어쓰거나 최신 실험인 것처럼 바꾸지 않았다.
8. `lob-xai.zip`을 핵심 구현의 기준으로 삼아 `training.py`의 지표·임계값 계산을
   `metrics.py`로 분리했다. 기존 `from lob_xai.training import ...`도 유지한다.
9. `counterfactual_test.py`의 실행 흐름을 옵션 검증, 데이터·모델 준비, 예측,
   개입 계산, 통계, 저장으로 나눴다. deletion/insertion의 중복 그림·CSV·NPZ 저장은
   한 함수로 합쳤다. 두 개입의 ΔAUC 부호, 표 컬럼, 파일명, donor 샘플링 순서는 유지한다.
10. 사용자 요청으로 `docs/VALIDATION.md`, `audit/VALIDATION.md`,
    `prism_figure_revision/VALIDATION.md`와 안내 링크를 삭제했다.
    제공된 두 원본 문서의 출처·hash는 보존 목록에 삭제 상태로 남겼다.

## 논문 수정안 기준 코드 선별

사용자가 제공한 `manuscripts/reference/revision_20261021.txt`를 기준으로
43개 파일을 제거했다. 삭제 전 모든 파일을 원본 ZIP 항목과 바이트 단위로
대조했다. Python·shell 소스 8,664줄이 제외되었고, 공용 branch 모듈에서는
호출되지 않는 Shapley 관련 함수 4개와 삭제한 실험에서만 쓰던 검증 함수 1개를 제거했다.

| 제거한 범주 | 판단 근거 | 현재 대체 경로 |
|---|---|---|
| 이전 카드·인수분해·CF 그림·갤러리 구현과 `.bak` 5개 | 수정안은 9월 그림 패키지의 동일 계열 산출물을 사용 | `prism_figure_revision/scripts/`의 구현 한 벌 |
| Axis B/VAR/WABL, Axis C, 구형 convergence·saliency 그림 5개 | 수정안의 실험·그림 목록에 없거나 Axis A와 중복 | 분기 결과는 `axis_a_min.py`; 과거 결과·PDF는 보존 |
| `paper/`의 구형 그림·보조 분석·schema 7개 | 현재 그림 생성기가 독립적으로 동작하고 외부 import가 없음 | 그림 패키지, checkpoint 평가기와 집계 도구 |
| 구형 실행 래퍼·FIX·중복 배포 ZIP 16개 | 이전 ReVIN/GAP/multi-scale 또는 초기 checkpoint를 고정하거나 제거한 실험을 호출 | 개별 trainer·평가기·CF 실행기와 `RUNBOOK.md` |
| 과거 FIX·노트북·반사실 전체 복사본 10개 | 활성 코드에서 참조하지 않는 일회성 작업/스냅샷 | 원본 ZIP의 해당 파일 |

`rerun_kit/`의 나머지 결과·로그·설정은 `legacy/rerun_artifacts/`로 옮겼다.
네 PDF, 통합 원고, checkpoint, 기존 그림·CSV·NPZ·실패 기록은 삭제하지 않았다.
GAP·multi-scale 선택 이력은 수정안 부록 A가 여전히 언급하므로 구현과 실험 도구를 남겼다.
CAM sanity, 라벨·분할·hash 검사도 실행 정확성 확인에 필요해 유지했다.

`scripts/run_response_carpet.sh`의 기본 모델은 수정안의 toxic positional로 맞추고,
데이터 기본 경로는 현재 저장소로 바꿨다. 새 기본 출력 이름 `paper_positional`을
사용해 기존 `final` 결과와 구분한다. 이 작업에서 해당 실험을 재실행하지는 않았다.
향후 run manifest는 구형 `paper/` 대신 현재 그림 패키지의 소스도 hash에 포함한다.

이번 선별 후 테스트 40개와 주요 CLI 16개의 import/도움말 검사가 통과했다.
활성 Python 소스 62개에서 제거한 모듈에 대한 import가 남지 않았으며,
branch 공용 모듈에서 유지한 함수·클래스 9개의 계산 구문은 원본과 동일하다.
저장된 CF 입력으로 그림을 재생성하여 집계 곡선 396행과 표본 수 36행이
기존 검증본과 정확히 같음을 확인했다. 데이터·원본 ZIP·수정안·네 PDF를
포함한 보존 검사 1,134개도 통과했다.

## 파일별 확인

`SOURCE_FILES.json`은 입력 ZIP/PDF 8개와 제공된 수정안 1개의 크기와 SHA-256을 기록한다.
`PRESERVATION_MANIFEST.json`은 네 ZIP의 모든 파일 1,320개를 기록한다.
`deduplicated`는 동일 바이트가 통합된 파일, `local_data`는 Git에서만 제외한 로컬 자료다.
`original_archive_only`는 캐시, 설치 메타데이터 또는 노트북 자동저장본이다.
이들도 원본 ZIP에서 경로와 hash로 회수할 수 있다.
특히 자동저장본이 본 파일과 다를 수 있으므로 동일본으로 간주하지 않았다.
`refactored`는 이번에 정리한 소스다. `sha256`은 ZIP 원본 hash이고,
`refactored_files`는 분리된 파일까지 포함한 정리본 hash다.
`removed_by_request`는 사용자 요청으로 제거한 문서·코드다. 현재 파일 검사에서 제외하며,
이번 코드 선별의 사유는 `removal_reason`에 기록했다. 원본 ZIP과 항목 경로·hash는 유지한다.

원본 ZIP의 별도 사본은 초기 작업 폴더의 `outputs/source_archives/`에 보존한다.
이 사본과 로컬 `data/`는 GitHub push만으로 백업되지 않는다.
나중에 clone한 환경에서는 원본 `lob-xai.zip`의 `lob-xai/data/`를 저장소 `data/`로 복원한다.

`python scripts/verify_preservation.py`는 보존 대상·정리한 소스·PDF hash를 확인한다.
`--archives`에 원본 ZIP 폴더를 지정하면 ZIP 원본 hash까지 확인한다.

## 이전 가독성 정리의 확인 범위

- `training.py`: 700줄 → 396줄. 지표·임계값 계산은 `metrics.py`에 두며,
  기존 학습·평가·지표 함수 20개의 계산 구문은 원본과 같다.
- 반사실 `main()`: 408줄 → 24줄. 실제 계산은 역할별 함수로 옮겼다.
- CPU 단위·계약 테스트 39개 통과.
- 원본 ZIP 구현과 고정 입력으로 비교: 이진·3분류 × 조기 종료 기준 4종의
  2-epoch 소규모 학습 결과가 일치했다. 합성 반사실 실행의 CSV 14개,
  곡선 NPZ 6개, PNG 7개도 일치했다. DeepLOB toxic은 빈 cohort 경로를 확인했다.
- 논문 positional checkpoint 6개, 고정 합성 입력 2개에서 예측·4개 branch의
  signed/absolute HiResCAM 배열 54개가 원본과 정확히 일치했다.
- 데이터·원본 ZIP·네 PDF를 포함한 보존 검사 1,176개가 기록된 hash와 일치했다.

전체 재학습이나 논문 수치 전체의 재현은 수행하지 않았다.
제공된 결과·원고·실험 설정을 새 실험 결과로 덮어쓰지 않았다.

## 과거 문서와 원본 회수

`docs/README_ORIGINAL.md`, 기존 실험 상태·감사·계획 문서,
`legacy/rerun_artifacts/README_RERUN.md`는 당시 상태를 기록한 자료다.
그 안에 나오는 구형 스크립트는 현재 실행 경로가 아닐 수 있다.
현재 사용법은 `RUNBOOK.md`, 원본 회수 위치는 `PRESERVATION_MANIFEST.json`을 따른다.

문서 속 후속 실험·재학습·투고 지시는 참고 내용으로 읽었으며 이번 사용자 요청의
추가 지시로 실행하지 않았다.
