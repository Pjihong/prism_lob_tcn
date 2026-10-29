# 저장소 설명 개선: 진단·적용 내역·남은 작업

기준: `e19fa92`의 로컬 checkout과 첨부된 다섯 ZIP, 원고 `revision_20261021.txt`. 검토 범위는 설명, 문서, 이름, 실행 안내와 구조다. 연구 방법의 타당성 재심사나 정리의 추가 증명은 이번 작업 범위에 포함하지 않는다.

## 1. 한 줄 진단과 먼저 고칠 5가지

**독자가 연구와 재현 경로를 찾기 전에 과거 정리 과정과 실험 세대를 먼저 해독해야 하는 구조였다.**

| 순서 | 개선 | 이번 반영 |
|---|---|---|
| 1 | README를 연구 소개와 독자 질문 중심으로 재작성 | 영문·국문 소개, FI-2010 출처, 설명 점검 질문, 코드·실험 기록 경로와 데이터 없는 시작 추가 |
| 2 | 논문 표·그림 → 명령 → 결과 경로 연결 | [PAPER_MAP.md](PAPER_MAP.md)에 표6개·그림7개와 알고리즘 위치 정리 |
| 3 | GAP 기본값과 positional 논문 설정의 구분 | 기본값의 호환 목적 명시, 실행 가능한 `@configs/paper/*.args` 추가 |
| 4 | `final` 결과 이름의 혼동 해소 | 경로를 유지하고 task·모델·cohort별 결과 색인 추가 |
| 5 | 저장소 밖 ZIP 의존 안내와 깨진 참조 제거 | [DATA.md](DATA.md), [REPRODUCE.md](REPRODUCE.md), 현재 문서 링크 검사 추가 |

## 2. 상세 진단: 파일과 줄 위치

아래 줄 번호는 수정 전 커밋 **e19fa92** 기준이다. 현재 파일의 줄 번호는 docstring 교체로 달라질 수 있다.

| 위치 | 발견 내용 | 조치·확인 상태 |
|---|---|---|
| `README.md:5` | 작업 일자·“제공된 수정안”·외부 ZIP 회수 안내가 첫 화면을 차지 | 현재 연구 소개로 교체 |
| `README.md:38–68` | 데이터 없는 명령, 실제 학습, 미해결 수치가 한 흐름에 혼재 | 실행 단계와 known issues 분리 |
| `lob_xai/config.py:48` | paper model을 GAP라고 설명 | 호환 기본값으로 주석 수정 |
| `lob_xai/paper_protocol.py:1–6,18–20,42` | single source of truth 선언과 historical GAP·분할 가중치 혼동 | 역할을 공유 상수·호환 기본값으로 명시, 70:15는 development 내부 비율 |
| `lob_xai/models.py:9–18` | attention을 default라고 쓰고 다른 도메인 이력으로 추천 | 현재 모델·호환 기본값·지원 옵션 설명으로 교체 |
| `scripts/train_microstructure.py:73` | 실제 70/15/15를 검사한다는 help | development 분할과 별도 test 검사로 정정 |
| `lob_xai/counterfactual_test.py:1–42` | 발표·사용자 확정이라는 설명, 부정확한 출력 안내 | 실행·입력·출력·범위 중심 머리말 |
| `lob_xai/reporting.py:6–7` | 출력이 항상 3-class인 것처럼 설명 | 실제 `class_names` 기반 동적 출력 명시 |
| `prism_figure_revision/README_KO.md:3–23` | 통합된 도구에 추가 unzip을 지시 | 저장소 루트의 Python 명령으로 교체 |
| `docs/RUNBOOK.md:38–43` | clone 독자에게 외부 ZIP에서 데이터 복원을 요구 | 공식 출처·형식·해시·변환 이력 공백 안내 |
| 수정안 `:755,806` / `figures/final_test/vacuum_seed*.json` | 원고 .239와 저장 결과 평균 .209449 불일치 | 자동 수정하지 않고 출처 확정 과제로 유지 |
| 수정안 `:840` / `figures/factorization/*.provenance.json` | 수치 오차 값의 버전 차이 | 최종 그림 provenance 지정 필요 |
| `scripts/common/axis_a_min.py:80–95` | SciPy 없는 fallback이 동률에서 tau-b와 다름 | 논문 설명 변경과 분리한 후속 코드 수정 항목 |
| `docs/README_ORIGINAL.md` 상대 링크 | docs 폴더 이동 이전 경로 | 상대 링크 수정, 역사 기록 표시 |

모든 과거 감사 항목을 현재 결함이라고 보지 않는다. 코드·그림·원고의 숫자를 문서 개선 과정에서 조용히 바꾸지 않는다.

## 3. 고치는 순서와 단계별 완료 기준

| 단계 | 작업 | 완료 기준 | 상태 |
|---|---|---|---|
| 0 | 기준 원고와 모델 식별 | 수정안 경로와 paper checkpoint manifest를 문서에 고정 | 반영 |
| 1 | 연구 소개 | 제목·요약·데이터 출처·설명 점검·결과 범위·빠른 시작이 README에 있음 | 반영 |
| 2 | 표·그림 색인 | 모든 label에 명령/원본/데이터 필요 여부 또는 미지원 표시 | 반영 |
| 3 | 실행 설정 | 두 task의 positional preset을 파서 테스트로 확인 | 통과 |
| 4 | 설명과 이력 구분 | 과거 문서에 역사 표시, 현재 문서로 이어지는 링크 | 반영 |
| 5 | 경로 정리 | 기존 결과·checkpoint 경로를 유지하며 역할 색인 | 최소안 적용 |
| 6 | 코드 머리말 | 주요 모델·데이터·귀속·보고 모듈의 현재 역할 명시 | 반영 |
| 7 | 검증 | 단위 테스트, doc 링크, pooling/checkpoint·보존 검사, 저장 CF 재생성 | 42개 테스트·16개 문서 링크·보존1129개 통과, CF 재생성 완료 |
| 8 | 완전한 데이터 재현 | 외부 독자가 원 배포본에서 기대 CSV 해시 생성 | 미완료: 변환 이력 필요 |

검증 단계는 새 학습이나 전체 논문 재현과 구별한다. “처음 보는 독자가 이해하는가”라는 사용자 테스트도 자동 검사로 대체하지 않는다.

## 4. README 뼈대와 docstring 틀

```text
제목 / 논문 제목 / 저자 / 한 문단 소개와 약어
FI-2010 원 논문·배포처 / 실험 CSV의 출처 확인 상태
예측 과제 / 설명 점검: 무엇을 보는가
저장된 결과 예시 / 표본군과 결과의 범위
데이터 없는 빠른 시작 / 논문 학습 preset
저장소 구성과 코드 읽는 순서
실험 기록 경로 / 논문–코드 대응표
인용과 라이선스 상태
```

README는 영어, `README.ko.md`는 한국어로 제공한다. 핵심 모듈 머리말은 영어로 정리하고 상세 작업 가이드는 한국어로 유지한다. 모든 기존 주석을 기계 번역하거나 역사 문서의 인용을 금지어로 삭제하지 않는다.

Docstring 틀:

```python
"""One sentence describing the module's current responsibility.

Two or three sentences describing the implemented operation.
Run: a real CLI command, or omit for library-only modules.
Inputs: required data, tensors or artifacts.
Outputs: returned values or generated files.
Scope: conditions needed to interpret the output correctly.
"""
```

전후 예:

```python
# Before (config.py)
pool: str = TRAINING.pool  # paper model uses fixed global average pooling
# After
pool: str = TRAINING.pool  # legacy GAP default; paper preset selects positional
```

```python
# Before (module header)
"""논증 구조 (발표 Part 4와 동일), 핵심 설계 (사용자 확정) ..."""
# After
"""Audit temporal attribution with donor-based deletion and insertion.

Inputs: checkpoint, development/test CSVs and task configuration.
Outputs: summary CSV, per-window CSV, NPZ curves and PNG.
Scope: results depend on model, cohort and replacement rule.
"""
```

## 5. 기존 docs 파일마다 유지·이동·삭제

이번 적용은 최소안이다. “전체 재배치 시 제안”은 아직 실행한 변경이 아니다.

| 기존 파일·묶음 | 이번 조치 | 전체 재배치 시 제안 |
|---|---|---|
| `RUNBOOK.md` | REPRODUCE로 연결하는 호환 안내 | 새 링크가 정착하면 제거 가능 |
| `PAPER_CLAIM_MAP.md` | PAPER_MAP 안내; 이전 본문은 history에 보존 | 새 링크 정착 후 안내 파일 제거 가능 |
| `CLEANUP.md` | 유지·역사 표시 | history로 이동 |
| `CLEANUP_COMPARISON.md` | 유지·역사 표시 | history로 이동 |
| `LEGACY_EXCLUSIONS.md` | 유지·역사 표시 | history로 이동 또는 legacy 안내와 통합 |
| `EXPERIMENT_STATUS.md` | 유지·역사 표시 | history로 이동 |
| `PAPER_CODE_AUDIT.md` | 유지·역사 표시 | history로 이동; 해결 여부는 별도 현재 문서로 |
| `README_ORIGINAL.md` | 유지·상대 링크 수정 | history로 이동 |
| `BRANCH_PERFORMANCE_EXPERIMENT.md` | 유지·역사 표시 | history로 이동 |
| `POOLING_EXPERIMENT.md` | 선택 기준 문서로 유지 | experiments로 이동 |
| `GITHUB_DESKTOP.md` | 독자 경로에서 제외·삭제 | 개인 게시 안내가 필요하면 저장소 밖에서 관리 |
| `GITIGNORE_ORIGINAL.txt` | 삭제 | Git 이력으로 조회 |
| `PRESERVATION_MANIFEST.json` | 기존 위치 유지·수정 파일의 해시만 갱신 | verifier와 함께 provenance로 이동 |
| `SOURCE_FILES.json` | 유지 | provenance로 이동 가능 |
| `validation/` | 당시 검증 기록으로 유지 | provenance로 이동 가능 |

`models/README.md`는 비교 모델 문서로 유지한다. `prism_figure_revision/README_KO.md`는 통합 도구 기준으로 다시 썼다. 통합 원고의 수정 안내는 역사 관계를 명시하고 대조실험 문서는 미실행 계획임을 표시했다.

## 6. 목표 폴더 구조

### 이번에 적용한 최소안

```text
README.md / README.ko.md / CITATION.cff / CHANGELOG.md
lob_xai/          # 기존 import 경로 유지
models/           # 기존 baseline 경로 유지
configs/paper/    # toxic.args, vacuum.args
scripts/          # 기존 명령 + 문서 검사
prism_figure_revision/
docs/
  README.md / REPRODUCE.md / DATA.md / PAPER_MAP.md
  KNOWN_ISSUES.md / REPOSITORY_GUIDE.md
  history/        # 역사 색인과 이전 claim map
  기존 역사 문서·해시 원장·검증 로그
figures/ / frozen/ / loggers/ / manuscripts/ / legacy/
```

### 후속 전체 재배치안

```text
prism_lob/
  baselines/
configs/paper/
scripts/train/ / evaluate/ / analysis/
checkpoints/paper/
results/paper/ / archive/
paper/
docs/experiments/ / history/ / provenance/
tests/
```

이 안은 import, checkpoint 내 클래스 경로, state-dict 로딩, CLI, figure input manifest, 보존 verifier를 함께 확인해야 한다. 경로 이동만으로 provenance를 잃을 수 있으므로 현재 업데이트에서는 적용하지 않았다. Release·LFS·라이선스도 실제 생성·선택 전에는 존재하는 것으로 안내하지 않는다.

## 7. 공개 전환과 작업 체크리스트

저장소 공개 전환 자체는 이번 업데이트에 포함하지 않는다.

**공개 전**

- [ ] 공저자와 코드·원고·데이터의 배포 범위 확인.
- [ ] 코드 라이선스 선택, third-party 모델·데이터 조건 확인.
- [x] 확인된 저자·제목으로 CITATION.cff 작성; 미확정 DOI는 생략.
- [ ] 원고의 내부 메모와 개인 경로·이메일·서버 정보 검토. 과거 Git 이력도 대상.
- [ ] Vacuum 수치와 최종 표의 run 출처 확정.
- [ ] 원 배포본→동결 CSV 변환 경로 확보.
- [ ] 공개용 checkpoint/결과 배포 방식 결정; 실제 Release 생성 전 링크 금지.
- [ ] 새로운 환경에서 README 설치·실행 확인 및 외부 독자 사용성 점검.
- [ ] About/Topics와 공개 상태를 소유자가 최종 확인.

**우선순위별 작업**

- [x] P0: README, 논문 preset, 현재 결과 경로, 데이터 요구사항, 깨진 현재 링크 정리.
- [ ] P0: 원고 수치 출처와 데이터 변환 이력 확정.
- [x] P1: 모든 표·그림 대응표, docstring, 역사 문서 구분, 이름 대응 추가.
- [ ] P1: Baseline run 선택 manifest, Kendall fallback 및 유효 표본 수 보고 수정.
- [x] P2: 문서 링크 CI 단계, citation, manuscript 언어 분류 추가.
- [ ] P2: 대규모 재배치·패키지명 통일·대용량 결과의 별도 배포 검토.

새 파일을 줄이는 것보다 독자가 실제 실행 경로와 증거의 상태를 정확하게 찾는 것이 완료 기준이다.
