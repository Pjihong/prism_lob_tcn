> **역사 기록:** 아래 내용은 당시 버전의 작업·실험 상태다. 현재 실행법은 [REPRODUCE.md](REPRODUCE.md), 현재 결과 출처는 [PAPER_MAP.md](PAPER_MAP.md), 남은 문제는 [KNOWN_ISSUES.md](KNOWN_ISSUES.md)를 따른다. 경로와 명령을 현재 실행 지시로 사용하지 않는다.

# Paper–code audit

## 결론

논문의 방법론적 뼈대와 세 baseline source는 코드로 구현되어 있지만, 현재 제공물만으로는 논문 수치를 재현하거나 제출본을 확정할 수 없다. 코드에서 결과 변경형 결함을 수정했으므로 탐지 성능과 XAI 결과는 전부 다시 계산해야 한다. 데이터·checkpoint·run metadata가 없는 상태에서 기존 숫자를 ‘검증됨’으로 표시하지 않았다.

## 제출 차단 항목

| 우선순위 | 문제 | 필요한 결정 또는 조치 |
|---|---|---|
| P0 | FI-2010 CSV, checkpoint, run metadata가 없음 | 수정 코드로 전체 재학습·재평가 |
| P0 | 미정의 결과 매크로 26개와 누락 그림 11개 | 검증된 산출물에서 매크로·그림 생성 후 LaTeX 재컴파일 |
| P0 | ours MCC가 주석의 `0.179/0.186`과 표의 `0.109/0.116`으로 충돌 | 동일 checkpoint·동일 split에서 재평가해 하나만 유지 |
| P0 | G2의 “material/multiplicative difference”에 수치 임계가 없음 | 데이터 확인 전에 effect-size/ratio 임계와 CI 판정 규칙 사전 명시 |
| P0 | 반사실 이웃 수 `K`가 원고에 없음 | 코드 기준 `K=50`, `M=10`을 원고에 명시하거나 둘을 함께 변경 |
| P0 | 정규화 provenance가 충돌함 | FI-2010 제공 z-score를 그대로 썼는지, train-only 통계를 다시 fit했는지 하나로 통일하고 hash/통계 저장 |
| P0 | abstract는 DeepLOB gate를 Grad-CAM이라 하고 표는 HiResCAM이라 함 | headline 방법을 하나로 통일하고 다른 방법은 sensitivity로 분리 |
| P0 | 각 삭제율에서 span을 재선택하면서 곡선을 decision survival로 해석함 | ‘비중첩 retention curve’로 명칭 변경하거나 진짜 nested deletion path 구현 |
| P1 | toxic 설명은 “consecutive changes”이나 식·코드는 endpoint `t` 대 `t+k` 비교 | 서술을 endpoint 기준으로 고치거나 label 정의·실험을 재작성 |
| P1 | insertion의 `rho=0` donor 기준이 불명확함 | training pool의 unconditional full-window draw임을 명시 |
| P1 | 겹치는 100-step 창에 독립 표본 검정을 적용함 | stock/day/block 단위 bootstrap 또는 clustered uncertainty를 본 분석으로 추가 |
| P1 | 현재 원고는 stock 경계를 사용할 수 없는 single-segment 분석임 | 일반화 한계를 유지하고, 경계 자료를 확보하면 전 실험 재실행 |
| P1 | 각 TCN 최종 시점의 local receptive field는 29이지만 원고는 100-step 전체 문맥을 dilation이 포착한다고 서술 | GAP가 100개 위치를 집계한다는 주장과 단일 표현의 RF=29를 구분해 기술 |
| P1 | 반합성 표는 take-rate 하나로 ours와 DeepLOB 시간 국소화를 비교해 채점 cohort가 불명확함 | 모델별 성공 주입 cohort와 take-rate를 각각 보고하고, 공통 교집합은 보조분석으로 분리 |
| P1 | 추가 제공 baseline은 self-contained 구현이지만 upstream commit/tag가 없음 | “원 논문의 구현 그대로”를 주장하려면 출처 revision을 기록하고, 아니면 “재구현”으로 표현 |

정적 감사의 전체 기계 판정은 `audit/paper_contract_audit.json`에 저장한다.

## 발견한 결과 변경형 코드 결함과 수정

| 영역 | 기존 동작 | 수정된 동작 |
|---|---|---|
| split | 제안 모델과 baseline의 loader/split 규칙이 달랐음 | 동일한 시간순 segment-aware loader와 70/15/15 검증 사용 |
| 결측치 | 결측 행을 버려 원래 떨어진 시점을 이어 붙일 수 있었음 | 결측 또는 열 수 불일치 시 즉시 실패 |
| 학습 | focal/Adam/무 scheduler/작은 batch 등 원고와 다른 기본값 | weighted CE/AdamW/cosine/256/100/patience 10/val macro-F1 |
| 성능 ablation | depth만 바꾸고 중첩 창을 무작위 train/val 분할했으며 모든 후보를 test에서 비교 | test feature/label을 열지 않는 `--dev-only`, 시간순 validation, paired-seed 수집 및 명시적 최종 test 개봉 |
| branch 성능 옵션 | pure GAP와 정규화 없는 TCN, GAP 뒤에서만 branch 결합 | 기존 기본값은 보존하고 channel-LayerNorm, 고정 avg+last, 명시적 interaction stream을 opt-in ablation으로 추가 |
| 불균형 진단 | 전체 역빈도 가중치 한 가지만 가능 | 역빈도 지수와 per-class support/prediction/precision/recall/F1을 저장해 collapse·과다 경보를 검출 |
| TLOB head | 최종 `Linear(..., 3)`이 고정되어 vacuum 2-class와 불일치 | `n_classes`를 constructor/builder/head 전 구간에 전달하고 출력 검증 |
| TLOB 의존성 | self-contained라고 했지만 `einops`가 필요했음 | 동일한 transpose/reshape 연산으로 바꿔 외부 의존성 제거 |
| BiN-C(TABL) constraint | `forward()` 중 `.data`로 parameter를 변경해 평가도 stateful | optimizer step 직후 명시적 max-norm projection, forward는 무상태 |
| baseline provenance | source revision과 실제 architecture 크기가 checkpoint에서 분리 | source SHA-256, model config, 분석적/실제 parameter count 일치 검사 |
| checkpoint | `strict=False`로 일부 weight만 적재해도 진행 가능 | 논문 경로는 strict loading과 출력 class 차원 검증 |
| CAM hook | 마지막 등록 Conv1d가 1×1 residual shortcut일 수 있었음 | post-residual, pre-GAP branch output을 기본 attribution target으로 고정 |
| CAM class | toxic/vacuum class 이름·범위가 불완전 | 과제별 class mapping과 target 범위 검증 |
| 모델 간 Grad-CAM | DeepLOB에만 ReLU를 적용해 제안 모델과 CAM 정의가 달랐음 | 두 모델 모두 signed Grad-CAM을 만든 뒤 평가에서 `|CAM|` 사용 |
| CAM 방법 상관 | 동률을 임의 순번으로 바꾸는 ordinal rank 사용 | 동률 평균순위를 쓰는 Spearman으로 교체 |
| CF 이웃 | query는 보존 context, 후보는 전체 창으로 요약 | query·후보 모두 같은 보존 context로 거리 계산 |
| CF Monte Carlo | 가장 가까운 `M`개를 결정적으로 사용 | 가까운 `K=50`개에서 seeded `M=10` 추출 |
| random span | 호출마다 독립 난수 생성 | run seed에서 전달한 단일 RNG 사용 |
| donor pool | validation 또는 test donor로 대체 가능 | development CSV의 training partition만 허용 |
| 평가 cohort | 무작위/첫 창 또는 예측만 기준 | 정확한 target alert 중 confidence 상위 창을 안정 정렬로 선택 |
| branch share | cohort 전체 CAM 질량의 비율을 계산해 큰-scale 창이 과대 가중됨 | 윈도우별 branch share를 먼저 계산한 뒤 cohort 평균 |
| branch baseline | test-set 평균 또는 input temporal mean | training partition의 내부 branch GAP 평균 |
| Axis A | signed CAM과 test baseline, 양측 검정 혼재 | `|HiResCAM|`, training baseline, 원고의 단측 sign test |
| Axis B | 불완전 안정성 proxy와 상관 크기만으로 판정 | companion spectral radius, condition number, Spearman p-value 포함 |
| 사례 선택 | 앞부분 일부만 검색하고 최고 confidence라고 표기 | 전체 test cohort를 스캔한 뒤 선택 |
| horizon 그림 | 일부 checkpoint가 없어도 남은 지평만 조용히 그림 | 요청한 모든 지평·양 모델이 없으면 중단하고 각 입력 hash 검증 |
| 실행 라벨 | 명시적 checkpoint가 CLI의 task/k/seed와 달라도 출력 이름은 CLI를 따름 | checkpoint 저장 protocol과 CLI가 불일치하면 즉시 중단 |
| 그림 | 실제 입력이 없으면 empirical-looking 합성값으로 대체 | 명시적 demo만 허용하고 `NOT FOR PAPER` watermark 적용 |
| provenance | 데이터·checkpoint identity가 결과와 분리 | SHA-256 검증, run manifest, metrics schema 추가 |

## 통계·해석상 주의

1. 설명 cohort는 모델별로 따로 뽑힌 ‘정확한 target alert 상위 400개’다. 모델 간 동일 사건 비교가 목적이라면 공통 교집합 cohort를 별도 보조분석으로 보고해야 한다.
2. `argmax` deletion span을 삭제율마다 다시 찾으면 span이 중첩되지 않는다. 이 곡선의 AUC는 비교 요약량일 수 있지만 time-to-event 생존량은 아니다.
3. Axis A의 branch가 4개뿐이어서 branch-mean 순위검정의 해상도가 매우 낮다. 창 단위 검정도 중첩 의존성을 무시하면 p-value가 지나치게 작아질 수 있다.
4. Axis B의 VAR는 설명 신호의 예측 구조이지 시장 인과 구조가 아니다. 개입 일치도와 안정성 gate를 통과해도 ‘모델 내부 구조에 대한 증거’로 한정해야 한다.
5. FI-2010 z-score 단위의 tick threshold를 경제적 tick과 동일시하지 말고, 원 변환 통계와 종목별 tick 대응을 보고해야 한다.

## 반드시 다시 실행할 실험

- ours/DeepLOB/TLOB/BiN-CTABL × toxic/vacuum × 논문 지평 × seeds 1–3 탐지 성능
- vacuum `k=10` 탐지 비교표와 `k=30` 운영 실험의 구분 검증
- ours/DeepLOB의 CAM/random/anti deletion·insertion 전체 곡선과 창 단위 결과
- Grad-CAM/Grad-CAM++/HiResCAM 민감도 분석
- branch drop-one/CAM 수렴, VAR–ablation 구조 검증
- 전체 test cohort에서의 대표 사례와 모든 실증 그림
- block/segment 단위 uncertainty와 다중검정 보정

새 결과가 나오기 전에는 기존 표·그림·매크로를 수정 코드의 결과처럼 재사용하면 안 된다.
