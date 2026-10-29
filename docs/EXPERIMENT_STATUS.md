> **역사 기록:** 아래 내용은 당시 버전의 작업·실험 상태다. 현재 실행법은 [REPRODUCE.md](REPRODUCE.md), 현재 결과 출처는 [PAPER_MAP.md](PAPER_MAP.md), 남은 문제는 [KNOWN_ISSUES.md](KNOWN_ISSUES.md)를 따른다. 경로와 명령을 현재 실행 지시로 사용하지 않는다.

# 실험 이력과 현재 상태 (2026-07 기준)

이 문서는 흩어져 있던 패키지들을 교차 대조해 실험이 어떻게 진행되었는지 재구성하고, 동결된 자산과 남은 결정을 기록한다.

## 1. 이력 재구성

**1세대 — mid-price 방향 예측 (`lob_xai` v3.x).** FI-2010 mid-price 3-class 예측. CNN+attention vs TCN+avg, type4/topdeep8 branching, branch별 Grad-CAM, 8-concept TCAV, regime/trend 조건부 CAM(fig00–fig17, figa–figj), rebuttal 대응(DeepLOB/TLOB, 4-way input×gradient faithfulness). attention pool의 last-step collapse 관찰이 이 시기에 기록됨 — 이후 positional readout 서사의 출발점.

**2세대 — 논문 pivot (`lob_xai_paper_code`).** 과제를 미시구조 위험 탐지(toxic k=10 3-class, vacuum k=30 2-class)로 재정의. 설명 체계를 HiResCAM + 분포 보존 반사실 + branch 분해(Axis A/B/C)로 압축하고, 결과 변경형 결함(split/CAM hook/donor pool/cohort 등)을 수정해 프로토콜을 `paper_protocol.py`에 동결. depth sweep 진단: L5(RF=125)가 100-step 창을 덮어도 개선 없음 → **병목은 receptive field가 아니라 readout**.

**3세대 — 입력 정규화·가중치 dev 선택 (`devsel_*` run 계열).** GAP readout에서 validation 전용 비교로 toxic=`revin_c`, vacuum=`identity` 입력 정규화와 class-weight-power 1.0을 선택. `scripts/run_locked_current_final_test.sh`가 이 계열의 봉인 test 개봉 절차.

**4세대 — branch-local pooling ablation (2026-07-21~22).** 두 sibling 저장소에서 GAP 대조군과 seed-일치 비교(`--dev-only`, test 봉인):

- 1순위 `multi_scale` (`lob_xai_multiscale_pooling`): branch당 tail 평균(8/32/100) convex 혼합, +12 파라미터. 7/21 16:54~ 학습.
- 2순위 `positional` (`lob_xai_positional_pooling`): branch당 100-tick softmax 위치 가중, +400 파라미터, 0-초기화 시 정확히 GAP. 7/22 00:58~03:23 학습.

validation 판정 결과가 `frozen/task_specific_selected_6ckpt_manifest.json`에 동결됨: **toxic=multi_scale(revin_c), vacuum=positional(identity)**.

**5세대 — readout 감사 (2026-07-24).** checkpoint에서 위치 가중치만 직접 읽는 audit(`scripts/common/plot_positional_readout.py`, 데이터 무접촉, config 검증 포함) 실행. 산출물: `fig_positional_readout_{toxic,vacuum}` PNG/PDF/CSV/JSON.

## 2. Positional readout 감사 결과 (3-seed 평균, GAP 기준 last-8=0.08, E[lag]=49.5)

| task | branch | last-8 | last-32 | E[lag] | entropy |
|---|---|---:|---:|---:|---:|
| toxic | ask_price | 0.185±0.017 | 0.444±0.019 | 41.1±0.9 | 0.969 |
| toxic | ask_volume | 0.284±0.043 | 0.510±0.027 | 36.9±1.8 | 0.947 |
| toxic | **bid_price** | **0.367±0.019** | **0.592±0.008** | **31.4±0.6** | 0.897 |
| toxic | bid_volume | 0.292±0.039 | 0.558±0.017 | 34.9±1.5 | 0.940 |
| vacuum | ask_price | 0.079±0.001 | 0.316±0.005 | 50.0±0.5 | 1.000 |
| vacuum | **ask_volume** | **0.231±0.008** | **0.538±0.008** | **34.7±0.6** | 0.946 |
| vacuum | bid_price | 0.080±0.001 | 0.313±0.006 | 50.4±0.6 | 0.999 |
| vacuum | **bid_volume** | **0.253±0.024** | **0.568±0.027** | **33.2±1.9** | 0.939 |

해석: toxic은 네 branch 모두 최근으로 기울며 bid_price가 가장 강함. vacuum은 price branch가 GAP와 사실상 동일(균등)하고 volume branch만 최근으로 기욺 — vacuum label이 volume 급감으로 정의된다는 의미론과 branch 수준에서 일치. entropy ≥ 0.897로 last-step collapse 아님. seed 간 편차 작음.

해석 경계: 이 가중치는 입력 독립적 branch-local readout prior다. attention map이나 창 단위 saliency로 서술하지 말 것. 창 단위 증거는 HiResCAM과 반사실 검정이 담당한다.

## 3. 알려진 긴장 지점 (원고 반영 전 정리 필요)

1. **선택 모델과 감사 대상 불일치(toxic).** validation 선택은 toxic=multi_scale인데, toxic readout figure는 *선택되지 않은* positional checkpoint에서 계산됨. 원고에 positional figure를 양 과제로 실으려면 (a) positional을 공통 최종 아키텍처로 승격하거나 (b) toxic figure를 multi_scale 혼합 가중(3개 값) 서술로 바꾸거나 (c) toxic positional 결과를 부록 관찰로 격하해야 함.
2. **과제별 아키텍처/전처리 분기.** toxic=multi_scale+revin_c, vacuum=positional+identity. "단일 아키텍처" 서사가 약해지므로, 사전 규칙에 따른 validation 선택임을 명시하고 선택 절차를 재현 가능하게 기술해야 함.
3. **성능 확인 미완.** pooling 채택의 근거인 validation 비교표(`figures/poolab_*_full_validation.csv`)와 최종 test 개봉 결과가 이 저장소 밖(실행 환경)에 있음. GAP 대비 이득 수치와 사전 gate(POOLING_EXPERIMENT: 3-seed 평균 MCC +0.03 이상, 부호 일치, PR-AUC 비퇴행, class collapse 없음) 통과 여부를 문서화할 것.
4. **채택 시 재실행 의무.** readout 교체는 CAM 분포·반사실·branch 분석·모든 표/그림의 재계산을 요구함(감사 규칙). "GAP라서 Grad-CAM==HiResCAM" 논증 삭제 포함.
5. **carpet cohort 차이.** carpet 기본 cohort(`stratified_correct`)는 집계 반사실 cohort(top-400)와 다름. 두 산출물을 함께 쓸 때 캡션에 명시.
6. **분할 선언 확정(해결됨, v2.1).** 제공 CSV(dev 362,400행 + test 31,937행)로는 합산 70/15/15가 재구성 불가(실현 75.7/16.2/8.1). 모든 frozen run이 이 현실에서 학습됐으므로 `assert_paper_split`를 "development 내 82.35/17.65 + 비어 있지 않은 test file" 계약으로 교체했고, 원고의 분할 서술도 이에 맞춰 수정해야 함.
7. **poolab 학습 예산이 논문 프로토콜과 다름.** `run_pooling_experiment.sh`는 Adam(무 scheduler), toxic 40 epochs/patience 8, vacuum 20/4로 학습함 — 논문 선언(AdamW, cosine, 100, patience 10)과 다르고 baseline 재학습은 논문 프로토콜을 따르므로, 현 상태로는 탐지 표 비교가 불공정함. 선택된 pooling 구성(toxic multi_scale+revin_c, vacuum positional+identity)을 논문 프로토콜로 3-seed 재학습해 재동결하거나, 최종 모델의 학습 서술을 재선언해야 함. 재학습 시 readout figure·carpet·XAI도 새 checkpoint 기준으로 재생성.
8. **vacuum validation의 양성 희소성.** validation 구간에 vacuum 창이 103개(0.16%)뿐 — train 2.32%, test 2.39%와 크게 다름(시간 비정상성). vacuum의 validation 기반 선택·early stopping이 이 103개에 의존함을 원고 한계에 명시할 것.

## 4. 남은 작업 (권장 순서)

1. validation 비교표와 (개봉했다면) 최종 test 결과를 `frozen/`에 SHA와 함께 추가.
2. 위 3-1 결정(공통 positional vs 과제별 선택) 확정 → 원고 아키텍처 절 수정.
3. 확정 checkpoint로 CAM/반사실/Axis A·B·C/carpet 전체 재실행.
4. 재실행 시 신규 감사 도구 포함: `cam_sanity_checks.py`(method 일치도·안정성·parameter randomization)와 `cf_effect_stats.py`(carpet 창 단위 paired AUC + bootstrap CI + Wilcoxon, 겹침 주의 문구 포함).
5. `scripts/verify_positional_checkpoint.py --manifest ...`로 산출물-코드 호환 재확인 후 그림·표 재생성.

## 5. 논문 프로토콜 재학습(final_*) 결과 (2026-07-28)

선택된 두 구성을 논문 프로토콜(AdamW/cosine/100ep/patience10, dev-only)로 3-seed 재학습.

- validation: toxic multi_scale MCC 0.1385/0.1317/0.1215 (평균 0.131), vacuum positional MCC 0.4134/0.4139/0.3509 (평균 0.393; seed3은 val 양성 103개 잡음으로 ep9 조기 종료).
- **readout 재현 — vacuum(positional)**: 해리가 더 선명해짐. price branch last-8 = 0.86–0.92× GAP(entropy ≥0.99), volume branch last-8 = 4.34×/4.89× GAP(E[lag] 25.9/23.0), 최근 1 tick에 질량 ~20–22%(seed1/2). seed3(조기 종료)만 recency 약함.
- **readout 재현 — toxic(multi_scale)**: w8/w32/w100 3-seed 평균 — ask_price 0.729/0.179/0.092, bid_price 0.764/0.155/0.081, ask_volume 0.587/0.168/0.245, bid_volume 0.589/0.221/0.191. price branch가 volume보다 최근-집중(E[lag] ≈9–10 vs ≈15–17).
- 해석: 두 과제에서 readout이 label 의미론을 거울상으로 복원. 전 6 checkpoint strict 로드·검증 통과, `frozen/final_6ckpt_manifest.json`으로 동결(데이터·checkpoint SHA-256 포함).

## 6. 원고 개정 결정 (2026-07-29 확정)

- **D1 탐지 표 지평**: 운영 지평 보고로 변경(toxic k=10, vacuum k=30). vacuum k=10 추가 학습 불요. 과제 간 직접 비교 서술은 완화.
- **D2 지평 스윕**: GAP-era 결과로 명시해 유지(readout 교체 전 아키텍처의 강건성 점검임을 본문/부록에 선언). baseline 행만 재학습분으로 교체.
- **D3 insertion(충분성)**: `lob_xai/counterfactual_test.py`에 `--direction {deletion,insertion,both}`로 구현 완료(v2.2). rho=0 기준선 = training pool 무조건부 전체-창 추출(감사 P1 정의 확정), 세 mode가 창별 동일 donor 기준선 공유(짝지음), 비중첩 곡선 해석 경계 문서화. final_* 재실행 시 `--direction both` 사용.
- **D5 readout prior 결과**: 이번 투고에서는 보류. 후속 논문 재료로 관리(figure·수치·해석은 RESEARCH_SUMMARY와 본 문서 §5에 보존).
- 추가 재서술 확정: "GAP라서 Grad-CAM==HiResCAM" 기제 문단은 학습 readout에서 성립하지 않으므로 재측정(`cam_sanity_checks.py`) 후 실증 서술로 교체. "결정 생존 곡선" 명명은 비중첩 retention으로 교체.

## 7. 재학습 큐 중간 판독 (2026-07-30, seed3 일부 진행 중)

**Baseline test (감사 코드·논문 프로토콜, 운영 지평):**

| 모델 | toxic k=10 MCC | vacuum k=30 MCC |
|---|---|---|
| DeepLOB | 0.098/0.092/0.162 → **0.117±0.038** (구 0.181±0.003에서 급락, 분산 증가) | 0.252/0.239/0.223 → **0.238±0.015** (구 0.251과 유사) |
| TLOB | 0.000/0.000/0.000 — **전 seed all-majority 붕괴** | 0.000/0.000(seed3 진행) — 붕괴 |
| BiN-CTABL | 0.170/0.202(seed3 진행) | 0.240/0.254(seed3 진행) |

**GAP 대조군 (validation, dev-only):** toxic GAP 0.057/0.067 vs multi_scale 0.139/0.132/0.122 → ΔMCC ≈ **+0.07** (gate +0.03 크게 상회). vacuum GAP 0.387/0.331 vs positional 0.413/0.414 → +0.026/+0.083 (seed3 대기; seed1 PR-AUC는 GAP이 소폭 우위 0.241 vs 0.214 — gate의 "PR-AUC 비퇴행" 항목은 seed3 후 판정).

**DeepLOB seed1 반사실 (del+ins, final 프로토콜):** deletion ΔAUC 0.230/0.232/0.241 (구 0.11–0.13의 ~2배 응집), flip cam 0.99–1.00/0.84 vs rand 0.36–0.38/0.02. **insertion 첫 실증**: recover(cam) 0.95/0.75/1.00 vs recover(rand) 0.02–0.04 — "소수 구간이 필요이자 충분"(응집형) 서사를 강하게 지지. toxic_up 정타 74개(재학습 DeepLOB의 정타 경보 급감) 기록.

**서사 영향(주의):** "ours가 가장 낮고 MCC 0.07을 지불" 구도가 흔들림 — ours(val) 0.131 vs DeepLOB(test) 0.117. 확정은 ours test 개봉 후. TLOB은 통일 프로토콜에서 전면 붕괴 → 원고의 "원 논문 full-capacity 그대로" 주장과 정합하려면 TLOB 원 레시피(하이퍼파라미터) 별도 학습 검토 필요.

**v2.3:** `run_response_carpet.sh`가 기본으로 final_* ours + 재학습 DeepLOB checkpoint를 보도록 패치(`OURS_*_PREFIX`/`DEEPLOB_*_PREFIX` env 오버라이드, 출력은 `CKPT_SET=final` 네임스페이스로 분리 — poolab-era 완료 run과 충돌·skip 없음).
