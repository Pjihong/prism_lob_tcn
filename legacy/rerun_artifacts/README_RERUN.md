# 재실험 킷 — 최종 모델을 positional(양 과제 통일)로 확정하기 위한 toxic 재실행
(2026-08-25. 실행 위치: ~/timeseries/lob-xai, PYTHONPATH=. 필수)

## 전제와 원칙

- **바뀌는 것**: ours의 toxic 산출물 전부(탐지·CF·Axis A/B·sanity·factorization·alert card·모집단 통계).
- **바뀌지 않는 것(재사용)**: vacuum ours(이미 positional), DeepLOB/TLOB/BiN 전 결과, 라벨·분할·프로토콜.
- **체크포인트**: frozen/positional_6ckpt_manifest.json 에 positional 3-seed × 2-task 6개가 이미
  동결되어 있으면 **재학습 불필요** — toxic positional(identity norm) ckpt를 그대로 쓴다.
  없거나 SHA 불일치면 91_train_fallback.sh 로 3-seed 재학습.
- **서사 주의(원고)**: 구 원고의 "validation 게이트가 과제별 readout을 선택(독성=multi_scale)"
  서사는 폐기된다. 새 서사 = "단일 readout(positional)으로 통일"이며, 이때
  (a) positional-vs-GAP의 toxic validation 비교 수치(poolab 로그)와
  (b) multi_scale이 val에서 더 좋았다는 사실을 한계/각주로 정직하게 보고해야 한다.
  → 95_collect.sh 가 poolab 요약도 함께 수집한다.

## 실행 순서

    export PYTHONPATH=. DEVICE=cuda
    export TRAIN_CSV=data/FI2010/FI2010_train.csv TEST_CSV=data/FI2010/FI2010_test.csv
    bash rerun_kit/00_verify_ckpts.sh          # 동결 ckpt 확인 (실패 시 91_train_fallback.sh)
    bash rerun_kit/10_final_test.sh            # test 재평가 → figures/final_test_positional/
    bash rerun_kit/20_cf_final.sh              # CF deletion+insertion (ours toxic, 3-seed)
    bash rerun_kit/30_carpet_stats.sh          # response carpet(ours toxic) + cf_effect_stats 갱신
    bash rerun_kit/40_axis.sh                  # branch probe → Axis A(+windows) → Axis B(참고)
    bash rerun_kit/50_sanity.sh                # cam_sanity (toxic, 3-seed)
    bash rerun_kit/60_figures.sh               # factorization/branch_importance/alert card/cf_mirror/readout
    bash rerun_kit/95_collect.sh               # 논문 갱신용 산출물 tar 수집 → 업로드

각 스크립트는 독립 실행 가능하고, 산출물 경로를 stdout 마지막에 출력한다.
GPU 기준 예상 소요: 10_final_test 수 분, 20_cf 약 30–60분, 30_carpet 약 15분,
40–60 각 수 분–수십 분. 재학습 fallback은 seed당 ~30–60분.

## 원고에서 무효가 되는 수치 (재실행 후 교체 대상)

- tab:detection ours toxic 행 (구 0.161±0.021 / PR 0.095)
- tab:cf_main·tab:cf_appendix ours toxic 두 행 (flip/recovery/ΔAUC/p)
- §res_methods: multi_scale 혼합 가중 m=(0.68,...)·83:6.8:1·계단(last-32/8) 문단 전체
  → positional α_t 서술로 교체 (factorization_toxic 그림도 재생성본으로)
- §res_completeness: drop-one 3.56/2.58/1.39/0.73, CAM share, τ 0.477/0.405/0.491
- §res_branch: 모집단 통계(매도측 88–94% 등), alert card toxic(창 21586) 전부
- abstract/§1.5/§6.2: readout 채택 게이트 수치 +0.063(독성) 및 "과제별 선택" 서사
- fig:architecture 캡션의 "독성은 multi-scale" 문구
