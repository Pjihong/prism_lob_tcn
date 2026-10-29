# Branch-local pooling 실험

네 개의 의미론적 TCN branch를 classifier head 이전까지 독립으로 유지한 채, 시간 readout 두 가지를 추가한다.

- `multi_scale`: 각 branch가 8/32/100 tick tail 평균의 convex 혼합을 학습한다. branch당 3개(총 12개) 파라미터가 추가된다.
- `positional`: 각 branch가 시점별 입력-독립 softmax 가중치 하나씩을 학습한다. 정확한 GAP로 시작하며 branch당 100개(총 400개) 파라미터가 추가된다.

두 방법 모두 sample-dependent attention이나 branch 간 혼합을 도입하지 않으며, branch activation에 선형이다. HiResCAM은 그대로 계산 가능하지만, **"시간 방향 gradient가 상수라서 Grad-CAM == HiResCAM"이라는 GAP 특수 논증은 어느 쪽을 채택하든 원고에서 제거해야 한다.**

## 실행 프로토콜

`scripts/run_pooling_experiment.sh`는 `EXPERIMENT_POOL`로 지정한 방법을 seed-일치 GAP 대조군과 **validation에서만** 비교한다. test feature/label은 로드하지 않는다(`--dev-only`).

스크리닝 (4 run: GAP/후보 × toxic/vacuum, seed 1):

```bash
TRAIN_CSV=.../FI2010_train.csv TEST_CSV=.../FI2010_test.csv \
EXPERIMENT_POOL=positional MODE=screen DEVICE=cuda \
bash scripts/run_pooling_experiment.sh
```

전체 확인 (12 run: GAP/후보 × 2 task × 3 seed):

```bash
TRAIN_CSV=... TEST_CSV=... \
EXPERIMENT_POOL=positional MODE=full DEVICE=cuda \
bash scripts/run_pooling_experiment.sh
```

스크립트는 먼저 forward/backward/strict-reload smoke test를 수행하고, 모든 cell을 학습한 뒤 validation CSV를 `figures/` 아래에 기록하고 학습된 시간 가중치를 JSON으로 내보낸다. 완료된 cell은 안전 재시작 시 건너뛰고, 중복 cell은 조용히 평균되지 않고 즉시 실패한다.

## 사전 등록 판정 기준 (paper 모델 교체 gate)

후보 − GAP가 다음을 모두 만족할 때만 교체를 고려한다.

- 3-seed 평균 validation MCC 이득 ≥ +0.03
- 세 paired seed 차이의 부호가 모두 동일
- non-flat PR-AUC가 실질적으로 퇴행하지 않음
- 희소 class collapse 없음

이 판정을 동결한 후에만 test를 평가한다.

## 참고 실행 시간

관측된 GPU 기준: screen 35–80분, full 2–4시간. early stopping과 저장 장치 속도에 따라 크게 달라질 수 있다.
