> **역사 기록:** 아래 내용은 당시 버전의 작업·실험 상태다. 현재 실행법은 [REPRODUCE.md](REPRODUCE.md), 현재 결과 출처는 [PAPER_MAP.md](PAPER_MAP.md), 남은 문제는 [KNOWN_ISSUES.md](KNOWN_ISSUES.md)를 따른다. 경로와 명령을 현재 실행 지시로 사용하지 않는다.

# Branch-TCN 성능 실험 (depth sweep 진단과 Stage 0–3)

## 기존 depth sweep이 확인한 것

제공된 toxic-flow 결과는 "TCN block을 늘리면 성능이 좋아진다"는 일반 결론을 지지하지 **않는다**. 3-seed 기준:

| horizon | model | MCC mean±sample SD | macro-F1 | PR-AUC_nf |
|---|---|---:|---:|---:|
| k=10 | L3 | 0.1221±0.0116 | 0.3484±0.0193 | 0.0584±0.0076 |
| k=10 | L4 | 0.1287±0.0072 | 0.3703±0.0056 | 0.0658±0.0012 |
| k=10 | L5 | 0.1247±0.0017 | 0.3860±0.0024 | 0.0719±0.0109 |
| k=30 | L3 | 0.0822±0.0120 | 0.3691±0.0143 | 0.1085±0.0070 |
| k=30 | L4 | 0.0738±0.0184 | 0.3645±0.0109 | 0.1075±0.0142 |
| k=30 | L5 | 0.0704±0.0277 | 0.3697±0.0165 | 0.1057±0.0091 |
| k=50 | L3 | 0.0560±0.0054 | 0.3521±0.0061 | 0.1363±0.0030 |
| k=50 | L4 | 0.0377±0.0095 | 0.3471±0.0105 | 0.1360±0.0123 |
| k=50 | L5 | 0.0700±0.0201 | 0.3724±0.0129 | 0.1490±0.0105 |

kernel 3, dilation base 2에서 L3/L4/L5의 최종 시점 receptive field는 29/61/125다. L5는 이미 100-step 창을 전부 덮는데도 일관되게 낫지 않다. 따라서 병목은 receptive field 단독이 아니다.

레거시 `train_tcn_sweep.py`는 겹치는 창을 무작위 분할해(인접 창이 100행 중 99행을 공유) validation이 독립 시간 블록이 아니었고, cosine 없는 Adam을 쓰며 모든 후보를 test에서 비교했다. 논문 모델 선택에 사용하지 말 것.

## 이 저장소에 추가된 opt-in 구조 변형

기존 기본값과 old checkpoint state key는 모두 보존된다.

1. `--tcn-norm layer`: 각 residual TCN block 내부의 시점별 channel-wise LayerNorm.
2. `--pool avg_last`: 전역 평균과 마지막 TCN state의 고정 연결. 학습되는 시간 attention 가중치는 없다.
3. `--interaction-channels 64`: 네 독립 TCN encoder 뒤의 time-aligned 1×1 interaction stream. 의미론적 encoder는 분리 유지되지만 설명은 **주효과 4개 + 명시적 interaction 항 1개**가 된다. 이 모델을 감사할 때는 `compute_cam_all_branches(..., include_interaction=True)`를 사용할 것.
4. `--class-weight-power 0.5`: 전체 역빈도 가중(1.0)의 완화 대안. 구조 계열 선택 후에만 시험하고, 아키텍처와 loss 가중을 동시에 바꾸지 말 것.

(이후 branch-local pooling `multi_scale`/`positional`이 같은 opt-in 원칙으로 추가되었다 — `docs/POOLING_EXPERIMENT.md` 참고.)

대략적 toxic-flow 파라미터 수: 141,955(원본), 144,003(LayerNorm), 160,387(LayerNorm+avg_last), 185,155(+interaction 64), 4-block interaction 모델 284,995. 실제 수는 각 run에 출력·저장된다.

## Stage 0: label·split 감사

프로젝트 루트에서:

```bash
export PYTHONPATH=.

python scripts/audit_tick_size.py \
  --train-csv data/FI2010/FI2010_train.csv \
  --out-csv figures/tick_size_audit.csv

python scripts/audit_label_prevalence.py \
  --train-csv data/FI2010/FI2010_train.csv \
  --test-csv data/FI2010/FI2010_test.csv \
  --k 10 --out-dir figures/label_audit_k10
```

실제 stock/day 경계가 있으면 `--train-boundaries`/`--test-boundaries`로 제공한다. 단일 z-space tick size와 단일 segment 가정을 점수가 편하다는 이유로 유지해서는 안 된다.

제공된 CSV 크기상 두 파일이 합산 행의 70/15/15가 아니다. 아래의 `--allow-split-mismatch`는 진단 전용이다. 최종 논문에서는 선언된 split을 재구성하거나 원고를 실현된 split로 수정해야 한다.

## Stage 1: k=10 validation 전용 구조 ablation

k=30/50은 아직 돌리지 말 것. 먼저 정규화/readout/time-aligned interaction 중 무엇이 k=10 격차를 줄이는지 확인한다.

```bash
export PYTHONPATH=.

run_cfg () {
  tag="$1"; shift
  for seed in 1 2 3; do
    python scripts/train_microstructure.py \
      --train-csv data/FI2010/FI2010_train.csv \
      --test-csv data/FI2010/FI2010_test.csv \
      --task toxic --k 10 --seed "$seed" \
      --backbone tcn --cnn-kernel 3 --tcn-dilation-base 2 \
      --loss ce_weighted --class-weight-power 1.0 \
      --optimizer adamw --scheduler cosine \
      --epochs 100 --early-stop-patience 10 \
      --early-stop-metric val_f1 \
      --lr 1e-3 --weight-decay 1e-5 --batch-size 256 \
      --device cuda --dev-only --skip-report --allow-split-mismatch \
      --name "branchdev_${tag}_toxic_k10_seed${seed}" "$@"
  done
}

run_cfg base       --cnn-channels 32 32 64    --pool avg      --tcn-norm none  --interaction-channels 0
run_cfg norm       --cnn-channels 32 32 64    --pool avg      --tcn-norm layer --interaction-channels 0
run_cfg avg_last   --cnn-channels 32 32 64    --pool avg_last --tcn-norm layer --interaction-channels 0
run_cfg mixer      --cnn-channels 32 32 64    --pool avg_last --tcn-norm layer --interaction-channels 64
run_cfg mixer_l4   --cnn-channels 32 32 64 64 --pool avg_last --tcn-norm layer --interaction-channels 64

python scripts/collect_branch_dev_results.py \
  --results-dir loggers/results \
  --name-contains branchdev_ \
  --out-csv figures/branch_dev_validation.csv
```

전진 조건: 평균 validation MCC와 PR-AUC가 모두 `base`보다 개선되고, MCC 이득이 3 seed 중 최소 2개의 paired seed에서 나타나며, 어떤 희소 class도 recall 0이 아니어야 한다. checkpoint early stopping은 현행 원고대로 validation macro-F1을 유지한다. 아키텍처 선택에 validation MCC를 쓴다면 논문에 별도로 명시할 것.

## Stage 2: loss 가중, 한 번에 한 요인만

최선의 구조 계열에 대해서만 `--class-weight-power 0.5`로 3 seed를 반복한다. 전체 역빈도 가중이 만드는 과다 경보를 줄이는 경우가 많지만 보장된 이득은 아니다. MCC 하나가 아니라 전체 confusion matrix와 class별 recall을 비교할 것.

## Stage 3: test 1회 개봉

이 명령을 실행하기 전에 구성 하나를 동결한다. validation으로 선택된 이미 학습된 checkpoint를 평가하며, test를 본 후 재학습하지 않는다.

```bash
python scripts/evaluate_ours_checkpoint.py \
  --checkpoint loggers/results/SELECTED_SEED1_RUN/model.pt \
  --test-csv data/FI2010/FI2010_test.csv \
  --device cuda --confirm-final-test
```

사전 선언된 seed-2, seed-3 checkpoint에 대해서만 반복한다. 동일한 데이터·label·loss·optimizer·schedule·checkpoint 규칙으로 학습한 DeepLOB/TLOB checkpoint와 그 세 결과를 비교한다.

`TLOB > ours > DeepLOB` 순서를 강제할 과학적으로 타당한 방법은 없다. 현재 표에서 DeepLOB–TLOB MCC 격차는 0.011에 불과해 3-seed "약간 아래" 순서는 표집 잡음일 수 있다. 타당한 주장은 봉인된 paired-seed 평가가 지지하는 것뿐이다.

마지막으로, 새 checkpoint를 채택하면 모든 CAM·반사실 삭제·branch ablation·그림/표 결과를 다시 실행해야 한다. interaction 변형은 원고의 "정확히 4항 분해" 주장도 무효화한다 — 주효과 4개 + 명시적 interaction 항으로 서술할 것.
