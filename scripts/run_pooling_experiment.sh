#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LOB_XAI_ROOT="$ROOT"

: "${EXPERIMENT_POOL:?set EXPERIMENT_POOL=multi_scale or positional}"
MODE="${MODE:-screen}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-256}"
TRAIN_CSV="${TRAIN_CSV:-$ROOT/../lob_tcn_xai/data/FI2010/FI2010_train.csv}"
TEST_CSV="${TEST_CSV:-$ROOT/../lob_tcn_xai/data/FI2010/FI2010_test.csv}"

case "$EXPERIMENT_POOL" in
  multi_scale|positional) ;;
  *) echo "EXPERIMENT_POOL must be multi_scale or positional" >&2; exit 2 ;;
esac
case "$MODE" in
  screen) SEEDS=(1) ;;
  full) SEEDS=(1 2 3) ;;
  *) echo "MODE must be screen or full" >&2; exit 2 ;;
esac
[[ -f "$TRAIN_CSV" ]] || { echo "TRAIN_CSV not found: $TRAIN_CSV" >&2; exit 2; }
[[ -f "$TEST_CSV" ]] || { echo "TEST_CSV not found: $TEST_CSV" >&2; exit 2; }

mkdir -p logs figures loggers/results
shopt -s nullglob
set -o pipefail

echo "[protocol] pool=$EXPERIMENT_POOL mode=$MODE seeds=${SEEDS[*]}"
echo "[protocol] test inference remains SEALED (--dev-only on every run)"
echo "[protocol] comparison is seed-matched GAP vs $EXPERIMENT_POOL"
python -u scripts/smoke_pooling.py --pool "$EXPERIMENT_POOL"

run_cell () {
  local task="$1" k="$2" norm="$3" pool="$4" seed="$5"
  local epochs patience
  if [[ "$task" == "toxic" ]]; then
    epochs=40
    patience=8
  else
    epochs=20
    patience=4
  fi

  local tag="poolab_${EXPERIMENT_POOL}_${MODE}_"
  local name="${tag}${task}_k${k}_db2_${norm}_w1_pool${pool}_seed${seed}"
  local completed=(loggers/results/${name}_*/summary.json)
  if (( ${#completed[@]} == 1 )); then
    echo "[skip] completed: ${completed[0]}"
    return
  fi
  if (( ${#completed[@]} > 1 )); then
    echo "[error] duplicate completed runs for $name; move duplicates before continuing" >&2
    exit 3
  fi

  python -u scripts/train_microstructure.py \
    --train-csv "$TRAIN_CSV" \
    --test-csv "$TEST_CSV" \
    --task "$task" \
    --k "$k" \
    --seed "$seed" \
    --backbone tcn \
    --cnn-channels 32 32 64 \
    --cnn-kernel 3 \
    --cnn-dropout 0.1 \
    --fc-hidden 64 \
    --tcn-dilation-base 2 \
    --tcn-norm none \
    --input-norm "$norm" \
    --pool "$pool" \
    --pool-windows 8 32 100 \
    --interaction-channels 0 \
    --loss ce_weighted \
    --class-weight-power 1.0 \
    --optimizer adam \
    --scheduler none \
    --lr 1e-3 \
    --weight-decay 0 \
    --epochs "$epochs" \
    --early-stop-patience "$patience" \
    --early-stop-metric val_mcc \
    --batch-size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --dev-only \
    --skip-report \
    --name "$name" \
    2>&1 | tee "logs/${name}.log"
}

SECONDS=0
for seed in "${SEEDS[@]}"; do
  run_cell toxic 10 revin_c avg "$seed"
  run_cell toxic 10 revin_c "$EXPERIMENT_POOL" "$seed"
  run_cell vacuum 30 identity avg "$seed"
  run_cell vacuum 30 identity "$EXPERIMENT_POOL" "$seed"
done

TAG="poolab_${EXPERIMENT_POOL}_${MODE}_"
OUT_CSV="figures/${TAG}validation.csv"
python -u scripts/collect_branch_dev_results.py \
  --results-dir loggers/results \
  --name-contains "$TAG" \
  --out-csv "$OUT_CSV"

CHECKPOINTS=(loggers/results/${TAG}*pool${EXPERIMENT_POOL}_seed*/model.pt)
if (( ${#CHECKPOINTS[@]} == 0 )); then
  echo "[error] no $EXPERIMENT_POOL checkpoints found" >&2
  exit 4
fi
python -u scripts/inspect_pool_weights.py \
  --checkpoints "${CHECKPOINTS[@]}" \
  --out-json "figures/${TAG}learned_weights.json"

printf '[done] elapsed=%02d:%02d:%02d\n' \
  "$((SECONDS/3600))" "$(((SECONDS%3600)/60))" "$((SECONDS%60))"
echo "[done] validation table: $ROOT/$OUT_CSV"
echo "[done] no test predictions or test metrics were produced"
