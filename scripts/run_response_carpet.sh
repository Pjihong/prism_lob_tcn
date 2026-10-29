#!/usr/bin/env bash
set -Eeuo pipefail
shopt -s nullglob

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BASE_ROOT="${BASE_ROOT:-$ROOT}"     # data location only
TRAIN_CSV="${TRAIN_CSV:-$BASE_ROOT/data/FI2010/FI2010_train.csv}"
TEST_CSV="${TEST_CSV:-$BASE_ROOT/data/FI2010/FI2010_test.csv}"

# Checkpoint run-name prefixes (seed number + _*/model.pt is appended).
# Defaults target the paper-protocol frozen runs in $ROOT/loggers/results.
# Override any prefix to audit a different checkpoint family; also set
# CKPT_SET so outputs never collide with another family's completed runs.
CKPT_SET="${CKPT_SET:-paper_positional}"
OURS_TOXIC_PREFIX="${OURS_TOXIC_PREFIX:-$ROOT/loggers/results/final_toxic_positional_seed}"
OURS_VACUUM_PREFIX="${OURS_VACUUM_PREFIX:-$ROOT/loggers/results/final_vacuum_positional_seed}"
DEEPLOB_TOXIC_PREFIX="${DEEPLOB_TOXIC_PREFIX:-$ROOT/loggers/results/deeplob_toxic_seed}"
DEEPLOB_VACUUM_PREFIX="${DEEPLOB_VACUUM_PREFIX:-$ROOT/loggers/results/deeplob_vacuum_seed}"

MODE="${MODE:-smoke}"               # smoke | screen | full
MODELS="${MODELS:-ours deeplob}"    # ours, deeplob, or both
TASKS="${TASKS:-toxic vacuum}"
METHOD="${METHOD:-hirescam}"
DEVICE="${DEVICE:-cuda}"
COHORT="${COHORT:-stratified_correct}"
ALLOW_UNVERIFIED="${ALLOW_UNVERIFIED:-0}"
SALIENCY_BATCH_SIZE="${SALIENCY_BATCH_SIZE:-32}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-128}"
COLOR_MAX="${COLOR_MAX:-1.25}"

case "$MODE" in
  smoke)
    SEEDS="${SEEDS:-1}"
    N_WINDOWS="${N_WINDOWS:-24}"
    POOL_SIZE="${POOL_SIZE:-3000}"
    MC="${MC:-3}"
    NEIGHBORS_K="${NEIGHBORS_K:-20}"
    Q_MAX="${Q_MAX:-0.20}"
    ;;
  screen)
    SEEDS="${SEEDS:-1}"
    N_WINDOWS="${N_WINDOWS:-120}"
    POOL_SIZE="${POOL_SIZE:-10000}"
    MC="${MC:-5}"
    NEIGHBORS_K="${NEIGHBORS_K:-30}"
    Q_MAX="${Q_MAX:-0.50}"
    ;;
  full)
    SEEDS="${SEEDS:-1 2 3}"
    N_WINDOWS="${N_WINDOWS:-400}"
    POOL_SIZE="${POOL_SIZE:-30000}"
    MC="${MC:-10}"
    NEIGHBORS_K="${NEIGHBORS_K:-50}"
    Q_MAX="${Q_MAX:-0.50}"
    ;;
  *)
    echo "[error] MODE must be smoke, screen, or full" >&2
    exit 2
    ;;
esac

cd "$ROOT"
export PYTHONPATH="$ROOT"
export MPLBACKEND=Agg

SCRIPT="scripts/run_cf_response_carpet.py"
test -s "$SCRIPT" || {
  echo "[error] missing $ROOT/$SCRIPT" >&2
  exit 1
}
test -f "$TRAIN_CSV" || { echo "[error] missing $TRAIN_CSV" >&2; exit 1; }
test -f "$TEST_CSV" || { echo "[error] missing $TEST_CSV" >&2; exit 1; }

mkdir -p "figures/cf_response_carpet/$CKPT_SET/$MODE" "logs/cf_response_carpet/$CKPT_SET/$MODE"

find_one () {
  local label="$1"
  shift
  local hits=("$@")
  if (( ${#hits[@]} != 1 )); then
    echo "[error] $label: expected one checkpoint, found ${#hits[@]}" >&2
    printf '  %s\n' "${hits[@]:-<none>}" >&2
    exit 1
  fi
  printf '%s\n' "${hits[0]}"
}

resolve_ckpt () {
  local model="$1"
  local task="$2"
  local seed="$3"
  local hits=()

  if [[ "$model" == "ours" && "$task" == "toxic" ]]; then
    hits=( "$OURS_TOXIC_PREFIX""$seed"_*/model.pt )
  elif [[ "$model" == "ours" && "$task" == "vacuum" ]]; then
    hits=( "$OURS_VACUUM_PREFIX""$seed"_*/model.pt )
  elif [[ "$model" == "deeplob" && "$task" == "toxic" ]]; then
    hits=( "$DEEPLOB_TOXIC_PREFIX""$seed"_*/model.pt )
  elif [[ "$model" == "deeplob" && "$task" == "vacuum" ]]; then
    hits=( "$DEEPLOB_VACUUM_PREFIX""$seed"_*/model.pt )
  else
    echo "[error] unsupported model/task: $model/$task" >&2
    exit 2
  fi
  find_one "$model/$task/seed$seed" "${hits[@]}"
}

SECONDS=0
for model in $MODELS; do
  for task in $TASKS; do
    for seed in $SEEDS; do
      ckpt="$(resolve_ckpt "$model" "$task" "$seed")"
      run_tag="${model}_${task}_${METHOD}_${COHORT}_seed${seed}_n${N_WINDOWS}_p${POOL_SIZE}_m${MC}_k${NEIGHBORS_K}_q${Q_MAX}"
      out_dir="figures/cf_response_carpet/$CKPT_SET/$MODE/$run_tag"
      log_file="logs/cf_response_carpet/$CKPT_SET/$MODE/${run_tag}.log"

      if [[ -s "$out_dir/run_manifest.json" ]] &&
         grep -q '"status": "complete"' "$out_dir/run_manifest.json"; then
        echo "[skip] completed: $out_dir"
        continue
      fi

      extra=()
      if [[ "$ALLOW_UNVERIFIED" == "1" ]]; then
        extra+=(--allow-unverified-data)
      fi

      echo "[run] mode=$MODE model=$model task=$task seed=$seed"
      echo "[run] ckpt=$ckpt"
      python -u "$SCRIPT" \
        --model "$model" \
        --ckpt "$ckpt" \
        --train-csv "$TRAIN_CSV" \
        --test-csv "$TEST_CSV" \
        --task "$task" \
        --method "$METHOD" \
        --cohort "$COHORT" \
        --n-windows "$N_WINDOWS" \
        --pool-size "$POOL_SIZE" \
        --mc "$MC" \
        --neighbors-k "$NEIGHBORS_K" \
        --q-max "$Q_MAX" \
        --q-step 0.05 \
        --seed "$seed" \
        --device "$DEVICE" \
        --saliency-batch-size "$SALIENCY_BATCH_SIZE" \
        --inference-batch-size "$INFERENCE_BATCH_SIZE" \
        --color-max "$COLOR_MAX" \
        --out-dir "$out_dir" \
        "${extra[@]}" \
        2>&1 | tee "$log_file"
    done
  done
done

printf '[success] elapsed=%02d:%02d:%02d\n' \
  $((SECONDS/3600)) $(((SECONDS%3600)/60)) $((SECONDS%60))

echo "[outputs]"
find "figures/cf_response_carpet/$CKPT_SET/$MODE" \
  -type f \( -name '*.png' -o -name '*.npz' -o -name '*.windows.csv' \) \
  -print | sort
