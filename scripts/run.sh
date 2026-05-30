#!/bin/bash
set -e

unset http_proxy https_proxy all_proxy
export WANDB_DISABLED=false
export WANDB_MODE=online
export WANDB_API_KEY="aebd2dbce5f9307b66375195bc8bb3d3ee7188c0"

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "$ROOT"

export WANDB_DIR="${ROOT}/wandb"
export WANDB_DATA_DIR="${ROOT}/wandb/data"
export WANDB_CACHE_DIR="${ROOT}/wandb/cache"
mkdir -p "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR"
source /home/data/lihao/miniconda3/etc/profile.d/conda.sh
conda activate distillation

MODEL="/ssddata/lihao/projects/models/Qwen3-4B-Instruct-2507"
LOG_DIR="${ROOT}/logs"
mkdir -p "$LOG_DIR"

# Grid: 28 NEW combos (5 already trained from W=1..8 sweep are pre-filtered out).
# Original 33 - {(1,1), (2,2), (4,4), (6,6), (8,8)} = 28 → exactly 7 phases × 4 tasks, no idle GPU pairs.
# Constraint: keep >= horizon (keep=0 => full sequence).
PAIRS=(
    "1 0" "2 0"
)

COMMON_ARGS="--model_name $MODEL \
    --learning_rate 1e-5 \
    --use_refusal_vector True \
    --alpha 1.0 \
    --voca_selection_mode 2 \
    --voca_selection_num 50 \
    --selection_method vote \
    --num_samples_per_prompt 8 \
    --safe_token_temperature 1.0 \
    --safe_token_top_p 1.0 \
    --exclude_special_tokens True \
    --vote_top_k_inner 200 \
    --min_steered_prob 1e-6 \
    --freeze_teacher False \
    --update_refusal_vector True \
    --freeze_safe_token False \
    --log_teacher_completions True"

GPU_LIST=("0,1" "2,3")

run_pair () {
    local H=$1
    local K=$2
    local GPUS=$3
    local TAG="Qwen3-4B_mode2_top50_vote_h${H}_k${K}"

    CUDA_VISIBLE_DEVICES=$GPUS python main.py $COMMON_ARGS \
        --safe_token_horizon $H \
        --num_loss_tokens_to_keep $K \
        > "${LOG_DIR}/${TAG}.log" 2>&1 &
    echo "  h=$H k=$K on GPU $GPUS (pid=$!)"
}

NUM_PAIRS=${#PAIRS[@]}
PHASE=1
i=0
while [ $i -lt $NUM_PAIRS ]; do
    echo "阶段 $PHASE: starting up to 2 tasks..."
    for g in 0 1; do
        if [ $i -ge $NUM_PAIRS ]; then break; fi
        read H K <<< "${PAIRS[$i]}"
        run_pair $H $K "${GPU_LIST[$g]}"
        i=$((i + 1))
    done
    wait
    echo "阶段 $PHASE 完成"
    PHASE=$((PHASE + 1))
done

echo "全部 ${NUM_PAIRS} 个 Qwen3-4B 网格任务完成"
