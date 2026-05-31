#!/bin/bash
set -e

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${_SCRIPT_DIR}/.." && pwd)"
cd "$ROOT"

export WANDB_DIR="${ROOT}/wandb"
export WANDB_DATA_DIR="${ROOT}/wandb/data"
export WANDB_CACHE_DIR="${ROOT}/wandb/cache"
mkdir -p "$WANDB_DIR" "$WANDB_DATA_DIR" "$WANDB_CACHE_DIR"
conda activate safesteer

MODEL="your_Llama-3-8B-Instruct_model_path"  # e.g., ./ckpts/meta-llama/Meta-Llama-3-8B-Instruct
LOG_DIR="${ROOT}/logs"
mkdir -p "$LOG_DIR"

# ----- Per-run knobs -----
GPUS="0,1"   # GPU pair: Student on the first, Teacher on the second
H=5          # --safe_token_horizon
K=0          # --num_loss_tokens_to_keep (0 = keep all)

COMMON_ARGS="--model_name $MODEL \
    --learning_rate 1e-6 \
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
    --min_steered_prob 1e-7 \
    --freeze_teacher True \
    --update_refusal_vector False \
    --freeze_safe_token True \
    --log_teacher_completions False \
    --renormalize_selected_tokens False"

# Descriptive tag for the log filename, built from the run config.
MODEL_TAG="$(basename "$MODEL")"
TAG="${MODEL_TAG}_mode2_top50_vote_h${H}_k${K}"

CUDA_VISIBLE_DEVICES=$GPUS python main.py $COMMON_ARGS \
    --safe_token_horizon $H \
    --num_loss_tokens_to_keep $K \
    > "${LOG_DIR}/${TAG}.log" 2>&1

echo "[run] finished. Log: ${LOG_DIR}/${TAG}.log"
