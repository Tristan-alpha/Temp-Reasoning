#!/bin/bash

# Example script to run answer generation with real-time evaluation and wandb logging

export CUDA_VISIBLE_DEVICES=4,5,6,7 # Specify the GPUs to use

# Base path variable for easy modification, not ending with a slash
BASE_PATH="$(cd "$(dirname "$0")"; pwd)"

# Set parameters
DATASETS=(
    math-1
    math-2
    math-3
    math-4
    math-5
    # aime
)
MODELS=(
    # "WizardMath-7B-V1.1" 
    # "Abel-7B-002"
    # 'Qwen3-0.6B'
    # 'Qwen3-4B'
    # 'Qwen3-8B'
    # 'Qwen3-14B'
    # 'Qwen3-32B'
    'Qwen3-30B-A3B'
    # 'Tongyi-DeepResearch-30B-A3B'
)
# TEMPERATURES=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5) # full
TEMPERATURES=(1.6 1.7 1.8 1.9 2.0 2.1 2.2 2.3 2.4 2.5) # extra
# TEMPERATURES=(0.2 0.4 0.6 0.8 1.0 1.2 1.4) # test

SUBSET_SIZE=0  # Use a subset for testing; set to 0 for full dataset
TENSOR_PARALLEL_SIZE=4 # Number of GPUs to use
GPU_MEMORY_UTILIZATION=0.9
BATCH_SIZE=1  # Batch size for evaluation model inference - increase this to reduce the number of processed prompts

# Paths
INPUT_PATH="${BASE_PATH}/dataset"
OUTPUT_DIR="${BASE_PATH}/Results"
EVAL_MODEL_SIZE=34B
REASONEVAL_PATH="GAIR/ReasonEval-$EVAL_MODEL_SIZE"
SHEPHERD_PATH="peiyi9979/math-shepherd-mistral-7b-prm"

# WandB settings
WANDB_ENTITY="dazhou_liu2023-southern-university-of-science-technology"
WANDB_PROJECT="Temperature-Reasoning-Evaluation"

echo "Starting answer generation with real-time evaluation and wandb logging..."

cd "${BASE_PATH}/t-codes"

for DATASET_NAME in "${DATASETS[@]}"
do
    echo "Processing $DATASET_NAME"
    python answer_generation.py \
        --input_path "$INPUT_PATH" \
        --dataset_name "$DATASET_NAME" \
        --output_dir "$OUTPUT_DIR" \
        --models "${MODELS[@]}" \
        --temperatures ${TEMPERATURES[@]} \
        --subset_size $SUBSET_SIZE \
        --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
        --gpu_memory_utilization $GPU_MEMORY_UTILIZATION \
        --vllm_dtype "auto" \
        --max_model_len 8192 \
        --batch_size $BATCH_SIZE \
        --logger \
        --entity "$WANDB_ENTITY" \
        --project "$WANDB_PROJECT" \
        --enable_evaluation \
        --reasoneval_path "$REASONEVAL_PATH" \
        --reasoneval_model_size "$EVAL_MODEL_SIZE" \
        --shepherd_path "$SHEPHERD_PATH" \
        --log_token_probs
done

echo "Evaluation completed!"
