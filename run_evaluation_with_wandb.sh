#!/bin/bash

# Example script to run answer generation with real-time evaluation and wandb logging

export CUDA_VISIBLE_DEVICES=1,2,0,3,4 # Specify the GPUs to use

# Set parameters
DATASET_NAME="math-3"  # or "aime"
MODELS=(
    # "WizardMath-7B-V1.1" 
    # "Abel-7B-002"
    # 'Qwen3-32B'
    # 'Qwen3-30B-A3B'
    'Qwen3-14B'
    'Qwen3-8B'
    'Qwen3-4B'
    'Qwen3-0.6B'
    # 'Qwen3-4B'
    # 'Qwen3-8B'
    # 'Qwen3-14B'
    # 'Qwen3-30B-A3B'
    # 'Qwen3-32B'
)
TEMPERATURES=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5)
SUBSET_SIZE=100  # Use a subset for testing; set to 0 for full dataset
TENSOR_PARALLEL_SIZE=2 # Number of GPUs to use
GPU_MEMORY_UTILIZATION=0.65
BATCH_SIZE=2  # Batch size for vLLM inference - increase this to reduce the number of processed prompts

# Paths
INPUT_PATH="/home/data/dazhou/ReasonEval/dataset"
OUTPUT_DIR="/home/data/dazhou/ReasonEval/Results"
EVAL_MODEL_SIZE=7B
REASONEVAL_PATH="GAIR/ReasonEval-$EVAL_MODEL_SIZE"
SHEPHERD_PATH="peiyi9979/math-shepherd-mistral-7b-prm"

# WandB settings
WANDB_ENTITY="dazhou_liu2023-southern-university-of-science-technology"
WANDB_PROJECT="Temperature-Reasoning-Evaluation"

echo "Starting answer generation with real-time evaluation and wandb logging..."

cd /home/data/dazhou/ReasonEval/t-codes

python answer_generation.py \
    --input_path "$INPUT_PATH" \
    --dataset_name "$DATASET_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --models "${MODELS[@]}" \
    --temperatures ${TEMPERATURES[@]} \
    --subset_size $SUBSET_SIZE \
    --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
    --gpu_memory_utilization $GPU_MEMORY_UTILIZATION \
    --vllm_dtype "half" \
    --max_model_len 4096 \
    --batch_size $BATCH_SIZE \
    --logger \
    --entity "$WANDB_ENTITY" \
    --project "$WANDB_PROJECT" \
    --enable_evaluation \
    --reasoneval_path "$REASONEVAL_PATH" \
    --reasoneval_model_size "$EVAL_MODEL_SIZE" \
    --shepherd_path "$SHEPHERD_PATH" \
    --log_token_probs

echo "Evaluation completed!"
