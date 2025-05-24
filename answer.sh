#!/bin/bash

# Configuration variables
SCRIPT_PATH="/home/dazhou/ReasonEval/t-codes/answer_generation.py"
MODEL="Qwen3-30B-A3B"  # Specify the model you want to use
DATASET="aime"
SUBSET_SIZE=0
TEMPERATURES=(0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 0.0 0.1 0.2 0.3)
# TEMPERATURES=(0.1 0.2 0.3)

# Use 3 GPUs (modify these numbers based on your available GPUs)
CUDA_VISIBLE_DEVICES=2,3

echo "Starting answer generation process for $DATASET dataset"

export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export HF_HOME="/home/data/dazhou/.cache/huggingface"
export HF_ENDPOINT="https://hf-mirror.com"

python $SCRIPT_PATH \
    --subset_size $SUBSET_SIZE \
    --temperatures ${TEMPERATURES[@]} \
    --models $MODEL \
    --dataset_name $DATASET \
    --use_vllm True \
    --tensor_parallel_size 2 \
    --gpu_memory_utilization 0.95

echo "Answer generation completed for $MODEL using GPUs $CUDA_VISIBLE_DEVICES"