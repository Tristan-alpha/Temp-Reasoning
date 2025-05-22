#!/bin/bash

# Configuration variables
DATASET="aime"
temp_group="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5"
MODEL_SIZE="7B"  # Default to 7B, can be changed to 34B if needed
CUDA_VISIBLE_DEVICES=3,4
MODEL=(
    # "WizardMath-7B-V1.1" 
    # "Abel-7B-002"
    # 'Qwen3-0.6B'
    # 'Qwen3-4B'
    # 'Qwen3-8B'
    'Qwen3-32B'
)

# Update ReasonEval path based on model size
if [ "$MODEL_SIZE" == "34B" ]; then
    REASONEVAL_PATH="GAIR/ReasonEval-34B"
else
    REASONEVAL_PATH="GAIR/ReasonEval-7B"
fi

export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

echo "Running evaluation for $MODEL on dataset $DATASET with automatic GPU allocation"
echo "Using ReasonEval-$MODEL_SIZE model"

python3 /home/dazhou/ReasonEval/t-codes/evaluate_results.py \
    --models "${MODEL[@]}" \
    --dataset_name $DATASET \
    --temperatures $temp_group \
    --model_size $MODEL_SIZE \
    --reasoneval_path $REASONEVAL_PATH

echo "Evaluation completed for ${MODEL[@]} with automatic GPU allocation"