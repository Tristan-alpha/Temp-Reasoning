#!/bin/bash

# Configuration variables
DATASET="hybrid_reasoning"
temp_group="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
MODEL="WizardMath-7B-V1.1"
MODEL_SIZE="7B"  # Default to 7B, can be changed to 34B if needed

# Update ReasonEval path based on model size
if [ "$MODEL_SIZE" == "34B" ]; then
    REASONEVAL_PATH="GAIR/ReasonEval-34B"
else
    REASONEVAL_PATH="GAIR/ReasonEval-7B"
fi

echo "Running evaluation for $MODEL on dataset $DATASET with automatic GPU allocation"
echo "Using ReasonEval-$MODEL_SIZE model"

python3 /home/dazhou/ReasonEval/t-codes/evaluate_results.py \
    --gpu auto \
    --models "$MODEL" \
    --dataset_name $DATASET \
    --temperatures $temp_group \
    --model_size $MODEL_SIZE \
    --reasoneval_path $REASONEVAL_PATH

echo "Evaluation completed for $MODEL with automatic GPU allocation"