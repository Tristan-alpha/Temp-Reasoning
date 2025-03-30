#!/bin/bash

# Configuration variables
SCRIPT_PATH="/home/dazhou/ReasonEval/t-codes/answer_generation.py"
DATASET="hybrid_reasoning"
SUBSET_SIZE=0
TEMPERATURES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5)

# Parse command line arguments
GPU=${1:-0}  # Default to GPU 0 if not specified
MODEL=${2:-"WizardMath-7B-V1.1"}  # Default model if not specified

# Function to run answer generation for a specific model on a specific GPU
run_model() {
    local model=$1
    local gpu=$2
    
    echo "Running answer generation for $model on GPU $gpu"
    python $SCRIPT_PATH \
        --gpu $gpu \
        --subset_size $SUBSET_SIZE \
        --temperatures ${TEMPERATURES[@]} \
        --models $model \
        --dataset_name $DATASET
}

# Main execution
echo "Starting answer generation process for $DATASET dataset on GPU $GPU"

# Run the specified model on the specified GPU
run_model "$MODEL" "$GPU"

echo "Answer generation completed for $MODEL on GPU $GPU"