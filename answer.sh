#!/bin/bash

# Configuration variables
SCRIPT_PATH="/home/dazhou/ReasonEval/t-codes/answer_generation.py"
DATASET="hybrid_reasoning"
SUBSET_SIZE=0
TEMPERATURES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5)

# Parse command line arguments
GPU="auto"  # Use auto GPU allocation
MODEL=${1:-"WizardMath-7B-V1.1"}  # Default model if not specified

# Function to run answer generation for a specific model with auto GPU allocation
run_model() {
    local model=$1
    
    echo "Running answer generation for $model with automatic GPU allocation"
    python $SCRIPT_PATH \
        --gpu auto \
        --subset_size $SUBSET_SIZE \
        --temperatures ${TEMPERATURES[@]} \
        --models $model \
        --dataset_name $DATASET
}

# Main execution
echo "Starting answer generation process for $DATASET dataset with automatic GPU allocation"

# Run the specified model with automatic GPU allocation
run_model "$MODEL"

echo "Answer generation completed for $MODEL with automatic GPU allocation"