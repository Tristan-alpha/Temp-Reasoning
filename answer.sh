#!/bin/bash

# Configuration variables
SCRIPT_PATH="/home/dazhou/ReasonEval/t-codes/answer_generation.py"
MODEL="WizardMath-7B-V1.1"
DATASET="aime"
SUBSET_SIZE=0
TEMPERATURES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5)
GPU="auto"

echo "Starting answer generation process for $DATASET dataset with automatic GPU allocation"

python $SCRIPT_PATH \
    --gpu auto \
    --subset_size $SUBSET_SIZE \
    --temperatures ${TEMPERATURES[@]} \
    --models $MODEL \
    --dataset_name $DATASET

echo "Answer generation completed for $MODEL with automatic GPU allocation"