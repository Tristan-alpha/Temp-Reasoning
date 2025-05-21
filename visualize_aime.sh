#!/bin/bash

# Configure visualization for Qwen on AIME datasets
SCRIPT_PATH="/home/dazhou/ReasonEval/t-codes/visualize_temperature_results.py"
MODEL="Qwen3-32B"
DATASET="aime"
INPUT_DIR="/home/dazhou/ReasonEval/evaluation_results"
OUTPUT_DIR="/home/dazhou/ReasonEval/evaluation_results/aime_visualizations"
EVALUATORS="ReasonEval_7B ReasonEval_34B"
DETAILED_FLAG="--detailed"

# Use same temperature range as in answer.sh
TEMPERATURES="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5"

echo "Starting visualization for $MODEL on $DATASET dataset"

# Make sure output directory exists
mkdir -p $OUTPUT_DIR

# Run visualization
python $SCRIPT_PATH \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --models $MODEL \
    --datasets $DATASET \
    --evaluators $EVALUATORS \
    --temperatures $TEMPERATURES \
    $DETAILED_FLAG \
    --custom_output_name "qwen_aime"

echo "Visualization complete!"
echo "Results saved to: $OUTPUT_DIR"
