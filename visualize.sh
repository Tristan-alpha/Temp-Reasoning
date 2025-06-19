#!/bin/bash

# Configure visualization for models on AIME datasets
SCRIPT_PATH="/home/dazhou/ReasonEval/t-codes/visualize_temperature_results.py"
MODELS="Abel-7B-002 WizardMath-7B-V1.1 Qwen3-0.6B Qwen3-4B Qwen3-8B Qwen3-14B Qwen3-32B Qwen3-30B-A3B"
DATASET="aime"
INPUT_DIR="/home/dazhou/ReasonEval/evaluation_results"
OUTPUT_DIR="/home/dazhou/ReasonEval/visualizations"
EVALUATORS="ReasonEval_7B ReasonEval_34B"

# Use same temperature range as in answer.sh
TEMPERATURES="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5"

echo "Starting visualization for models on $DATASET dataset"

# Make sure output directory exists
mkdir -p $OUTPUT_DIR

# Run visualization
python3 $SCRIPT_PATH \
    --input_dir $INPUT_DIR \
    --output_dir $OUTPUT_DIR \
    --models $MODELS \
    --datasets $DATASET \
    --evaluators $EVALUATORS \
    --temperatures $TEMPERATURES

echo "Visualization complete!"
echo "Results saved to: $OUTPUT_DIR"
