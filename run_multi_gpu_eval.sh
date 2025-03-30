#!/bin/bash

# Configuration variables
SCRIPT_PATH="/home/dazhou/ReasonEval/t-codes/evaluate_results.py"
DATASET="hybrid_reasoning"
TEMPERATURES=(0.0 0.3 0.6 1.0 1.3 1.6)

# Local Models to evaluate (add more models as needed)
LOCAL_MODELS=(
    "WizardMath-7B-V1.1" 
    "Abel-7B-002"
)

# API Models to evaluate
API_MODELS=(
    "gpt-4o-mini"
    "deepseek-v3"
    "deepseek-r1"
    "claude-3-7-sonnet-20250219"
    "gemini-2.0-flash"
)

# Parse command line arguments
NUM_GPUS=${1:-5}  # Default to 5 GPUs if not specified
RUN_LOCAL=${2:-true}  # Whether to run local models
RUN_API=${3:-false}   # Whether to run API models

# Function to create a screen session for a local model evaluation on a specific GPU
create_eval_session() {
    local model=$1
    local gpu=$2
    local session_name="eval_${model//[^a-zA-Z0-9]/_}_gpu${gpu}"
    
    echo "Creating evaluation screen session $session_name for $model on GPU $gpu"
    
    # Create detached screen session
    screen -dmS "$session_name" bash -c "cd /home/dazhou/ReasonEval && python $SCRIPT_PATH --gpu $gpu --models \"$model\" --dataset_name $DATASET --temperatures ${TEMPERATURES[@]}; exec bash"
    
    echo "Evaluation screen session $session_name created"
}

# Main execution
echo "Starting multiple GPU evaluation tasks"

# Launch local model screen sessions distributing across GPUs
if [ "$RUN_LOCAL" = true ]; then
    echo "Setting up local model evaluation on $NUM_GPUS GPUs"
    for i in "${!LOCAL_MODELS[@]}"; do
        gpu=$((i % $NUM_GPUS))  # Distribute across available GPUs
        create_eval_session "${LOCAL_MODELS[i]}" $gpu
        
        # Add a small delay to avoid screen creation conflicts
        sleep 1
    done
fi

# Launch API model screen sessions distributing across GPUs
if [ "$RUN_API" = true ]; then
    echo "Setting up API model evaluation"
    for i in "${!API_MODELS[@]}"; do
        gpu=$(( (i + ${#LOCAL_MODELS[@]}) % $NUM_GPUS ))  # Continue distributing across GPUs
        create_eval_session "${API_MODELS[i]}" $gpu
        
        # Add a small delay to avoid screen creation conflicts
        sleep 1
    done
fi

echo "All evaluation screen sessions have been created. Use 'screen -ls' to list active sessions."
echo "To attach to a session, use 'screen -r session_name'"