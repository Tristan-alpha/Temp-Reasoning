#!/bin/bash

# Configuration variables
SCRIPT_PATH="/home/dazhou/ReasonEval/t-codes/evaluate_results.py"
DATASET="hybrid_reasoning"
TEMPERATURES=(0.0 0.2 0.3 0.6 1.0 1.3 1.6)
NUM_TEMPS=${#TEMPERATURES[@]}
MODEL_SIZE="7B"  # Default model size

# GPU Selection - modify this array to specify which GPUs to use
GPU_LIST=(4 5 6 7)  # Default GPUs to use - Change this to set specific GPUs
NUM_GPUS=${#GPU_LIST[@]}  # Calculate number of GPUs from the list

# Run settings
RUN_LOCAL=true  # Whether to run local models
RUN_API=false   # Whether to run API models

# Local Models to evaluate (add more models as needed)
LOCAL_MODELS=(
    "WizardMath-7B-V1.1" 
    # "Abel-7B-002"
)

# API Models to evaluate
API_MODELS=(
    "gpt-4o-mini"
    "deepseek-v3"
    "deepseek-r1"
    "claude-3-7-sonnet-20250219"
    "gemini-2.0-flash"
)

# Display selected GPUs
echo "Using GPUs: ${GPU_LIST[*]}"

# Validate MODEL_SIZE
if [ "$MODEL_SIZE" != "7B" ] && [ "$MODEL_SIZE" != "34B" ]; then
    echo "Error: MODEL_SIZE must be either '7B' or '34B'"
    exit 1
fi

echo "Using ReasonEval-$MODEL_SIZE for evaluation"

# Update ReasonEval path based on model size
if [ "$MODEL_SIZE" == "34B" ]; then
    REASONEVAL_PATH="GAIR/ReasonEval-34B"
else
    REASONEVAL_PATH="GAIR/ReasonEval-7B"
fi

# Function to create a screen session for a local model evaluation on a specific GPU with a specific temperature group
create_eval_session() {
    local model=$1
    local gpu=$2
    local temperature_group=$3
    local session_name="eval_${model//[^a-zA-Z0-9]/_}_gpu${gpu}_${MODEL_SIZE}"
    
    echo "Creating evaluation screen session $session_name for $model on GPU $gpu with temperatures $temperature_group"
    
    # Create detached screen session with multiple temperatures
    screen -dmS "$session_name" bash -c "cd /home/dazhou/ReasonEval && python $SCRIPT_PATH --gpu $gpu --models \"$model\" --dataset_name $DATASET --temperatures $temperature_group --model_size $MODEL_SIZE --reasoneval_path $REASONEVAL_PATH; exec bash"
    
    echo "Evaluation screen session $session_name created"
}

# Main execution
echo "Starting multiple GPU evaluation tasks with ReasonEval-$MODEL_SIZE"

# Launch local model screen sessions distributing temperatures across GPUs
if [ "$RUN_LOCAL" = true ]; then
    echo "Setting up local model evaluation on specified GPUs with distributed temperatures"
    
    # For each local model
    for model in "${LOCAL_MODELS[@]}"; do
        # For each GPU
        for gpu_index in $(seq 0 $((NUM_GPUS-1))); do
            gpu=${GPU_LIST[$gpu_index]}
            temp_group=""
            
            # Gather all temperatures for this GPU
            for temp_index in $(seq 0 $((NUM_TEMPS-1))); do
                if [ $((temp_index % NUM_GPUS)) -eq $gpu_index ]; then
                    if [ -z "$temp_group" ]; then
                        temp_group="${TEMPERATURES[$temp_index]}"
                    else
                        temp_group="$temp_group ${TEMPERATURES[$temp_index]}"
                    fi
                fi
            done
            
            # Create a single session for this model-GPU combination with all assigned temperatures
            if [ ! -z "$temp_group" ]; then
                create_eval_session "$model" $gpu "$temp_group"
                sleep 1
            fi
        done
    done
fi

# Launch API model screen sessions with distributed temperatures
if [ "$RUN_API" = true ]; then
    echo "Setting up API model evaluation with distributed temperatures"
    
    # For each API model
    for model in "${API_MODELS[@]}"; do
        # For each GPU
        for gpu_index in $(seq 0 $((NUM_GPUS-1))); do
            gpu=${GPU_LIST[$gpu_index]}
            temp_group=""
            
            # Gather all temperatures for this GPU
            for temp_index in $(seq 0 $((NUM_TEMPS-1))); do
                if [ $((temp_index % NUM_GPUS)) -eq $gpu_index ]; then
                    if [ -z "$temp_group" ]; then
                        temp_group="${TEMPERATURES[$temp_index]}"
                    else
                        temp_group="$temp_group ${TEMPERATURES[$temp_index]}"
                    fi
                fi
            done
            
            # Create a single session for this model-GPU combination with all assigned temperatures
            if [ ! -z "$temp_group" ]; then
                create_eval_session "$model" $gpu "$temp_group"
                sleep 1
            fi
        done
    done
fi

echo "All evaluation screen sessions have been created. Use 'screen -ls' to list active sessions."
echo "To attach to a session, use 'screen -r session_name'"
echo "Using ReasonEval-$MODEL_SIZE for all evaluations"