#!/bin/bash

# Configuration variables
SCRIPT_PATH="/home/dazhou/ReasonEval/t-codes/evaluate_results.py"
DATASET="hybrid_reasoning"
TEMPERATURES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1)
# TEMPERATURES=(0.0 0.2)
# TEMPERATURES=(0.1 0.4 0.5 0.7 0.8 0.9 1.1 1.2 1.4 1.5)
NUM_TEMPS=${#TEMPERATURES[@]}
MODEL_SIZE="7B"  # Default model size

# GPU Selection - modify this array to specify which GPUs to use
GPU_LIST=(3 4 0 1 2)  # Default GPUs to use - Change this to set specific GPUs
NUM_GPUS=${#GPU_LIST[@]}  # Calculate number of GPUs from the list

# Local Models to evaluate (add more models as needed)
MODELS=(
    # "WizardMath-7B-V1.1" 
    # "Abel-7B-002"
    # "gpt-4o-mini"
    # "deepseek-v3"
    # "deepseek-r1"
    "deepseek-chat"
    # "deepseek-reasoner"
    # "claude-3-7-sonnet-20250219"
    # "gemini-2.0-flash"
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

# if [ "$MODEL_SIZE" == "7B" ]; then
#     # For each model
#     gpu_index=0  # Initialize outside the loop to distribute models across GPUs
#     for model in "${MODELS[@]}"; do
#         gpu=${GPU_LIST[$gpu_index]}
#         create_eval_session "$model" $gpu "${TEMPERATURES[*]}"
        
#         # Move to next GPU in rotation
#         ((gpu_index=(gpu_index+1)%NUM_GPUS))
#     done
# else
#     create_eval_session "${MODELS[0]}" "${GPU_LIST[*]}" "${TEMPERATURES[*]}"
# fi

create_eval_session "${MODELS[0]}" "${GPU_LIST[*]}" "${TEMPERATURES[*]}"

echo "All evaluation screen sessions have been created. Use 'screen -ls' to list active sessions."
echo "To attach to a session, use 'screen -r session_name'"
echo "Using ReasonEval-$MODEL_SIZE for all evaluations"