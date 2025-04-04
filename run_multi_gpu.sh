#!/bin/bash

# Configuration variables
SCRIPT_PATH="/home/dazhou/ReasonEval/t-codes/answer_generation.py"
API_SCRIPT_PATH="/home/dazhou/ReasonEval/t-codes/api_answer_generation.py"
DATASET="hybrid_reasoning"
SUBSET_SIZE=0
TEMPERATURES=(0.3 0.6 1.0)
# TEMPERATURES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6)
# TEMPERATURES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1)
# TEMPERATURES=(0.1 0.4 0.5 0.7 0.8 0.9 1.1 1.2 1.4 1.5)
# TEMPERATURES=(0.0 0.2 0.3 0.6 1.0 1.3 1.6)
NUM_TEMPS=${#TEMPERATURES[@]}  # Calculate number of temperatures

# API Keys
API_KEY=sk-dqbCjalqSxgKaqe4YyNGGByaNLFk6vv0gXp0LnErebFmTZkx
API_KEY_DEEPSEEK=sk-a00c3e30d7524683883aa59f82191ae4

# GPU Selection - modify this array to specify which GPUs to use
GPU_LIST=(0 1)  # Default GPUs to use - Change this to set specific GPUs
NUM_GPUS=${#GPU_LIST[@]}  # Calculate number of GPUs from the list

# Run settings
RUN_LOCAL=false  # Whether to run local models
RUN_API=true   # Whether to run API models

# Local Models to evaluate (add more models as needed)
LOCAL_MODELS=(
    # "WizardMath-7B-V1.1" 
    "Abel-7B-002"
)

# API Models to evaluate
API_MODELS=(
    # "gpt-4o-mini"
    # "deepseek-chat"
    "deepseek-reasoner"
    # "deepseek-v3"
    # "deepseek-r1"
    # "claude-3-7-sonnet-20250219"
    # "gemini-2.0-flash"
)

# Display selected GPUs
echo "Using GPUs: ${GPU_LIST[*]}"

# Ensure the hybrid dataset exists
if [ ! -f "/home/dazhou/ReasonEval/dataset/hybrid_reasoning.json" ]; then
  echo "Creating hybrid dataset..."
  python /home/dazhou/ReasonEval/dataset/create_hybrid_dataset.py
else
  echo "Hybrid dataset already exists, skipping creation step."
fi

# Function to create a screen session for a local model on a specific GPU with a specific temperature
create_local_model_session() {
    local model=$1
    local gpu=$2
    local temperature=$3  # This can now be multiple temperatures separated by spaces
    local session_name="answer_${model//[^a-zA-Z0-9]/_}_gpu${gpu}"
    
    echo "Creating screen session $session_name for $model on GPU $gpu with temperature(s) $temperature"
    
    # Create detached screen session
    screen -dmS "$session_name" bash -c "cd /home/dazhou/ReasonEval && python $SCRIPT_PATH --gpu $gpu --models \"$model\" --dataset_name $DATASET --subset_size $SUBSET_SIZE --temperatures $temperature; exec bash"
    
    echo "Screen session $session_name created"
}

# Function to create a screen session for an API model with a specific temperature
create_api_model_session() {
    local model=$1
    local temperature=$2
    local base_url=""
    local api_key=""
    local session_name="api_${model//[^a-zA-Z0-9]/_}_all_temperatures"
    
    echo "Creating API screen session $session_name for $model with temperature $temperature"
    
    # Determine the API base URL and key based on the model
    if [[ $model == *"deepseek"* ]]; then
        base_url="https://api.deepseek.com"
        api_key=$API_KEY_DEEPSEEK
    else
        base_url="https://api.chatanywhere.tech/v1"
        api_key=$API_KEY
    fi

    # Create detached screen session
    screen -dmS "$session_name" bash -c "cd /home/dazhou/ReasonEval && python $API_SCRIPT_PATH --api_key \"$api_key\" --base_url \"$base_url\" --dataset_name $DATASET --models \"$model\" --temperatures $temperature --subset_size $SUBSET_SIZE; exec bash"
    
    echo "API screen session $session_name created"
}

# Main execution
echo "Starting multiple GPU evaluation tasks"

# Launch local model screen sessions distributing across GPUs and temperatures
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
                create_local_model_session "$model" $gpu "$temp_group"
                sleep 1
            fi
        done
    done
fi

# Launch API model screen sessions with distributed temperatures
if [ "$RUN_API" = true ]; then
    echo "Setting up API model evaluation with distributed temperatures"
    for model in "${API_MODELS[@]}"; do
        create_api_model_session "$model" "${TEMPERATURES[*]}"
        
        # Add a small delay to avoid screen creation conflicts
        sleep 1
    done
fi

echo "All screen sessions have been created. Use 'screen -ls' to list active sessions."
echo "To attach to a session, use 'screen -r session_name'"
