#!/bin/bash
# Configuration variables
SCRIPT_PATH="/home/dazhou/ReasonEval/t-codes/answer_generation.py"
API_SCRIPT_PATH="/home/dazhou/ReasonEval/t-codes/api_answer_generation.py"
DATASET="aime"
SUBSET_SIZE=0

# TEMPERATURES=(1.5 1.6)
TEMPERATURES=(0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6 0.0 0.1 0.2 0.3 0.4)
# TEMPERATURES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1)
# TEMPERATURES=(0.1 0.4 0.5 0.7 0.8 0.9 1.1 1.2 1.4 1.5)
# TEMPERATURES=(0.0 0.2 0.3 0.6 1.0 1.3 1.6)

# API Keys
API_KEY=sk-dqbCjalqSxgKaqe4YyNGGByaNLFk6vv0gXp0LnErebFmTZkx
API_KEY_DEEPSEEK=sk-a00c3e30d7524683883aa59f82191ae4

# Run settings
RUN_LOCAL=true  # Whether to run local models
RUN_API=false   # Whether to run API models

# Local Models to evaluate (add more models as needed)
LOCAL_MODELS=(
    "WizardMath-7B-V1.1" 
    "Abel-7B-002"
    'Qwen3-0.6B'
    'Qwen3-4B'
    'Qwen3-8B'
    'Qwen3-32B'
)

# API Models to evaluate
API_MODELS=(
    # "gpt-4o-mini"
    "deepseek-chat"
    # "deepseek-reasoner"
    # "deepseek-v3"
    # "deepseek-r1"
    # "claude-3-7-sonnet-20250219"
    # "gemini-2.0-flash"
)

# # Ensure the hybrid dataset exists
# if [ ! -f "/home/dazhou/ReasonEval/dataset/hybrid_reasoning.json" ]; then
#   echo "Creating hybrid dataset..."
#   python /home/dazhou/ReasonEval/dataset/create_hybrid_dataset.py
# else
#   echo "Hybrid dataset already exists, skipping creation step."
# fi

# Function to create a screen session for a local model with automatic GPU allocation
create_local_model_session() {
    local model=$1
    local session_name="${model//[^a-zA-Z0-9]}"
    
    echo "Creating screen session $session_name for $model with automatic GPU allocation"
    
    # Create detached screen session
    screen -dmS "$session_name" bash -c "cd /home/dazhou/ReasonEval && python $SCRIPT_PATH --models \"$model\" --dataset_name $DATASET --subset_size $SUBSET_SIZE --temperatures ${TEMPERATURES[@]}; exec bash"
    
    echo "Screen session $session_name created"
}

# Function to create a screen session for an API model
create_api_model_session() {
    local model=$1
    local base_url=""
    local api_key=""
    local session_name="api_${model//[^a-zA-Z0-9]/_}_all_temperatures"
    
    echo "Creating API screen session $session_name for $model with temperatures ${TEMPERATURES[*]}"
    
    # Determine the API base URL and key based on the model
    if [[ $model == *"deepseek"* ]]; then
        base_url="https://api.deepseek.com"
        api_key=$API_KEY_DEEPSEEK
    else
        base_url="https://api.chatanywhere.tech/v1"
        api_key=$API_KEY
    fi
    # Create detached screen session
    screen -dmS "$session_name" bash -c "cd /home/dazhou/ReasonEval && python $API_SCRIPT_PATH --api_key \"$api_key\" --base_url \"$base_url\" --dataset_name $DATASET --models \"$model\" --temperatures ${TEMPERATURES[*]} --subset_size $SUBSET_SIZE; exec bash"
    
    echo "API screen session $session_name created"
}

# Main execution
echo "Starting model generation tasks with automatic GPU allocation"

# Launch local model screen sessions using automatic GPU allocation
if [ "$RUN_LOCAL" = true ]; then
    echo "Setting up local model generation with automatic GPU allocation"
    
    # For each local model
    for model in "${LOCAL_MODELS[@]}"; do
        create_local_model_session "$model"
        sleep 1
    done
fi

# Launch API model screen sessions
if [ "$RUN_API" = true ]; then
    echo "Setting up API model generation"
    for model in "${API_MODELS[@]}"; do
        create_api_model_session "$model"
        
        # Add a small delay to avoid screen creation conflicts
        sleep 1
    done
fi

echo "All screen sessions have been created. Use 'screen -ls' to list active sessions."
echo "To attach to a session, use 'screen -r session_name'"
