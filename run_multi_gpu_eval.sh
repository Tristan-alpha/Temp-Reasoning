#!/bin/bash

# Configuration variables
SCRIPT_PATH="/home/dazhou/ReasonEval/t-codes/evaluate_results.py"
DATASET="hybrid_reasoning"
# TEMPERATURES=(1.5 1.6)
TEMPERATURES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5 1.6)
MODEL_SIZE="34B"  # Default model size

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

# Function to create a screen session for model evaluation
create_eval_session() {
    local model=$1
    local session_name="eval_${model//[^a-zA-Z0-9]/_}_auto_${MODEL_SIZE}"
    
    echo "Creating evaluation screen session $session_name for $model using automatic GPU allocation"
    
    # Create detached screen session with specified temperatures
    screen -dmS "$session_name" bash -c "cd /home/dazhou/ReasonEval && python $SCRIPT_PATH --models \"$model\" --dataset_name $DATASET --temperatures ${TEMPERATURES[*]} --model_size $MODEL_SIZE --reasoneval_path $REASONEVAL_PATH; exec bash"
    
    echo "Evaluation screen session $session_name created"
}

# Main execution
echo "Starting evaluation tasks with ReasonEval-$MODEL_SIZE using automatic GPU allocation"

for model in "${MODELS[@]}"; do
    create_eval_session "$model"
done

echo "All evaluation screen sessions have been created. Use 'screen -ls' to list active sessions."
echo "To attach to a session, use 'screen -r session_name'"
echo "Using ReasonEval-$MODEL_SIZE for all evaluations"