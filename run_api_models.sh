#!/bin/bash

API_KEY=sk-dqbCjalqSxgKaqe4YyNGGByaNLFk6vv0gXp0LnErebFmTZkx

# # deepseek API
API_KEY_DEEPSEEK=sk-a00c3e30d7524683883aa59f82191ae4

# Set default subset size or use provided value
SUBSET_SIZE=10

# Define the advanced models we want to test
MODELS="gpt-4o-mini,o1,o3-mini,deepseek-v3,deepseek-r1,claude-3-7-sonnet-20250219,gemini-2.0-flash,grok-3-reasoner"


echo "Starting advanced models evaluation pipeline..."

# Step 1: Ensure the hybrid dataset exists
if [ ! -f "/home/dazhou/ReasonEval/dataset/hybrid_reasoning.json" ]; then
  echo "Creating hybrid dataset..."
  python /home/dazhou/ReasonEval/dataset/create_hybrid_dataset.py
else
  echo "Hybrid dataset already exists, skipping creation step."
fi

# Step 2: Generate model answers using ChatAnywhere API
echo "Generating model answers with ChatAnywhere API..."
echo "Using models: $MODELS"
# python /home/dazhou/ReasonEval/t-codes/api_answer_generation.py \
#   --api_key "$API_KEY_DEEPSEEK" \
#   --base_url "https://api.deepseek.com" \
#   --dataset_name hybrid_reasoning \
#   --models "deepseek-reasoner" \
#   --temperatures 0.0 \
#   --subset_size $SUBSET_SIZE \
#   --start_index 78
#   # --models $(echo $MODELS | tr ',' ' ') \
#   # --base_url "https://api.chatanywhere.tech/v1" \

  python /home/dazhou/ReasonEval/t-codes/api_answer_generation.py \
  --api_key "$API_KEY" \
  --base_url "https://api.chatanywhere.tech/v1" \
  --dataset_name hybrid_reasoning \
  --models "gpt-4o-mini" \
  --temperatures 0.0 \
  --subset_size $SUBSET_SIZE \
  --start_index 75
  # --models $(echo $MODELS | tr ',' ' ') \


# # Step 3: Evaluate model answers
# echo "Evaluating model answers..."
# python /home/dazhou/ReasonEval/t-codes/evaluate_results.py \
#   --dataset_name hybrid_reasoning \
#   --models $(echo $MODELS | tr ',' ' ') \
#   --temperatures 0.1

echo "Pipeline completed! Results available in /home/dazhou/ReasonEval/evaluation_results/"