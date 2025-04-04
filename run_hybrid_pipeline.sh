#!/bin/bash

# Set up environment (assuming requirements are already installed)
echo "Starting hybrid reasoning evaluation pipeline..."

# Step 1: Generate the hybrid dataset if it doesn't exist
if [ ! -f "/home/dazhou/ReasonEval/dataset/hybrid_reasoning.json" ]; then
  echo "Creating hybrid dataset..."
  python /home/dazhou/ReasonEval/dataset/create_hybrid_dataset.py
else
  echo "Hybrid dataset already exists, skipping creation step."
fi

# Step 2: Generate model answers for the hybrid dataset
# Using a subset of 100 examples for faster execution (remove --subset_size for full dataset)
echo "Generating model answers using automatic GPU allocation..."
python /home/dazhou/ReasonEval/t-codes/answer_generation.py \
  --dataset_name hybrid_reasoning \
  --models Abel-7B-002 WizardMath-7B-V1.1 \
  --temperatures 0.1 0.3 0.5 0.7 1.0 1.3 \
  --gpu auto

# Step 3: Evaluate model answers
echo "Evaluating model answers using automatic GPU allocation..."
python /home/dazhou/ReasonEval/t-codes/evaluate_results.py \
  --dataset_name hybrid_reasoning \
  --models Abel-7B-002 WizardMath-7B-V1.1 \
  --temperatures 0.1 0.3 0.5 0.7 1.0 1.3 \
  --gpu auto

echo "Pipeline completed! Results available in /home/dazhou/ReasonEval/evaluation_results/"