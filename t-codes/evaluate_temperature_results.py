import os
import json
import sys
import torch
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


reasoneval_path = "/home/dazhou/ReasonEval/codes"
sys.path.append(reasoneval_path)
# Import ReasonEval model classes from the examples
from examples import ReasonEval_7B

def load_temperature_results(results_dir, model_name, temperature):
    """Load the results for a specific model and temperature"""
    filename = os.path.join(results_dir, f"{model_name}_temperature_{temperature}.json")
    with open(filename, 'r') as f:
        results = json.load(f)
    return results

def evaluate_solution_with_reasoneval(model, tokenizer, question, reasoning_steps, device):
    """Evaluate reasoning steps with ReasonEval model"""
    PROMPT_FORMAT = "Question:\n{input}\nAnswer:\nLet's think step by step.\n"
    step_separator = f"{tokenizer.pad_token}"
    combined_steps = ""
    
    # Filter out empty steps
    valid_steps = [step for step in reasoning_steps if step and not step.isspace()]
    if not valid_steps:
        return [], [], 0, 0  # Return empty scores if no valid steps
        
    for step in valid_steps:
        combined_steps += step + step_separator
    
    prompt = PROMPT_FORMAT.format(input=question)
    tokenized_result = tokenizer(prompt + step_separator + combined_steps)['input_ids']

    # Separating labels and adjusting token IDs
    separator_token_id = tokenizer(step_separator)['input_ids'][-1]
    labeled_token_indices = []
    adjusted_token_ids = []
    separator_count = 0
    
    for idx, token_id in enumerate(tokenized_result):
        if token_id == separator_token_id:
            labeled_token_indices.append(idx - 1 - separator_count)
            separator_count += 1
        else:
            adjusted_token_ids.append(token_id)
    
    # Adjusting for ReasonEval-7B specifically
    adjusted_token_ids = [1] + adjusted_token_ids  # Adjusting to recover the first token_ids
    adjusted_token_ids = torch.tensor([adjusted_token_ids]).to(device)
    
    # Skip the first two separators (beginning and ending of the problem)
    if len(labeled_token_indices) > 2:
        labeled_token_indices = labeled_token_indices[2:]
    else:
        return [], [], 0, 0  # Not enough steps to evaluate
    
    # Make sure we have as many indices as valid steps
    if len(labeled_token_indices) != len(valid_steps):
        labeled_token_indices = labeled_token_indices[:len(valid_steps)]
    
    attention_mask = adjusted_token_ids.new_ones(adjusted_token_ids.size(), dtype=torch.bool)
    
    # Evaluating reasoning steps using ReasonEval
    with torch.no_grad():
        reasoning_scores = model(adjusted_token_ids, attention_mask)[0, labeled_token_indices, :]
        scores = torch.softmax(reasoning_scores, dim=-1).tolist()
    
    # Calculating the validity and redundancy scores
    ## score: [p_{negative}, p_{neutral}, p_{positive}]
    
    ## S_{validity} = p_{neutral} + p_{positive}
    step_level_validity_scores = [(score[1] + score[2]) for score in scores]
    
    ## S_{redundancy} = p_{neutral}
    step_level_redundancy_scores = [score[1] for score in scores]
    
    solution_level_validity_scores = min(step_level_validity_scores) if step_level_validity_scores else 0
    solution_level_redundancy_scores = max(step_level_redundancy_scores) if step_level_redundancy_scores else 0
    
    return step_level_validity_scores, step_level_redundancy_scores, solution_level_validity_scores, solution_level_redundancy_scores

def evaluate_solution_with_math_shepherd(model, tokenizer, question, reasoning_steps, device):
    """Evaluate correctness of the solution with math-shepherd model"""
    # Define special tokens
    step_tag = "ки"
    good_token = "+"
    bad_token = "-"
    
    # Get token IDs
    candidate_tokens = tokenizer.encode(f"{good_token} {bad_token}")[1:]  # [+, -]
    step_tag_id = tokenizer.encode(f"{step_tag}")[-1]
    
    # The steps already have step labels, so we just need to add the special tag
    formatted_steps = []
    for step in reasoning_steps:
        if step and not step.isspace():
            # Add the step tag at the end of each existing step
            formatted_steps.append(f"{step} {step_tag}")
    
    # Combine the question with formatted steps
    formatted_output = "\n".join(formatted_steps)
    input_for_prm = f"{question} {formatted_output}"
    
    # Convert to tensor and move to device
    input_ids = torch.tensor([tokenizer.encode(input_for_prm)]).to(device)
    
    # Get model prediction (logits)
    with torch.no_grad():
        logits = model(input_ids).logits[:, :, candidate_tokens]
        scores = logits.softmax(dim=-1)[:, :, 0]  # Probability for the "+" token
        
        # Extract scores at positions where step_tag_id appears
        step_tag_positions = (input_ids == step_tag_id).nonzero(as_tuple=True)[1]
        
        # If no step tags were found
        if len(step_tag_positions) == 0:
            return 0.0  # Return a default score
            
        # Get scores at step tag positions
        step_scores = scores[0, step_tag_positions]

        solution_scores = min(step_scores).item()
        
        # Return the last step score as the final correctness indicator
        # The example shows correct answers have high scores (~0.99)
        # and incorrect answers have low scores (~0.02)
        return step_scores, solution_scores  # Convert last tensor element to Python scalar

def main(args):

    # Set CUDA device   
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    # Path to results from temperature study
    results_dir = args.results_dir
    detailed_dir = os.path.join(args.output_dir, 'detailed_results')
    aggregated_dir = os.path.join(args.output_dir, 'aggregated_results')    
    os.makedirs(detailed_dir, exist_ok=True)
    os.makedirs(aggregated_dir, exist_ok=True)
    
    # Get models to evaluate from command line arguments
    models_to_evaluate = args.models
    temperatures = args.temperatures
    
    # Define the model paths
    model_paths = {
        'Abel-7B-002': 'GAIR/Abel-7B-002',
        'WizardMath-7B-V1.1': 'WizardLMTeam/WizardMath-7B-V1.1'
    }
    
    # Validate requested models
    for model in models_to_evaluate:
        if model not in model_paths:
            print(f"Warning: Model '{model}' is not recognized. Available models: {', '.join(model_paths.keys())}")
    
    # Filter to only include valid models
    models_to_evaluate = [model for model in models_to_evaluate if model in model_paths]
    
    if not models_to_evaluate:
        print("Error: No valid models specified. Exiting.")
        return
    
    print(f"Evaluating models: {', '.join(models_to_evaluate)}")
    
    # Load ReasonEval model
    print("Loading ReasonEval model...")
    reasoneval_path = args.reasoneval_path
    reasoneval_tokenizer = AutoTokenizer.from_pretrained(reasoneval_path)
    reasoneval_model = ReasonEval_7B.from_pretrained(reasoneval_path).to(device)
    reasoneval_model.eval()
    
    # Load Math-Shepherd model
    print("Loading Math-Shepherd model...")
    shepherd_path = args.shepherd_path
    shepherd_tokenizer = AutoTokenizer.from_pretrained(shepherd_path)
    shepherd_model = AutoModelForCausalLM.from_pretrained(
        shepherd_path,
        torch_dtype=torch.float16,
        device_map={"": device}
    ).to(device)
    shepherd_model.eval()
    
    # Create empty DataFrame to store results
    results_df = pd.DataFrame(columns=[
        'Model', 'Temperature', 'Question_UUID', 
        'Solution_Level_Validity', 'Solution_Level_Redundancy',
        'Solution_Score_Shepherd'
    ])
    
    # Process each model
    for model_name in models_to_evaluate:
        print(f"Evaluating {model_name}...")
        
        for temp in temperatures:
            print(f"  Temperature: {temp}")
            try:
                results = load_temperature_results(results_dir, model_name, temp)
            except FileNotFoundError:
                print(f"  Results file not found for {model_name} at temperature {temp}. Skipping.")
                continue
            
            # Process each solution
            for item in tqdm(results):
                uuid = item.get("uuid", "unknown")
                question = item.get("question", "")
                steps = item.get("model_output_steps", [])
                
                if not steps or not question:
                    continue
                
                try:
                    # Evaluate with ReasonEval
                    _, _, solution_validity, solution_redundancy = evaluate_solution_with_reasoneval(
                        reasoneval_model, reasoneval_tokenizer, question, steps, device
                    )
                    
                    # Evaluate correctness with Math-Shepherd
                    _, solution_shepherd_score = evaluate_solution_with_math_shepherd(
                        shepherd_model, shepherd_tokenizer, question, steps, device
                    )
                    
                    # Add to DataFrame
                    new_row = {
                        'Model': model_name,
                        'Temperature': temp,
                        'Question_UUID': uuid,
                        'Solution_Level_Validity': solution_validity,
                        'Solution_Level_Redundancy': solution_redundancy,
                        'Solution_Score_Shepherd': solution_shepherd_score  # Now correctly a scalar value
                    }
                    
                    results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
                    
                except Exception as e:
                    print(f"Error evaluating solution {uuid}: {str(e)}")
    
    # Skip saving if no results
    if results_df.empty:
        print("No results to save. Exiting.")
        return
    
    # Save detailed results
    results_df.to_csv(os.path.join(args.output_dir, f'{model_name}_detailed.csv'), index=False)
    
    # Aggregate results by model and temperature
    agg_results = results_df.groupby(['Model', 'Temperature']).agg({
        'Solution_Level_Validity': 'mean',
        'Solution_Level_Redundancy': 'mean',
        'Solution_Score_Shepherd': 'mean'  # Percentage of correct answers
    }).reset_index()
    
    agg_results.to_csv(os.path.join(args.output_dir, f'{model_name}_aggregated.csv'), index=False)
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Validity Score
    plt.subplot(2, 2, 1)
    sns.lineplot(data=agg_results, x='Temperature', y='Solution_Level_Validity', hue='Model', marker='o')
    plt.title('Average Validity Score vs Temperature')
    plt.grid(True)
    
    # Redundancy Score
    plt.subplot(2, 2, 2)
    sns.lineplot(data=agg_results, x='Temperature', y='Solution_Level_Redundancy', hue='Model', marker='o')
    plt.title('Average Redundancy Score vs Temperature')
    plt.grid(True)
    
    # Correctness
    plt.subplot(2, 2, 3)
    sns.lineplot(data=agg_results, x='Temperature', y='Solution_Score_Shepherd', hue='Model', marker='o')
    plt.title('Answer Correctness vs Temperature')
    plt.grid(True)
    
    # Combined Bar Chart
    plt.subplot(2, 2, 4)
    bar_data = agg_results.melt(id_vars=['Model', 'Temperature'], 
                               value_vars=['Solution_Level_Validity', 'Solution_Level_Redundancy', 'Solution_Score_Shepherd'],
                               var_name='Metric', value_name='Score')
    sns.barplot(data=bar_data, x='Temperature', y='Score', hue='Metric', col='Model')
    plt.title('Metrics Comparison Across Temperatures')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'temperature_evaluation_plots.png'))
    plt.close()
    
    # Create separate plots for each model for better clarity
    for model in models_to_evaluate:
        model_data = agg_results[agg_results['Model'] == model]
        
        plt.figure(figsize=(15, 5))
        
        # Plot all metrics for this model
        plt.subplot(1, 2, 1)
        model_melted = model_data.melt(id_vars=['Temperature'], 
                                     value_vars=['Solution_Level_Validity', 'Solution_Level_Redundancy', 'Solution_Score_Shepherd'],
                                     var_name='Metric', value_name='Score')
        sns.lineplot(data=model_melted, x='Temperature', y='Score', hue='Metric', marker='o')
        plt.title(f'{model} - Metrics vs Temperature')
        plt.grid(True)
        
        # Bar chart
        plt.subplot(1, 2, 2)
        sns.barplot(data=model_melted, x='Temperature', y='Score', hue='Metric')
        plt.title(f'{model} - Metrics Comparison')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f'temperature_evaluation_{model}.png'))
        plt.close()
        
    print("Evaluation complete. Results and visualizations saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="/home/dazhou/ReasonEval/temperature_study")
    parser.add_argument("--output_dir", type=str, default="/home/dazhou/ReasonEval/evaluation_results")
    parser.add_argument("--reasoneval_path", type=str, default="GAIR/ReasonEval-7B")
    parser.add_argument("--shepherd_path", type=str, default="peiyi9979/math-shepherd-mistral-7b-prm")
    parser.add_argument("--model_size", type=str, choices=['7B', '34B'], default='7B')
    parser.add_argument("--models", type=str, nargs='+', 
                        default=['Abel-7B-002', 'WizardMath-7B-V1.1'],
                        help="List of models to evaluate. Choices: Abel-7B-002, WizardMath-7B-V1.1")
    parser.add_argument("--temperatures", type=float, nargs='+',
                        default=[0.1, 0.3, 0.6, 1.0, 1.3, 1.6, 2.0],
                        help="List of temperature values to test")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
