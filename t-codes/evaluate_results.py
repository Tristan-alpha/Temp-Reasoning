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
from transformers import MistralModel, MistralPreTrainedModel, LlamaModel, LlamaPreTrainedModel, AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
import torch
import torch.nn as nn


reasoneval_path = "/home/dazhou/ReasonEval/codes"
sys.path.append(reasoneval_path)
# Import ReasonEval model classes from the examples

class ReasonEval_34B(LlamaPreTrainedModel):
    _keys_to_ignore_on_load_missing = ['lm_head.weight']

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)
        self.model = LlamaModel(config)
        self.score_head = nn.Linear(config.hidden_size, config.score_dim, bias=config.bias)
        self.post_init() # Initialize weights and apply final processing

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> torch.Tensor:
      
        assert attention_mask is not None
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # (batch_size, sequence_length, dim)
        scores = self.score_head(hidden_states)  # (batch_size, sequence_length, class)
        return scores

class ReasonEval_7B(MistralPreTrainedModel):
    _keys_to_ignore_on_load_missing = ['lm_head.weight']

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)
        self.model = MistralModel(config)
        self.score_head = nn.Linear(config.hidden_size, config.score_dimension, bias=config.use_bias)
        self.post_init()  # Initialize weights and apply final processing

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> torch.Tensor:
        assert attention_mask is not None
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]  # (batch_size, sequence_length, dim)
        scores = self.score_head(hidden_states)  # (batch_size, sequence_length, class)
        return scores
    

def load_temperature_results(results_dir, model_name, dataset_name, temperature):
    """Load the results for a specific model and temperature"""
    # Updated to handle the new directory structure
    filename = os.path.join(results_dir, model_name, dataset_name, f"temperature_{temperature}.json")
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

def load_reasoneval_model(model_path, model_size, device):
    """Load the appropriate ReasonEval model based on model size"""
    print(f"Loading ReasonEval-{model_size} model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if model_size == '34B':
        model = ReasonEval_34B.from_pretrained(model_path).to(device)
    else:  # Default to 7B
        model = ReasonEval_7B.from_pretrained(model_path).to(device)
    
    model.eval()
    return model, tokenizer

def main(args):

    # Set CUDA device   
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    # Path to results from temperature study
    results_dir = args.results_dir
    dataset_name = args.dataset_name
    # detailed_dir = os.path.join(args.output_dir, 'detailed_results')
    # aggregated_dir = os.path.join(args.output_dir, 'aggregated_results')    
    # os.makedirs(detailed_dir, exist_ok=True)
    # os.makedirs(aggregated_dir, exist_ok=True)
    
    # Get models to evaluate from command line arguments
    models_to_evaluate = args.models
    temperatures = args.temperatures
    
    # Define the model paths
    model_paths = {
        'Abel-7B-002': 'GAIR/Abel-7B-002',
        'WizardMath-7B-V1.1': 'WizardLMTeam/WizardMath-7B-V1.1',
        'o1': 'o1',
        'o3-mini': 'o3-mini',
        'deepseek-v3': 'deepseek-v3',
        'deepseek-r1': 'deepseek-r1',
        'claude-3-7-sonnet-20250219': 'claude-3-7-sonnet-20250219',
        'gemini-2.0-flash': 'gemini-2.0-flash',
        'grok-3-reasoner': 'grok-3-reasoner'
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
    
    # Load ReasonEval model with selected size
    reasoneval_path = args.reasoneval_path
    reasoneval_model, reasoneval_tokenizer = load_reasoneval_model(reasoneval_path, args.model_size, device)
    
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
    
    # Dictionary to store results for all models
    all_model_results = {}
    
    # Process each model
    for model_name in models_to_evaluate:
        print(f"Evaluating {model_name}...")
        
        # Check if detailed CSV already exists and load it
        detailed_csv_path = os.path.join(args.output_dir, model_name, dataset_name, 'detailed.csv')
        if os.path.exists(detailed_csv_path):
            print(f"Loading existing detailed results for {model_name} on {dataset_name}...")
            results_df = pd.read_csv(detailed_csv_path)
            # Get list of already processed temperature-uuid combinations to avoid duplicates
            processed_temp_uuids = set()
            for _, row in results_df.iterrows():
                processed_temp_uuids.add((row['Temperature'], row['Question_UUID']))
        else:
            # Create empty DataFrame if no existing file
            results_df = pd.DataFrame(columns=[
                'Model', 'Dataset', 'Temperature', 'Question_UUID', 'Source',
                'Solution_Level_Validity', 'Solution_Level_Redundancy',
                'Solution_Score_Shepherd'
            ])
            processed_temp_uuids = set()
        
        for temp in temperatures:
            print(f"  Temperature: {temp}")
            try:
                results = load_temperature_results(results_dir, model_name, dataset_name, temp)
            except FileNotFoundError:
                print(f"  Results file not found for {model_name} on {dataset_name} at temperature {temp}. Skipping.")
                continue
            
            # Process each solution
            for item in tqdm(results):
                uuid = item.get("uuid", "unknown")
                
                # Skip if this temperature-uuid combination has already been processed
                if (temp, uuid) in processed_temp_uuids:
                    continue
                
                question = item.get("question", "")
                steps = item.get("model_output_steps", [])
                source = item.get("source", "unknown")
                
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
                        'Dataset': dataset_name,
                        'Temperature': temp,
                        'Question_UUID': uuid,
                        'Source': source,
                        'Solution_Level_Validity': solution_validity,
                        'Solution_Level_Redundancy': solution_redundancy,
                        'Solution_Score_Shepherd': solution_shepherd_score
                    }
                    
                    results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
                    
                except Exception as e:
                    print(f"Error evaluating solution {uuid}: {str(e)}")
        
        # Save detailed results for this model
        os.makedirs(os.path.dirname(detailed_csv_path), exist_ok=True)
        results_df.to_csv(detailed_csv_path, index=False)
        print(f"Saved detailed results for {model_name} on {dataset_name}")
        
        # Store in our dictionary for aggregation later
        all_model_results[model_name] = results_df
    
    
    # Aggregate results by model and temperature
    for model_name, model_df in all_model_results.items():
        if not model_df.empty:
            # Calculate overall aggregate metrics
            agg_results = model_df.groupby(['Model', 'Dataset', 'Temperature']).agg({
                'Solution_Level_Validity': 'mean',
                'Solution_Level_Redundancy': 'mean',
                'Solution_Score_Shepherd': 'mean'
            }).reset_index()
            
            # Save overall aggregated results
            agg_csv_path = os.path.join(args.output_dir, model_name, dataset_name, "aggregated.csv")
            agg_results.to_csv(agg_csv_path, index=False)
            print(f"Saved aggregated results for {model_name} on {dataset_name}")
            
            # Calculate source-specific metrics
            source_agg_results = model_df.groupby(['Model', 'Dataset', 'Temperature', 'Source']).agg({
                'Solution_Level_Validity': 'mean',
                'Solution_Level_Redundancy': 'mean',
                'Solution_Score_Shepherd': 'mean'
            }).reset_index()
            
            # Save source-specific aggregated results
            source_agg_csv_path = os.path.join(args.output_dir, model_name, dataset_name, "source_aggregated.csv")
            source_agg_results.to_csv(source_agg_csv_path, index=False)
            print(f"Saved source-specific aggregated results for {model_name} on {dataset_name}")
    
    print("Evaluation complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="/home/dazhou/ReasonEval/answer_by_models")
    parser.add_argument("--output_dir", type=str, default="/home/dazhou/ReasonEval/evaluation_results")
    parser.add_argument("--dataset_name", type=str, default="hybrid_reasoning",
                        help="Name of the dataset being evaluated")
    parser.add_argument("--reasoneval_path", type=str, default="GAIR/ReasonEval-7B")
    parser.add_argument("--shepherd_path", type=str, default="peiyi9979/math-shepherd-mistral-7b-prm")
    parser.add_argument("--model_size", type=str, choices=['7B', '34B'], default='7B',
                        help="Size of the ReasonEval model to use (7B or 34B)")
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
