import json
import torch
import argparse
import os
import re
import sys
import numpy as np
import random
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from vllm import LLM, SamplingParams
from utils import wandb_logger

# Add ReasonEval path
reasoneval_path = "/home/data/dazhou/ReasonEval/t-codes"
sys.path.append(reasoneval_path)
from evaluate_results import (
    evaluate_solution_with_reasoneval, 
    evaluate_solution_with_math_shepherd,
    load_reasoneval_model
)


# Function to generate solutions in batch with vLLM
def generate_solutions_batch(model, model_name, questions, temperature, return_logprobs=False):
    """Generate solutions for multiple questions in batch using vLLM"""
    # Prepare prompts for batch processing
    prompts = []
    for question in questions:
        if model_name == "Abel-7B-002" or model_name == "WizardMath-7B-V1.1":
            prompt = f"Question: {question}\n\nProvide a step-by-step solution:"
        else:
            prompt = f"Solve math problems step-by-step. You MUST end each COMPLETE step with a double newline (\\n\\n). Question: {question}"
        prompts.append(prompt)
    
    # vLLM batch generation
    if temperature == 0:
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=2048,
            # use_beam_search=False,
            top_p=1.0,
            top_k=-1,
            skip_special_tokens=True,
            logprobs=10 if return_logprobs else None
        )
    else:
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=2048,
            top_p=1.0,
            top_k=-1,
            skip_special_tokens=True,
            logprobs=10 if return_logprobs else None
        )

    # Generate for all prompts at once
    outputs = model.generate(prompts, sampling_params)
    
    # Process outputs
    results = []
    for i, output in enumerate(outputs):
        solution = output.outputs[0].text
        
        # Extract logprobs if requested
        logprobs_info = None
        if return_logprobs and output.outputs[0].logprobs:
            logprobs_info = []
            for token_logprobs in output.outputs[0].logprobs:
                if token_logprobs:
                    token_probs = {}
                    for token_id, logprob in token_logprobs.items():
                        prob = np.exp(logprob.logprob)
                        token_probs[str(token_id)] = {
                            'token': logprob.decoded_token,
                            'logprob': logprob.logprob,
                            'prob': prob
                        }
                    logprobs_info.append(token_probs)
        
        # Format solution steps
        steps = format_solution_steps(solution, model_name)
        
        if return_logprobs:
            results.append((steps, logprobs_info))
        else:
            results.append(steps)
    
    return results

def format_solution_steps(solution, model_name):
    """Extract the solution formatting logic into a separate function"""
    steps = []
    step_num = 1

    if model_name == "Abel-7B-002" or model_name == "WizardMath-7B-V1.1":
        # Split by lines and process each line that looks like a step
        for line in solution.split('\n'):
            line = line.strip()
            if line:
                # Check if the line already starts with Step N:
                if re.match(r'^Step \d+:', line):
                    steps.append(line)
                else:
                    # Add step number
                    steps.append(f"Step {step_num}: {line}")
                    step_num += 1
    else:
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n|\\n\s*\\n', solution) if p.strip()]
        # If we have reasonable step divisions from triple newlines, use them
        if paragraphs and len(paragraphs) > 1:
            for i, paragraph in enumerate(paragraphs):
                steps.append(f"Step {i+1}: {paragraph}")
        else:
            # Last resort: If no step structure is detected, use double newlines as fallback
            paragraphs = [p.strip() for p in solution.split('\n') if p.strip()]
            for i, paragraph in enumerate(paragraphs):
                steps.append(f"Step {i+1}: {paragraph}")
    
    return steps

def dataset_extraction(item, dataset_name):

    if dataset_name == "hybrid_reasoning":
        source = item.get("source", "")
        if source == "math":
            uuid = item.get("unique_id", "unknown")
            question = item.get("problem", "")
        elif source == "mr-gsm8k":
            uuid = item.get("uuid", "unknown")
            question = item.get("question", "")
        return question, uuid, source
    
    elif dataset_name == "aime":
        source = dataset_name
        uuid = str(item['id'])
        question = item['problem']
        return question, uuid, source
    
    else:
        print("Invalid dataset.")
        return None, None

def load_json_data(file_path):
    dataset = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    obj = json.loads(line)
                    dataset.append(obj)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line as JSON: {line[:50]}...")
    return dataset


def main(args):
    # Clear CUDA cache at the start to ensure clean environment
    torch.cuda.empty_cache()
    
    input_path = args.input_path
    output_dir = args.output_dir
    dataset_name = args.dataset_name
    models = args.models
    temperatures = args.temperatures

    os.environ["WANDB_API_KEY"] = "KEY"
    os.environ["WANDB_MODE"] = 'offline'

    # Load the dataset
    if dataset_name == "aime":
        raw_dataset = load_dataset("AI-MO/aimo-validation-aime", split="train")
        dataset = [raw_dataset[i] for i in range(len(raw_dataset))]
    else:
        dataset_path = os.path.join(input_path, f"{dataset_name}.json")
        dataset = load_json_data(dataset_path)

    os.makedirs(output_dir, exist_ok=True)
    
    model_paths = {
        'Abel-7B-002': 'GAIR/Abel-7B-002',
        'WizardMath-7B-V1.1': 'WizardLMTeam/WizardMath-7B-V1.1',
        'Qwen3-0.6B': 'Qwen/Qwen3-0.6B',
        'Qwen3-4B': 'Qwen/Qwen3-4B',
        'Qwen3-8B': 'Qwen/Qwen3-8B',
        'Qwen3-32B': 'Qwen/Qwen3-32B',
        'Qwen3-14B': 'Qwen/Qwen3-14B',
        'Qwen3-30B-A3B': 'Qwen/Qwen3-30B-A3B'
    }

    # Validate and filter models
    models = [model for model in models if model in model_paths]
    if not models:
        print("Error: No valid models specified. Exiting.")
        return
   
    models = {model: model_paths[model] for model in models}

    if args.subset_size > 0 and args.subset_size < len(dataset):
        print(f"Using random subset of {args.subset_size} examples from the dataset")

        random.seed(42)
        dataset = random.sample(dataset, args.subset_size)
    
    # Load evaluation models if needed
    reasoneval_model = None
    reasoneval_tokenizer = None
    shepherd_model = None
    shepherd_tokenizer = None
    
    if args.enable_evaluation:
        print("Loading evaluation models...")
        try:
            reasoneval_model, reasoneval_tokenizer = load_reasoneval_model(args.reasoneval_path, args.reasoneval_model_size)
            
            print("Loading Math-Shepherd model...")
            shepherd_tokenizer = AutoTokenizer.from_pretrained(args.shepherd_path)
            shepherd_model = AutoModelForCausalLM.from_pretrained(
                args.shepherd_path,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            shepherd_model.eval()
        except Exception as e:
            print(f"Warning: Failed to load evaluation models: {e}")
            print("Continuing without evaluation...")
            args.enable_evaluation = False

    # Process each model
    for model_name, model_path in models.items():
        print(f"Loading model: {model_name} from {model_path}")
        
        model = None  # Initialize model variable
        logger = None  # Initialize logger variable
        
        try:
            # Initialize wandb logger for this model
            if args.logger:
                args.name = f"{model_name}-{dataset_name}"
                logger = wandb_logger(args)
            
            # Create model-specific directory
            model_output_dir = os.path.join(output_dir, model_name, dataset_name, "random_probs")
            os.makedirs(model_output_dir, exist_ok=True)
            
            # Load model with vLLM
            print(f"Using vLLM for optimized inference with {args.tensor_parallel_size} GPUs")
            model = load_model_with_vllm(
                model_path,
                dtype=args.vllm_dtype,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len
            )

            for temp in temperatures:
                print(f"Processing with temperature: {temp}")
                
                counter = 0
                results = []
                
                # Metrics for aggregation
                temp_validity_scores = []
                temp_redundancy_scores = []
                temp_shepherd_scores = []
                temp_avg_top1_probs = []
                temp_avg_top5_probs = []

                # Process in batches
                for i in tqdm(range(0, len(dataset), args.batch_size), desc="Processing batches"):
                    batch_items = dataset[i:i + args.batch_size]
            
                    batch_questions = []
                    batch_metadata = []
                    
                    for item in batch_items:
                        # print(f"Item type: {type(item)}")
                        # print(f"Item content: {item}")
                        # print(f"Dataset type: {type(dataset)}")
                        # print(f"Batch_items type: {type(batch_items)}")
                        question, uuid, source = dataset_extraction(item, dataset_name)
                        batch_questions.append(question)
                        batch_metadata.append((uuid, source))
                    
                    try:
                        # Generate solutions for the batch
                        if args.enable_evaluation and args.log_token_probs:
                            batch_results = generate_solutions_batch(
                                model, model_name, batch_questions, temp, return_logprobs=True
                            )
                        else:
                            batch_results = generate_solutions_batch(
                                model, model_name, batch_questions, temp, return_logprobs=False
                            )
                        
                        # Process each result in the batch
                        for j, batch_result in enumerate(batch_results):
                            uuid, source = batch_metadata[j]
                            question = batch_questions[j]
                            
                            if args.enable_evaluation and args.log_token_probs:
                                solution_steps, logprobs_info = batch_result
                            else:
                                solution_steps = batch_result
                                logprobs_info = None
                            
                            # Create result object
                            result = {
                                "uuid": uuid,
                                "question": question,
                                "source": source,
                                "model_output_steps": solution_steps
                            }
                            
                            # Evaluate solution if evaluation models are loaded
                            if args.enable_evaluation and reasoneval_model and shepherd_model:
                                try:
                                    # ReasonEval evaluation
                                    _, _, solution_validity, solution_redundancy = evaluate_solution_with_reasoneval(
                                        reasoneval_model, reasoneval_tokenizer, question, solution_steps
                                    )
                                    
                                    # Math-Shepherd evaluation
                                    _, solution_shepherd_score = evaluate_solution_with_math_shepherd(
                                        shepherd_model, shepherd_tokenizer, question, solution_steps
                                    )
                                    
                                    # Add evaluation scores to result
                                    result["validity_score"] = solution_validity
                                    result["redundancy_score"] = solution_redundancy
                                    result["shepherd_score"] = solution_shepherd_score
                                    
                                    # Collect scores for aggregation
                                    temp_validity_scores.append(solution_validity)
                                    temp_redundancy_scores.append(solution_redundancy)
                                    temp_shepherd_scores.append(solution_shepherd_score)
                                    
                                    # Add token probability metrics if available
                                    if args.log_token_probs and logprobs_info:
                                        # Calculate average top-1 and top-5 probabilities
                                        top1_probs = []
                                        top5_probs = []
                                        
                                        for token_probs in logprobs_info:
                                            if token_probs:
                                                sorted_probs = sorted(token_probs.values(), 
                                                                    key=lambda x: x['prob'], reverse=True)
                                                if sorted_probs:
                                                    top1_probs.append(sorted_probs[0]['prob'])
                                                    top5_prob_sum = sum([p['prob'] for p in sorted_probs[:5]])
                                                    top5_probs.append(top5_prob_sum)
                                        
                                        if top1_probs:
                                            avg_top1_prob = np.mean(top1_probs)
                                            avg_top5_prob = np.mean(top5_probs)
                                            temp_avg_top1_probs.append(avg_top1_prob)
                                            temp_avg_top5_probs.append(avg_top5_prob)
                                        
                                       
                                        
                                except Exception as e:
                                    print(f"Error evaluating solution {uuid}: {str(e)}")
                            
                            results.append(result)
                            counter += 1
                            
                    except Exception as e:
                        print(f"Error processing batch starting at index {i}: {str(e)}")
                
                # Log aggregated metrics for this temperature
                if logger and temp_validity_scores:
                    agg_metrics = {
                        "avg_validity": np.mean(temp_validity_scores),
                        "avg_redundancy": np.mean(temp_redundancy_scores),
                        "avg_shepherd": np.mean(temp_shepherd_scores),
                        "num_samples": len(temp_validity_scores)
                    }
                    
                    if temp_avg_top1_probs:
                        agg_metrics["avg_top1_prob"] = np.mean(temp_avg_top1_probs)
                        agg_metrics["avg_top5_prob"] = np.mean(temp_avg_top5_probs)
                    
                    # Log with temperature as x-axis
                    logger.log_temperature_metrics(agg_metrics, temp)
                    
                    print(f"Temperature {temp} - Avg Validity: {np.mean(temp_validity_scores):.4f}, "
                          f"Avg Redundancy: {np.mean(temp_redundancy_scores):.4f}, "
                          f"Avg Shepherd: {np.mean(temp_shepherd_scores):.4f}")
                
                # Save results to file
                output_path = os.path.join(model_output_dir, f"temperature_{temp}.json")
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
                
                print(f"Results saved to {output_path}")
        
        except Exception as e:
            print(f"Error processing model {model_name}: {str(e)}")
            print("Continuing to next model...")
        
        finally:
            if logger:
                logger.finish()
            
            if model is not None:
                try:

                    del model
                    # Force garbage collection
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()

                except Exception as cleanup_error:
                    print(f"Warning: Error during model cleanup for {model_name}: {cleanup_error}")

# Function to load a model with vLLM
def load_model_with_vllm(model_path, dtype='half', tensor_parallel_size=None, gpu_memory_utilization=0.85, max_model_len=None):
    """Load a model with vLLM for optimized inference across multiple GPUs"""
    try:
        torch.cuda.empty_cache()
        
        model = LLM(
            model=model_path,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,  # Number of GPUs to use for tensor parallelism
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=True
        )
        
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        torch.cuda.empty_cache()
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate solutions with different temperature settings')
    parser.add_argument("--input_path", type=str, default="/home/data/dazhou/ReasonEval/dataset",
                        help="Path to the input dataset directory")
    parser.add_argument("--dataset_name", type=str, default="hybrid_reasoning",
                        help="Name of the dataset being processed (defaults to hybrid_reasoning)")
    parser.add_argument("--output_dir", type=str, default="/home/data/dazhou/ReasonEval/Results",
                        help="Directory to save the results")
    parser.add_argument("--models", type=str, nargs='+', 
                        default=['Abel-7B-002', 'WizardMath-7B-V1.1'],
                        help="List of models to evaluate. Choices: Abel-7B-002, WizardMath-7B-V1.1")
    parser.add_argument("--temperatures", type=float, nargs='+',
                        default=[0.1, 0.3, 0.6, 1.0, 1.3, 1.6, 2.0],
                        help="List of temperature values to test")
    parser.add_argument("--subset_size", type=int, default=0,
                        help="Number of examples to process (0 = all)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for vLLM inference")
    # vLLM configuration
    parser.add_argument("--tensor_parallel_size", type=int, default=None,
                        help="Number of GPUs to use for tensor parallelism (None = auto)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="Fraction of GPU memory to use (0.0 to 1.0)")
    parser.add_argument("--vllm_dtype", type=str, default="half", 
                        choices=["half", "float16", "bfloat16", "float"],
                        help="Data type for vLLM inference")
    parser.add_argument("--max_model_len", type=int, default=8192,
                        help="Maximum sequence length for the model")
    
    # WandB logging arguments
    parser.add_argument('--logger', action='store_true', default=True, 
                        help='Enable WandB logging')
    parser.add_argument('--entity', type=str, default="dazhou_liu2023-southern-university-of-science-technology", 
                        help='WandB entity name')
    parser.add_argument('--project', type=str, default="Temp-Reasoning", 
                        help='WandB project name')
    parser.add_argument('--name', type=str, default="model", 
                        help='Experiment name')
    
    # Evaluation arguments
    parser.add_argument('--enable_evaluation', action='store_true', default=False,
                        help='Enable real-time evaluation with ReasonEval and Math-Shepherd')
    parser.add_argument('--reasoneval_path', type=str, default="GAIR/ReasonEval-7B",
                        help='Path to ReasonEval model')
    parser.add_argument('--reasoneval_model_size', type=str, choices=['7B', '34B'], default='7B',
                        help='Size of ReasonEval model to use')
    parser.add_argument('--shepherd_path', type=str, default="peiyi9979/math-shepherd-mistral-7b-prm",
                        help='Path to Math-Shepherd model')
    parser.add_argument('--log_token_probs', action='store_true', default=False,
                        help='Log top-k token probabilities during generation')

    args = parser.parse_args()
    main(args)
