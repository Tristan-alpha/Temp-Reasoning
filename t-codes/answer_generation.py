import json
import torch
import argparse
import os
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
# Add vLLM imports
from vllm import LLM, SamplingParams

# Function to generate solution with a model
def generate_solution(model, model_name, tokenizer, question, temperature, use_vllm=False):
    if model_name == "Abel-7B-002" or model_name == "WizardMath-7B-V1.1":
        prompt = f"Question: {question}\n\nProvide a step-by-step solution:"
    else:
        prompt = f"Solve math problems step-by-step. You MUST end each COMPLETE step with a double newline (\\n\\n). Question: {question}"

    if use_vllm:
        # vLLM generation
        if temperature == 0:
            # Greedy decoding
            sampling_params = SamplingParams(
                temperature=1.0,  # any value, will be ignored in greedy mode
                max_tokens=4096,
                use_beam_search=False,
                top_p=1.0,
                top_k=-1,
                skip_special_tokens=True
            )
        else:
            # Sampling
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=4096,
                top_p=0.95,
                top_k=20,
                skip_special_tokens=True
            )
        
        outputs = model.generate([prompt], sampling_params)
        solution = outputs[0].outputs[0].text
    else:
        # Original HuggingFace generation
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            if temperature == 0:
                # 使用纯贪婪解码
                outputs = model.generate(
                    **inputs,
                    max_length=4096,
                    do_sample=False,  # 明确使用贪婪解码
                    pad_token_id=tokenizer.eos_token_id
                )
            else:
                # 使用采样
                outputs = model.generate(
                    **inputs,
                    max_length=4096,
                    temperature=temperature,
                    do_sample=True,  # 使用采样
                    top_p=0.95,  # 可选：添加nucleus sampling
                    top_k=20,    # 可选：添加top-k sampling
                    pad_token_id=tokenizer.eos_token_id
                )
        
        solution = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the solution part
        solution = solution[len(prompt):].strip()
    
    # Format solution steps
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

# Function to load a model with vLLM
def load_model_with_vllm(model_path, dtype='half', tensor_parallel_size=None, gpu_memory_utilization=0.85, max_model_len=None):
    """Load a model with vLLM for optimized inference across multiple GPUs"""
    return LLM(
        model=model_path,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,  # Number of GPUs to use for tensor parallelism
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=True
    )

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
    input_path = args.input_path
    output_dir = args.output_dir
    dataset_name = args.dataset_name
    models = args.models
    temperatures = args.temperatures
    use_vllm = args.use_vllm

    # Load the dataset

    if dataset_name == "aime":
        dataset = load_dataset("AI-MO/aimo-validation-aime", split="train")

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
        'Qwen3-32B': 'Qwen/Qwen3-32B'
    }

    for model in models:
        if model not in model_paths:
            print(f"Warning: Model '{model}' is not recognized. Available models: {', '.join(model_paths.keys())}")
    
    # Check whether it's in model_paths, select them
    models = [model for model in models if model in model_paths]
    
    if not models:
        print("Error: No valid models specified. Exiting.")
        return
    
    models = {model: model_paths[model] for model in models}

    if args.subset_size > 0 and args.subset_size < len(dataset):
        print(f"Using subset of {args.subset_size} examples from the dataset")
        dataset = dataset[:args.subset_size]
    
    #---------------------------------------------------------------------------

    # Process each model
    for model_name, model_path in models.items():
        print(f"Loading model: {model_name} from {model_path}")
        
        # Create model-specific directory
        model_output_dir = os.path.join(output_dir, model_name, dataset_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        if use_vllm:
            print(f"Using vLLM for optimized inference with {args.tensor_parallel_size} GPUs")
            model = load_model_with_vllm(
                model_path,
                dtype=args.vllm_dtype,
                tensor_parallel_size=args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=args.max_model_len
            )
            tokenizer = None  # Not needed for vLLM
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=torch.float16,
                device_map="auto"  # Use automatic device allocation
            )
            print(f"Model loaded with device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'Not available'}")
            model.eval()

        for temp in temperatures:
            print(f"Processing with temperature: {temp}")
            
            counter = 0
            results = []

            for item in tqdm(dataset):

                question, uuid, source = dataset_extraction(item, dataset_name)
                
                try:
                    solution_steps = generate_solution(model, model_name, tokenizer, question, temp, use_vllm=use_vllm)
                    
                    # Create result object with only uuid, question, source and model_output_steps
                    result = {
                        "uuid": uuid,
                        "question": question,
                        "source": source,
                        "model_output_steps": solution_steps
                    }
                    
                    # if counter % 50 == 1:
                    #     print(result['question'])
                    #     print(result['model_output_steps'])
                    
                    results.append(result)
                    counter = counter + 1
                except Exception as e:
                    print(f"Error processing question {uuid}: {str(e)}")
            
            # Save results to file with new path structure
            output_path = os.path.join(model_output_dir, f"temperature_{temp}.json")
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate solutions with different temperature settings')
    parser.add_argument("--input_path", type=str, default="/home/dazhou/ReasonEval/dataset",
                        help="Path to the input dataset directory")
    parser.add_argument("--dataset_name", type=str, default="hybrid_reasoning",
                        help="Name of the dataset being processed (defaults to hybrid_reasoning)")
    parser.add_argument("--output_dir", type=str, default="/home/dazhou/ReasonEval/evaluation_results",
                        help="Directory to save the results")
    parser.add_argument("--models", type=str, nargs='+', 
                        default=['Abel-7B-002', 'WizardMath-7B-V1.1'],
                        help="List of models to evaluate. Choices: Abel-7B-002, WizardMath-7B-V1.1")
    parser.add_argument("--temperatures", type=float, nargs='+',
                        default=[0.1, 0.3, 0.6, 1.0, 1.3, 1.6, 2.0],
                        help="List of temperature values to test")
    parser.add_argument("--subset_size", type=int, default=0,
                        help="Number of examples to process (0 = all)")
    parser.add_argument("--use_vllm", default=True,
                        help="Use vLLM for optimized inference")
    # Add new arguments for vLLM multi-GPU configuration
    parser.add_argument("--tensor_parallel_size", type=int, default=None,
                        help="Number of GPUs to use for tensor parallelism (None = auto)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="Fraction of GPU memory to use (0.0 to 1.0)")
    parser.add_argument("--vllm_dtype", type=str, default="half", 
                        choices=["half", "float16", "bfloat16", "float"],
                        help="Data type for vLLM inference")
    parser.add_argument("--max_model_len", type=int, default=4096,
                        help="Maximum sequence length for the model (None = auto)")
    
    args = parser.parse_args()
    main(args)
