import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import re
from datasets import load_dataset

# Function to generate solution with a model
def generate_solution(model, tokenizer, question, temperature, device):
    prompt = f"Question: {question}\n\nProvide a step-by-step solution:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=1024,
            temperature=temperature,
            do_sample=(temperature > 0),
            pad_token_id=tokenizer.eos_token_id
        )
    
    solution = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the solution part
    solution = solution[len(prompt):].strip()
    
    # Format solution steps
    steps = []
    step_num = 1
    
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
        uuid = "unknown"
        question = item.get("problem", "")
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
    # Get GPU ID from command-line arguments
    gpu_id = args.gpu
    input_path = args.input_path
    output_dir = args.output_dir
    dataset_name = args.dataset_name
    models = args.models
    temperatures = args.temperatures

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device} for specific operations")
    
    # Load the dataset

    if dataset_name == "AIME":
        dataset = load_dataset("AI-MO/aimo-validation-aime")

    else:
        dataset_path = os.path.join(input_path, f"{dataset_name}.json")
        dataset = load_json_data(dataset_path)

    os.makedirs(output_dir, exist_ok=True)
    
    model_paths = {
        'Abel-7B-002': 'GAIR/Abel-7B-002',
        'WizardMath-7B-V1.1': 'WizardLMTeam/WizardMath-7B-V1.1'
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
                    solution_steps = generate_solution(model, tokenizer, question, temp, device)
                    
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
                        help="Path to the input dataset (dataset name will be extracted from filename if --dataset_name not provided)")
    parser.add_argument("--dataset_name", type=str, default="hybrid_reasoning",
                        help="Name of the dataset being processed (defaults to hybrid_reasoning)")
    parser.add_argument("--output_dir", type=str, default="/home/dazhou/ReasonEval/answer_by_models",
                        help="Directory to save the results")
    parser.add_argument("--models", type=str, nargs='+', 
                        default=['Abel-7B-002', 'WizardMath-7B-V1.1'],
                        help="List of models to evaluate. Choices: Abel-7B-002, WizardMath-7B-V1.1")
    parser.add_argument("--temperatures", type=float, nargs='+',
                        default=[0.1, 0.3, 0.6, 1.0, 1.3, 1.6, 2.0],
                        help="List of temperature values to test")
    parser.add_argument("--subset_size", type=int, default=0,
                        help="Number of examples to process (0 = all)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU ID to use for specific operations")
    
    args = parser.parse_args()
    main(args)
