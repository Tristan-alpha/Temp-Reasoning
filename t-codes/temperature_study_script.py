import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import re

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

def main(args):
    # Set CUDA device
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    # Paths
    input_path = args.input_path
    output_dir = args.output_dir

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define the model paths
    model_paths = {
        'Abel-7B-002': 'GAIR/Abel-7B-002',
        'WizardMath-7B-V1.1': 'WizardLMTeam/WizardMath-7B-V1.1'
    }
    
    # Get models to evaluate from command line arguments
    models_to_evaluate = args.models
    
    # Validate requested models
    for model in models_to_evaluate:
        if model not in model_paths:
            print(f"Warning: Model '{model}' is not recognized. Available models: {', '.join(model_paths.keys())}")
    
    # Filter to only include valid models
    models_to_evaluate = [model for model in models_to_evaluate if model in model_paths]
    
    if not models_to_evaluate:
        print("Error: No valid models specified. Exiting.")
        return
    
    # Create dictionary of models to evaluate with their paths
    models = {model: model_paths[model] for model in models_to_evaluate}
    
    # Temperature settings to test
    temperatures = args.temperatures

    # Load the dataset
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

    dataset = load_json_data(input_path)
    
    # Only process the subset if specified
    if args.subset_size > 0 and args.subset_size < len(dataset):
        print(f"Using subset of {args.subset_size} examples from the dataset")
        dataset = dataset[:args.subset_size]
    
    # Process each model
    for model_name, model_path in models.items():
        print(f"Loading model: {model_name} from {model_path}")
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,
            device_map={"": device}
        ).to(device)
        
        for temp in temperatures:
            print(f"Processing with temperature: {temp}")
            
            counter = 0
            results = []
            
            for item in tqdm(dataset):
                uuid = item.get("uuid", "unknown")
                question = item.get("question", "")
                
                if not question:
                    print(f"Skipping item {uuid} as it has no question")
                    continue
                
                print(f"Processing question: {uuid}")
                
                try:
                    solution_steps = generate_solution(model, tokenizer, question, temp, device)
                    
                    # Create result object with only uuid, question and model_output_steps
                    result = {
                        "uuid": uuid,
                        "question": question,
                        "model_output_steps": solution_steps
                    }
                    
                    if counter % 50 == 1:
                        print(result['question'])
                        print(result['model_output_steps'])
                    
                    results.append(result)
                    counter = counter + 1
                except Exception as e:
                    print(f"Error processing question {uuid}: {str(e)}")
            
            # Save results to file
            output_path = os.path.join(output_dir, f"{model_name}_temperature_{temp}.json")
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate solutions with different temperature settings')
    parser.add_argument("--input_path", type=str, default="/home/dazhou/ReasonEval/dataset/mr-gsm8k.json",
                        help="Path to the input dataset")
    parser.add_argument("--output_dir", type=str, default="/home/dazhou/ReasonEval/temperature_study",
                        help="Directory to save the results")
    parser.add_argument("--models", type=str, nargs='+', 
                        default=['Abel-7B-002', 'WizardMath-7B-V1.1'],
                        help="List of models to evaluate. Choices: Abel-7B-002, WizardMath-7B-V1.1")
    parser.add_argument("--temperatures", type=float, nargs='+',
                        default=[0.1, 0.3, 0.6, 1.0, 1.3, 1.6, 2.0],
                        help="List of temperature values to test")
    parser.add_argument("--subset_size", type=int, default=0,
                        help="Number of examples to process (0 = all)")
    parser.add_argument("--gpu", type=int, default=0)
    
    args = parser.parse_args()
    main(args)
