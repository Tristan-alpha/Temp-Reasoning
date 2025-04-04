import json
import argparse
import os
from tqdm import tqdm
import re
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import http.client
import urllib.parse
import json
import anthropic
from openai import OpenAI
from google import genai
from google.genai import types


def openai_answer(api_key, base_url, model_name, question, temperature, instructions):
    """Generate answer using OpenAI API"""
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": question}
        ],
        temperature=temperature,
        max_completion_tokens=1024,
    )
    
    solution = response.choices[0].message.content
    
    tokens_used = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens
    }
    
    return solution, tokens_used


def deepseek_answer(api_key, base_url, model_name, question, temperature, instructions):
    """Generate answer using DeepSeek API"""
    client = OpenAI(api_key=api_key, base_url=base_url)
    
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": question}
        ],
        temperature=temperature,
        max_completion_tokens=1024,
        max_tokens=1024,
        stream=False
    )
    
    solution = response.choices[0].message.content
    
    tokens_used = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens
    }
    
    return solution, tokens_used


def gemini_answer(api_key, question, temperature, instructions):
    """Generate answer using Google Gemini API"""
    client = genai.Client(api_key=api_key)
 
    # Combine instructions and question for Gemini
    prompt = f"{instructions}\n\nQuestion: {question}"
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
        max_output_tokens=1024,
        temperature=temperature
    )
    )
    
    solution = response.text
    
    # Gemini API might not provide token usage in the same way
    tokens_used = {
        "prompt_tokens": 0,  # Gemini may not provide this info
        "completion_tokens": 0,  # Gemini may not provide this info
        "total_tokens": 0  # Gemini may not provide this info
    }
    
    return solution, tokens_used


def claude_answer(api_key, base_url, question, temperature, instructions):
    """Generate answer using Anthropic Claude API"""
    # client = anthropic.Anthropic(api_key=api_key)
    client = OpenAI(
    api_key=api_key,  # Your Anthropic API key
    base_url=base_url  # Anthropic's API endpoint
)
    
    response = client.chat.completions.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=1024,
        max_completion_tokens=1024,
        temperature=temperature,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": question}
        ],
    )
    
    solution = response.choices[0].message.content
    
    tokens_used = {
        "prompt_tokens": response.usage.prompt_tokens,
        "completion_tokens": response.usage.completion_tokens,
        "total_tokens": response.usage.total_tokens
    }

    return solution, tokens_used


def grok_answer(api_key, base_url, question, temperature, instructions):
    """Generate answer using xAI Grok API"""
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    
    completion = client.chat.completions.create(
        model="grok-3-reasoner",
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": question},
        ],
        temperature=temperature,
        max_tokens=1024,
        max_completion_tokens=1024
    )
    
    solution = completion.choices[0].message.content
    
    tokens_used = {
        "prompt_tokens": completion.usage.prompt_tokens if hasattr(completion, 'usage') else 0,
        "completion_tokens": completion.usage.completion_tokens if hasattr(completion, 'usage') else 0,
        "total_tokens": completion.usage.total_tokens if hasattr(completion, 'usage') else 0
    }
    
    return solution, tokens_used


# Function to generate solution with appropriate API based on model name
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60))
def generate_solution(args, question, model_name, temperature):
    """Generate a solution using appropriate API based on model name with retry capability for API errors"""
    # Extract base URL from client object or use default
    base_url = args.base_url
    api_key = args.api_key
    
    # Define system instructions
    instructions = "Solve math problems step-by-step. You MUST end each COMPLETE step with a double newline (\\n\\n)."

    try:
        # Choose the appropriate API based on model name
        if model_name.startswith("deepseek"):
            solution, tokens_used = deepseek_answer(api_key, base_url, model_name, question, temperature, instructions)
        elif model_name.startswith("o1") or model_name.startswith("o3"):
            solution, tokens_used = openai_answer(api_key, base_url, model_name, question, temperature, instructions)
        elif model_name.startswith("gemini"):
            solution, tokens_used = gemini_answer(api_key, question, temperature, instructions)
        elif model_name.startswith("claude"):
            solution, tokens_used = claude_answer(api_key, base_url, question, temperature, instructions)
        elif model_name.startswith("grok"):
            solution, tokens_used = grok_answer(api_key, base_url, question, temperature, instructions)
        else:
            # Default to OpenAI for other models
            solution, tokens_used = openai_answer(api_key, base_url, model_name, question, temperature, instructions)
        
        # Format solution steps to match the style of Abel-7B-002 and WizardMath models
        steps = []
        
        # First, split by triple newlines as instructed in the system prompt
        # This should be the primary way to split steps if the model follows instructions
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
        
        return steps, solution, tokens_used
    
    except Exception as e:
        print(f"Error calling API: {str(e)}")
        raise e

def main(args):
    
    # Paths
    input_path = args.input_path
    output_dir = args.output_dir
    
    # Extract dataset name
    dataset_name = args.dataset_name
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the model mappings for OpenAI models
    model_paths = {
        'gpt-4o-mini': 'gpt-4o-mini',
        'deepseek-v3': 'deepseek-v3',
        'deepseek-r1': 'deepseek-r1',
        'deepseek-chat': 'deepseek-chat',
        'deepseek-reasoner': 'deepseek-reasoner',
        'claude-3-7-sonnet-20250219': 'claude-3-7-sonnet-20250219',
        'gemini-2.0-flash': 'gemini-2.0-flash',
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
    
    dataset_path = os.path.join(input_path, f"{dataset_name}.json")
    dataset = load_json_data(dataset_path)
    
    # Only process the subset if specified
    if args.subset_size > 0 and args.subset_size < len(dataset):
        print(f"Using subset of {args.subset_size} examples from the dataset")
        dataset = dataset[:args.subset_size] if not args.start_index else dataset[args.start_index:args.start_index + args.subset_size]
    
    # Process each model
    for model_name, model_path in models.items():
        print(f"Using model: {model_name}")
        
        # Create model-specific directory
        model_output_dir = os.path.join(output_dir, model_name, dataset_name)
        os.makedirs(model_output_dir, exist_ok=True)
        
        for temp in temperatures:
            print(f"Processing with temperature: {temp}")
            
            counter = 0
            results = []
            
            for item in tqdm(dataset):
                # Determine the source of the problem (math or mr-gsm8k)
                source = item.get("source", "")
                level = item.get("level", "")
                
                # Use appropriate fields based on the source
                if source == "math" or dataset_name == "math":
                    uuid = item.get("unique_id", "unknown")
                    question = item.get("problem", "")
                else:  # source == "mr-gsm8k" or default to mr-gsm8k format
                    uuid = item.get("uuid", "unknown")
                    question = item.get("question", "")
                
                if not question:
                    print(f"Skipping item {uuid} as it has no question")
                    continue
                
                # print(f"Processing question: {uuid}")
                
                try:
                    solution_steps, solution, tokens_used = generate_solution(args, question, model_path, temp)
                    
                    # Create result object with only uuid, question, source and model_output_steps
                    result = {
                        "uuid": uuid,
                        "question": question,
                        "source": source,
                        "level": level,
                        "model_output_steps": solution_steps,
                        "original_solution": solution,
                        "tokens": tokens_used
                    }
                    
                    results.append(result)
                    counter = counter + 1
                    
                    # Add a small delay to avoid hitting API rate limits
                    time.sleep(1)
                except Exception as e:
                    print(f"Error processing question {uuid}: {str(e)}")
            
            # Save results to file with new path structure
            output_path = os.path.join(model_output_dir, f"temperature_{temp}.json")
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"Results saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate solutions with OpenAI API at different temperature settings')
    parser.add_argument("--input_path", type=str, default="/home/dazhou/ReasonEval/dataset",
                        help="Path to the input dataset directory")
    parser.add_argument("--dataset_name", type=str, default="hybrid_reasoning",
                        help="Name of the dataset being processed (defaults to hybrid_reasoning)")
    parser.add_argument("--output_dir", type=str, default="/home/dazhou/ReasonEval/answer_by_models",
                        help="Directory to save the results")
    parser.add_argument("--api_key", type=str, required=True,
                        help="OpenAI API key")
    parser.add_argument("--base_url", type=str, default="https://api.chatanywhere.tech/v1",
                        help="Base URL for API (default: https://api.chatanywhere.tech/v1)")
    parser.add_argument("--models", type=str, nargs='+', 
                        default=['deepseek-v3'],
                        help="List of models to evaluate. Choices: gpt-3.5-turbo, gpt-3.5-turbo-1106, gpt-4, gpt-4-turbo, gpt-4-1106-preview, o1, o3-mini, deepseek-v3, deepseek-r1, claude-3-7-sonnet-20250219, gemini-2.0-flash, grok-3-reasoner")
    parser.add_argument("--temperatures", type=float, nargs='+',
                        default=[0.0, 0.7],
                        help="List of temperature values to test")
    parser.add_argument("--subset_size", type=int, default=0,
                        help="Number of examples to process (0 = all)")
    parser.add_argument("--start_index", type=int, default=0,
                        help="Start index for processing subset (default: 0)")
    
    args = parser.parse_args()
    main(args)