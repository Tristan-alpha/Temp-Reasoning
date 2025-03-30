import json
import random
import os

# Set random seed for reproducibility
random.seed(42)

def load_math_data():
    """Load math.json data and organize by level"""
    math_data = []
    with open("/home/dazhou/ReasonEval/dataset/math.json", "r") as f:
        for line in f:
            if line.strip():
                try:
                    obj = json.loads(line)
                    math_data.append(obj)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line as JSON: {line[:50]}...")
    
    # Organize data by level
    data_by_level = {}
    for item in math_data:
        level = item.get("level")
        if level not in data_by_level:
            data_by_level[level] = []
        data_by_level[level].append(item)
    
    return data_by_level

def load_mr_gsm8k_data():
    """Load mr-gsm8k.json data"""
    gsm8k_data = []
    with open("/home/dazhou/ReasonEval/dataset/mr-gsm8k.json", "r") as f:
        for line in f:
            if line.strip():
                try:
                    obj = json.loads(line)
                    gsm8k_data.append(obj)
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line as JSON: {line[:50]}...")
    
    return gsm8k_data

def create_hybrid_dataset():
    # Load data
    math_by_level = load_math_data()
    mr_gsm8k_data = load_mr_gsm8k_data()
    
    # Sample 50 problems from each level of math.json
    math_samples = []
    for level in range(1, 6):  # Levels 1-5
        if level in math_by_level and len(math_by_level[level]) > 50:
            samples = random.sample(math_by_level[level], 50)
            math_samples.extend(samples)
        else:
            print(f"Warning: Not enough problems for level {level}. Using all available.")
            math_samples.extend(math_by_level.get(level, []))
    
    # Sample 50 problems from mr-gsm8k.json
    if len(mr_gsm8k_data) > 50:
        gsm8k_samples = random.sample(mr_gsm8k_data, 50)
    else:
        print(f"Warning: Not enough problems in mr-gsm8k.json. Using all available.")
        gsm8k_samples = mr_gsm8k_data
    
    # Add source field to track the origin of each problem
    for item in math_samples:
        item["source"] = "math"
    
    for item in gsm8k_samples:
        item["source"] = "mr-gsm8k"
    
    # Combine the datasets
    hybrid_dataset = math_samples + gsm8k_samples
    
    # Shuffle the combined dataset
    # random.shuffle(hybrid_dataset)
    
    # Save the hybrid dataset
    with open("/home/dazhou/ReasonEval/dataset/hybrid_reasoning.json", "w") as f:
        for item in hybrid_dataset:
            f.write(json.dumps(item) + "\n")
    
    print(f"Created hybrid dataset with {len(hybrid_dataset)} problems")
    print(f"- Math problems: {len(math_samples)}")
    print(f"- GSM8K problems: {len(gsm8k_samples)}")

if __name__ == "__main__":
    create_hybrid_dataset()