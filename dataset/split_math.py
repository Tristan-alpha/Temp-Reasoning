import json

def load_math_data():
    """Load math.json data and organize by level"""
    math_data = []
    with open("math.json", "r") as f:
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

data = load_math_data()

for level, items in data.items():
    with open(f"math-{level}.json", "w") as f:
        for item in items:
            json.dump(item, f)
            f.write("\n")
