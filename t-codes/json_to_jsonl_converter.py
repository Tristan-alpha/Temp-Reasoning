import json
import os
import argparse
from pathlib import Path


def convert_json_to_jsonl(input_json_file, output_jsonl_file):
    """Convert a JSON file containing an array of objects to JSONL format."""
    try:
        # Read the JSON file
        with open(input_json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Write each object as a separate line in the JSONL file
        with open(output_jsonl_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        
        print(f"✓ Converted {input_json_file} to {output_jsonl_file}")
        return True
    except Exception as e:
        print(f"✗ Error converting {input_json_file}: {e}")
        return False


def process_directory(base_dir, output_dir=None):
    """Process all JSON files in the given directory and its subdirectories."""
    if output_dir is None:
        output_dir = base_dir  # Use the same directory structure by default
    
    base_path = Path(base_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    if not output_path.exists():
        output_path.mkdir(parents=True)
    
    successful = 0
    failed = 0
    
    # Find all JSON files in the directory and subdirectories
    for json_file in base_path.glob("**/*.json"):
        # Skip if the file is already a JSONL file renamed to .json
        with open(json_file, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            if first_char == '{':  # This is likely a valid JSON object
                f.seek(0)
                try:
                    # Check if it's an array of objects (typical JSON format we want to convert)
                    data = json.load(f)
                    if not isinstance(data, list):
                        continue  # Skip files that aren't lists of objects
                except:
                    continue  # Skip invalid JSON files
            else:
                continue  # Skip files that don't start with {
        
        # Determine the relative path to maintain directory structure
        rel_path = json_file.relative_to(base_path)
        
        # Create the output directory structure if needed
        output_file_dir = output_path / rel_path.parent
        if not output_file_dir.exists():
            output_file_dir.mkdir(parents=True)
        
        # Define the output file path (replacing .json with .jsonl)
        output_file = output_file_dir / (rel_path.stem + '.jsonl')
        
        # Convert the file
        success = convert_json_to_jsonl(str(json_file), str(output_file))
        if success:
            successful += 1
        else:
            failed += 1
    
    print(f"\nConversion completed: {successful} files converted, {failed} files failed")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSON files to JSONL format")
    parser.add_argument("--input", type=str, default="/home/dazhou/ReasonEval/answer_by_models",
                        help="Directory containing JSON files to convert")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (defaults to same as input)")
    args = parser.parse_args()
    
    process_directory(args.input, args.output)
