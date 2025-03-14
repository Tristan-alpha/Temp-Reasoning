import os
import argparse
import pandas as pd

def split_detailed_by_temperature(input_file, output_dir):
    """
    Split a detailed results CSV file into separate files based on temperature values.
    
    Args:
        input_file: Path to the detailed CSV file
        output_dir: Directory to save the split files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the detailed CSV file
    df = pd.read_csv(input_file)
    
    # Get the model name from the filename
    model_name = os.path.basename(input_file).replace('_detailed.csv', '')
    
    # Group by temperature and save each group to a separate file
    for temp, group in df.groupby('Temperature'):
        output_file = os.path.join(output_dir, f"{model_name}_temp_{temp}.csv")
        group.to_csv(output_file, index=False)
        print(f"Created file: {output_file} with {len(group)} entries")

def main():
    parser = argparse.ArgumentParser(description="Split detailed results CSV by temperature")
    parser.add_argument("--input_file", type=str, required=True, 
                        help="Path to the detailed CSV file")
    parser.add_argument("--output_dir", type=str, default="/home/dazhou/ReasonEval/evaluation_results/by_temperature",
                        help="Directory to save the split files")
    args = parser.parse_args()
    
    split_detailed_by_temperature(args.input_file, args.output_dir)
    print("Splitting complete!")

if __name__ == "__main__":
    main()
