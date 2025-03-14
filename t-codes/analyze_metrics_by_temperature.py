import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_metrics_by_temperature(input_dir, output_dir, models=None):
    """
    Analyze metrics from aggregated CSV files by temperature.
    
    Args:
        input_dir: Directory containing the aggregated CSV files
        output_dir: Directory to save the analysis results
        models: List of model names to analyze (if None, analyze all)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all aggregated CSV files in the input directory if models not specified
    all_files = []
    if models:
        for model in models:
            file_path = os.path.join(input_dir, f"{model}_aggregated.csv")
            if os.path.exists(file_path):
                all_files.append(file_path)
            else:
                print(f"Warning: File not found for model {model}")
    else:
        all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                    if f.endswith('_aggregated.csv')]
    
    if not all_files:
        print("No aggregated CSV files found.")
        return
        
    # Read and combine all aggregated CSV files
    dfs = []
    for file in all_files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Group by temperature
    temp_groups = combined_df.groupby('Temperature')
    
    # Calculate metrics for each temperature
    temp_metrics = []
    for temp, group in temp_groups:
        metrics = {
            'Temperature': temp,
            'Models_Count': len(group),
            'Avg_Solution_Level_Validity': group['Solution_Level_Validity'].mean(),
            'Max_Solution_Level_Validity': group['Solution_Level_Validity'].max(),
            'Min_Solution_Level_Validity': group['Solution_Level_Validity'].min(),
            'Avg_Solution_Level_Redundancy': group['Solution_Level_Redundancy'].mean(),
            'Avg_Solution_Score_Shepherd': group['Solution_Score_Shepherd'].mean(),
            'Best_Model': group.loc[group['Solution_Score_Shepherd'].idxmax()]['Model']
        }
        temp_metrics.append(metrics)
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(temp_metrics)
    
    # Save metrics to CSV
    metrics_file = os.path.join(output_dir, "temperature_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Metrics saved to {metrics_file}")
    
    # Create individual temperature comparison files
    for temp in combined_df['Temperature'].unique():
        temp_data = combined_df[combined_df['Temperature'] == temp].sort_values(
            by='Solution_Score_Shepherd', ascending=False
        )
        temp_file = os.path.join(output_dir, f"temperature_{temp}_comparison.csv")
        temp_data.to_csv(temp_file, index=False)
        print(f"Temperature {temp} comparison saved to {temp_file}")
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Validity across temperatures
    plt.subplot(2, 2, 1)
    sns.lineplot(data=metrics_df, x='Temperature', y='Avg_Solution_Level_Validity', marker='o')
    plt.title('Average Validity Score vs Temperature')
    plt.grid(True)
    
    # Plot 2: Redundancy across temperatures
    plt.subplot(2, 2, 2)
    sns.lineplot(data=metrics_df, x='Temperature', y='Avg_Solution_Level_Redundancy', marker='o')
    plt.title('Average Redundancy Score vs Temperature')
    plt.grid(True)
    
    # Plot 3: Correctness across temperatures
    plt.subplot(2, 2, 3)
    sns.lineplot(data=metrics_df, x='Temperature', y='Avg_Solution_Score_Shepherd', marker='o')
    plt.title('Average Solution Score vs Temperature')
    plt.grid(True)
    
    # Plot 4: Model comparison at each temperature
    plt.subplot(2, 2, 4)
    sns.barplot(data=combined_df, x='Temperature', y='Solution_Score_Shepherd', hue='Model')
    plt.title('Model Solution Scores by Temperature')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temperature_metrics_plots.png'))
    print(f"Plots saved to {os.path.join(output_dir, 'temperature_metrics_plots.png')}")

def main():
    parser = argparse.ArgumentParser(description="Analyze metrics from aggregated CSV files by temperature")
    parser.add_argument("--input_dir", type=str, default="/home/dazhou/ReasonEval/evaluation_results",
                       help="Directory containing the aggregated CSV files")
    parser.add_argument("--output_dir", type=str, default="/home/dazhou/ReasonEval/evaluation_results/temperature_analysis",
                       help="Directory to save the analysis results")
    parser.add_argument("--models", type=str, nargs='+', 
                       help="List of model names to analyze (if not specified, analyze all)")
    args = parser.parse_args()
    
    analyze_metrics_by_temperature(args.input_dir, args.output_dir, args.models)
    print("Analysis complete!")

if __name__ == "__main__":
    main()
