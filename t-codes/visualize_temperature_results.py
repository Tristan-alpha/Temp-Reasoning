import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
from matplotlib.colors import Normalize

def load_aggregated_results(file_paths, output_path="/home/dazhou/ReasonEval/evaluation_results/summary.csv"):
    """
    Load multiple aggregated result CSV files and combine them
    
    Args:
        file_paths: List of paths to aggregated CSV files
        
    Returns:
        Combined DataFrame with all results
    """
    dfs = []
    for path in file_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            # Extract model and dataset info from the path if not in the dataframe
            path_parts = path.split(os.sep)
            
            # Find model name from path
            if 'Model' not in df.columns:
                # Match a wider range of model names from your workspace
                model_names = [
                    'Abel-7B-002', 'WizardMath-7B-V1.1', 'gpt-4o-mini', 'deepseek-v3', 
                    'deepseek-chat', 'Qwen3-8B', 'Claude-3-7-Sonnet-20250219', 'Gemini-2.0-Flash',
                    'claude-3-7-sonnet-20250219', 'gemini-2.0-flash'
                ]
                for part in path_parts:
                    if any(model_name in part for model_name in model_names):
                        df['Model'] = part
                        break
            
            # Find dataset name from path
            if 'Dataset' not in df.columns:
                dataset_names = ['hybrid_reasoning', 'mr-gsm8k', 'aime', 'MATH', 'math']
                for part in path_parts:
                    if any(dataset_name in part for dataset_name in dataset_names):
                        df['Dataset'] = part
                        break
            
            # Check for evaluation folder
            if 'EvalModel' not in df.columns:
                evaluator_names = ['ReasonEval_7B', 'ReasonEval_34B']
                for part in path_parts:
                    if any(evaluator_name in part for evaluator_name in evaluator_names):
                        df['EvalModel'] = part
                        break

            
            dfs.append(df)
        else:
            print(f"Warning: File not found: {path}")
    
    if not dfs:
        raise ValueError("No valid data files found")
        
    combined_df = pd.concat(dfs, ignore_index=True)
    
    combined_df.to_csv(output_path, index=False)

    return combined_df

def create_metric_comparison_plot(df, output_path, figsize=(15, 10), dataset=None):
    """Create plots comparing metrics across models and temperatures"""
    # Filter by dataset if specified
    if dataset:
        df = df[df['Dataset'] == dataset]
        dataset_title = f" - {dataset}"
    else:
        dataset_title = ""
    
    # Separate data by evaluation model
    df_7B = df[df['EvalModel'] == 'ReasonEval_7B']
    df_34B = df[df['EvalModel'] == 'ReasonEval_34B']
    
    # Generate separate plots for each evaluation model
    if not df_7B.empty:
        _create_metric_comparison_plot_for_eval_model(df_7B, output_path, figsize, dataset, "ReasonEval_7B")
    
    if not df_34B.empty:
        _create_metric_comparison_plot_for_eval_model(df_34B, output_path, figsize, dataset, "ReasonEval_34B")
    
def _create_metric_comparison_plot_for_eval_model(df, output_path, figsize=(15, 10), dataset=None, eval_model=""):
    """Helper function to create plots for a specific evaluation model"""
    plt.figure(figsize=figsize)
    
    # Sort temperatures for proper line ordering
    all_temps = sorted(df['Temperature'].unique())
    
    # Dataset title for plot
    dataset_title = f" - {dataset}" if dataset else ""
    
    # Create a grid with 2 rows, 2 columns
    gs = GridSpec(2, 2, figure=plt.gcf())
    
    # Plot 1: Validity comparison
    ax1 = plt.subplot(gs[0, 0])
    sns.lineplot(data=df, x='Temperature', y='Solution_Level_Validity', 
                 hue='Model', marker='o', linewidth=2, ax=ax1)
    ax1.set_title('Validity Score vs Temperature', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Temperature', fontsize=12)
    ax1.set_ylabel('Validity Score', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xticks(all_temps)
    
    # Plot 2: Redundancy comparison
    ax2 = plt.subplot(gs[0, 1])
    sns.lineplot(data=df, x='Temperature', y='Solution_Level_Redundancy', 
                 hue='Model', marker='o', linewidth=2, ax=ax2)
    ax2.set_title('Redundancy Score vs Temperature', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Temperature', fontsize=12)
    ax2.set_ylabel('Redundancy Score', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_xticks(all_temps)
    
    # Plot 3: Shepherd score comparison
    ax3 = plt.subplot(gs[1, 0])
    sns.lineplot(data=df, x='Temperature', y='Solution_Score_Shepherd', 
                 hue='Model', marker='o', linewidth=2, ax=ax3)
    ax3.set_title('Shepherd Score vs Temperature', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Temperature', fontsize=12)
    ax3.set_ylabel('Shepherd Score', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.set_xticks(all_temps)
    
    # # Plot 4: All metrics normalized and combined
    # ax4 = plt.subplot(gs[1, 1])
    
    # # Normalize each metric to 0-1 scale for better comparison
    # metrics = ['Solution_Level_Validity', 'Solution_Level_Redundancy', 'Solution_Score_Shepherd']
    # models = df['Model'].unique()
    
    # normalized_data = []
    # for model in models:
    #     model_data = df[df['Model'] == model]
    #     for metric in metrics:
    #         values = model_data[metric].values
    #         min_val = values.min()
    #         max_val = values.max()
    #         range_val = max_val - min_val if max_val > min_val else 1.0
            
    #         for temp, value in zip(model_data['Temperature'], values):
    #             norm_value = (value - min_val) / range_val
    #             normalized_data.append({
    #                 'Model': model,
    #                 'Temperature': temp,
    #                 'Metric': metric,
    #                 'Value': norm_value
    #             })
    
    # norm_df = pd.DataFrame(normalized_data)
    # sns.lineplot(data=norm_df, x='Temperature', y='Value', hue='Metric', 
    #              style='Model', markers=True, dashes=False, ax=ax4)
    # ax4.set_title('Normalized Metrics Comparison', fontsize=14, fontweight='bold')
    # ax4.set_xlabel('Temperature', fontsize=12)
    # ax4.set_ylabel('Normalized Score', fontsize=12)
    # ax4.grid(True, linestyle='--', alpha=0.7)
    # ax4.set_xticks(all_temps)
    
    plt.suptitle(f'Metrics Comparison Across Models{dataset_title}\nEvaluation Model: {eval_model}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Adjust the output path to include dataset and eval model info
    output_dir = os.path.dirname(output_path)
    filename = os.path.splitext(os.path.basename(output_path))[0]
    if dataset:
        filename = f"{filename}_{dataset}"
    filename = f"{filename}_{eval_model}.png"
    new_output_path = os.path.join(output_dir, filename)
        
    plt.savefig(new_output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created metric comparison plot for {eval_model}: {new_output_path}")

def create_heatmap_visualization(df, output_path, figsize=(12, 8), dataset=None):
    """Create heatmap visualization for temperature vs metrics"""
    # Filter by dataset if specified
    if dataset:
        df = df[df['Dataset'] == dataset]
        dataset_title = f" - {dataset}"
    else:
        dataset_title = ""
    
    # Separate data by evaluation model
    df_7B = df[df['EvalModel'] == 'ReasonEval_7B']
    df_34B = df[df['EvalModel'] == 'ReasonEval_34B']
    
    # Generate separate plots for each evaluation model
    if not df_7B.empty:
        _create_heatmap_visualization_for_eval_model(df_7B, output_path, figsize, dataset, "ReasonEval_7B")
    
    if not df_34B.empty:
        _create_heatmap_visualization_for_eval_model(df_34B, output_path, figsize, dataset, "ReasonEval_34B")

def _create_heatmap_visualization_for_eval_model(df, output_path, figsize=(12, 8), dataset=None, eval_model=""):
    """Helper function to create heatmap for a specific evaluation model"""
    # Dataset title for plot
    dataset_title = f" - {dataset}" if dataset else ""
    
    # Set up a grid with 3 rows (one for each metric)
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    metrics = ['Solution_Level_Validity', 'Solution_Level_Redundancy', 'Solution_Score_Shepherd']
    titles = ['Validity Score', 'Redundancy Score', 'Shepherd Score']
    
    # Create a heatmap for each metric
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        # Pivot the data
        pivot_data = df.pivot(index='Model', columns='Temperature', values=metric)
        
        # Plot heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', ax=axes[i], cbar_kws={'label': metric})
        axes[i].set_title(f'{title} by Temperature', fontsize=12, fontweight='bold')
        
    plt.suptitle(f'Temperature vs Metrics Heatmap{dataset_title}\nEvaluation Model: {eval_model}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Adjust the output path to include dataset and eval model info
    output_dir = os.path.dirname(output_path)
    filename = os.path.splitext(os.path.basename(output_path))[0]
    if dataset:
        filename = f"{filename}_{dataset}"
    filename = f"{filename}_{eval_model}.png"
    new_output_path = os.path.join(output_dir, filename)
        
    plt.savefig(new_output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created heatmap visualization for {eval_model}: {new_output_path}")

def create_optimal_temperature_plot(df, output_path, figsize=(10, 6), dataset=None):
    """Create plot to identify optimal temperature for each model"""
    plt.figure(figsize=figsize)
    
    # Filter by dataset if specified
    if dataset:
        df = df[df['Dataset'] == dataset]
        dataset_title = f" - {dataset}"
    else:
        dataset_title = ""
    
    # Get unique models
    models = df['Model'].unique()
    
    # Find peak performance points for each model
    peak_points = []
    
    for model in models:
        model_data = df[df['Model'] == model]
        
        # Find temperature with best shepherd score (Shepherd)
        best_temp_idx = model_data['Solution_Score_Shepherd'].idxmax()
        best_temp_row = model_data.loc[best_temp_idx]
        
        peak_points.append({
            'Model': model,
            'Best_Temperature': best_temp_row['Temperature'],
            'Shepherd_Score': best_temp_row['Solution_Score_Shepherd'],
            'Validity_Score': best_temp_row['Solution_Level_Validity'],
            'Redundancy_Score': best_temp_row['Solution_Level_Redundancy']
        })
    
    # Plot all metrics with highlighted optimal points
    for model in models:
        model_data = df[df['Model'] == model]
        plt.plot(model_data['Temperature'], model_data['Solution_Score_Shepherd'], 
                 marker='o', linewidth=2, label=f'{model} (Shepherd)')
    
    # Add peak points with annotations
    for point in peak_points:
        plt.scatter(point['Best_Temperature'], point['Shepherd_Score'], 
                   color='red', s=100, zorder=5)
        plt.annotate(f"{point['Model']}\nTemp={point['Best_Temperature']}\nScore={point['Shepherd_Score']:.3f}", 
                    (point['Best_Temperature'], point['Shepherd_Score']),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title(f'Optimal Temperature for Each Model{dataset_title}', fontsize=16, fontweight='bold')
    plt.xlabel('Temperature', fontsize=14)
    plt.ylabel('Shepherd Score', fontsize=14)
    plt.legend()
    
    plt.tight_layout()
    
    # Adjust the output path to include dataset info if necessary
    if dataset:
        output_dir = os.path.dirname(output_path)
        filename = f"{os.path.splitext(os.path.basename(output_path))[0]}_{dataset}.png"
        new_output_path = os.path.join(output_dir, filename)
    else:
        new_output_path = output_path
        
    plt.savefig(new_output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a summary CSV
    summary_df = pd.DataFrame(peak_points)
    summary_df = summary_df.rename(columns={
        'Best_Temperature': 'Best Temperature',
        'Shepherd_Score': 'Best Shepherd Score',
        'Validity_Score': 'Corresponding Validity',
        'Redundancy_Score': 'Corresponding Redundancy'
    })
    
    # Save the summary CSV
    summary_path = os.path.splitext(new_output_path)[0] + '_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    
    print(f"Created optimal temperature plot: {new_output_path}")
    print(f"Created optimal temperature summary: {summary_path}")
    
    return summary_df

def create_trade_off_plot(df, output_path, figsize=(12, 10), dataset=None):
    """Create scatter plot showing trade-offs between metrics"""
    plt.figure(figsize=figsize)
    
    # Filter by dataset if specified
    if dataset:
        df = df[df['Dataset'] == dataset]
        dataset_title = f" - {dataset}"
    else:
        dataset_title = ""
    
    # Set up a grid with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Validity vs Shepherd
    sns.scatterplot(data=df, x='Solution_Level_Validity', y='Solution_Score_Shepherd', 
                    hue='Model', size='Temperature', sizes=(50, 200), ax=axes[0])
    
    for _, row in df.iterrows():
        axes[0].annotate(f"{row['Temperature']}", 
                       (row['Solution_Level_Validity'], row['Solution_Score_Shepherd']),
                       fontsize=9, ha='center')
    
    axes[0].set_title('Validity vs Shepherd', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Validity Score', fontsize=12)
    axes[0].set_ylabel('Shepherd Score', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Redundancy vs Shepherd
    sns.scatterplot(data=df, x='Solution_Level_Redundancy', y='Solution_Score_Shepherd', 
                    hue='Model', size='Temperature', sizes=(50, 200), ax=axes[1])
    
    for _, row in df.iterrows():
        axes[1].annotate(f"{row['Temperature']}", 
                       (row['Solution_Level_Redundancy'], row['Solution_Score_Shepherd']),
                       fontsize=9, ha='center')
    
    axes[1].set_title('Redundancy vs Shepherd', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Redundancy Score', fontsize=12)
    axes[1].set_ylabel('Shepherd Score', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.suptitle(f'Trade-offs Between Metrics{dataset_title}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Adjust the output path to include dataset info if necessary
    if dataset:
        output_dir = os.path.dirname(output_path)
        filename = f"{os.path.splitext(os.path.basename(output_path))[0]}_{dataset}.png"
        new_output_path = os.path.join(output_dir, filename)
    else:
        new_output_path = output_path
        
    plt.savefig(new_output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created trade-off plot: {new_output_path}")

def create_score_distribution_plots(input_dir, output_dir, models, datasets=None):
    """
    Create plots showing the distribution of scores across ranges for each model, dataset and temperature.
    
    Args:
        input_dir: Directory containing the detailed CSV files
        output_dir: Directory to save the visualizations
        models: List of model names to analyze
        datasets: List of dataset names to analyze (optional)
    """
    # Define the score ranges
    ranges = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    range_labels = ['0-25%', '25-50%', '50-75%', '75-100%']
    
    # Structure to hold data by model, dataset and evaluation model
    data_by_model_dataset_eval = {}
    
    for model in models:
        # Search for detailed CSVs containing this model name
        model_files = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if model in file and (file == "detailed.csv" or file == "source_aggregated.csv") and ("ReasonEval_7B" in root or "ReasonEval_34B" in root):
                    model_files.append(os.path.join(root, file))
        
        if not model_files:
            print(f"Warning: No detailed CSVs found for model {model}")
            continue
            
        for file_path in model_files:
            # Try to determine dataset from file path
            dataset = None
            for potential_dataset in ['hybrid_reasoning', 'mr-gsm8k', 'gsm8k', 'MATH', 'math']:
                if potential_dataset in file_path:
                    dataset = potential_dataset
                    break
            
            if not dataset and datasets:
                # If we couldn't determine dataset but user provided a list
                dataset = datasets[0]  # Default to first dataset
            
            # Determine evaluation model
            eval_model = None
            if "ReasonEval_7B" in file_path:
                eval_model = "ReasonEval_7B"
            elif "ReasonEval_34B" in file_path:
                eval_model = "ReasonEval_34B"
            else:
                continue  # Skip if no evaluation model found
            
            # Load the CSV
            df = pd.read_csv(file_path)
            
            # Add Model column if it doesn't exist
            if 'Model' not in df.columns:
                df['Model'] = model
                
            # Add Dataset column if it doesn't exist
            if 'Dataset' not in df.columns and dataset:
                df['Dataset'] = dataset
                
            # Add EvalModel column if it doesn't exist
            if 'EvalModel' not in df.columns:
                df['EvalModel'] = eval_model
            
            # Add to the data structure
            key = (model, dataset if dataset else 'unknown', eval_model)
            if key not in data_by_model_dataset_eval:
                data_by_model_dataset_eval[key] = df
    
    if not data_by_model_dataset_eval:
        print("No detailed data found. Cannot create distribution plots.")
        return
    
    # Define metrics to analyze
    metrics = {
        'Solution_Level_Validity': 'Validity Score',
        'Solution_Level_Redundancy': 'Redundancy Score',
        'Solution_Score_Shepherd': 'Shepherd Score'
    }
    
    # For each dataset, evaluation model, and metric, create a distribution plot
    for (model, dataset, eval_model), df in data_by_model_dataset_eval.items():
        # Create a specific output directory for this model/dataset/eval_model
        model_dataset_dir = os.path.join(output_dir, model, dataset if dataset != 'unknown' else 'default', eval_model)
        os.makedirs(model_dataset_dir, exist_ok=True)
        
        # For each metric, create distribution plots
        for metric_key, metric_name in metrics.items():
            # Skip if metric not in this dataframe
            if metric_key not in df.columns:
                continue
                
            # Get all temperatures for this dataset
            temperatures = sorted(df['Temperature'].unique())
            
            # Skip if no temperature info
            if not temperatures:
                continue
            
            # Set up the plot grid based on number of temperatures
            n_temps = len(temperatures)
            n_cols = min(3, n_temps)
            n_rows = (n_temps + n_cols - 1) // n_cols  # ceiling division
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = np.array([axes])
            elif n_rows == 1:
                axes = np.array([axes])
            axes = axes.flatten()
            
            # For each temperature, create a subplot
            for i, temp in enumerate(temperatures):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                
                # Filter for this temperature
                temp_df = df[df['Temperature'] == temp]
                if temp_df.empty:
                    ax.text(0.5, 0.5, f"No data for temperature {temp}", 
                           ha='center', va='center', fontsize=12)
                    ax.axis('off')
                    continue
                
                # Calculate distribution across ranges
                counts = [0] * len(ranges)
                for j, (low, high) in enumerate(ranges):
                    counts[j] = ((temp_df[metric_key] >= low) & (temp_df[metric_key] < high)).sum()
                
                total = len(temp_df)
                percentages = [count / total * 100 for count in counts] if total > 0 else [0] * len(ranges)
                
                # Create a DataFrame for plotting
                dist_df = pd.DataFrame({
                    'Range': range_labels,
                    'Percentage': percentages
                })
                
                # Plot bar chart
                sns.barplot(x='Range', y='Percentage', data=dist_df, ax=ax)
                
                ax.set_title(f'Temperature = {temp}', fontsize=14, fontweight='bold')
                ax.set_ylabel('Percentage (%)')
                ax.set_ylim(0, 100)
                
                # Add percentage labels on bars
                for j, p in enumerate(ax.patches):
                    ax.annotate(f"{percentages[j]:.1f}%", 
                               (p.get_x() + p.get_width()/2., p.get_height()),
                               ha='center', va='bottom', fontsize=10)
                
                ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Remove empty subplots
            for j in range(i + 1, len(axes)):
                axes[j].set_visible(False)
            
            plt.suptitle(f'Distribution of {metric_name} - {model} on {dataset}\nEvaluation Model: {eval_model}', 
                       fontsize=18, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            # Save the figure
            output_path = os.path.join(model_dataset_dir, f'{metric_key}_distribution.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Created distribution plot for {metric_name} ({eval_model}): {output_path}")

def prepare_output_directories(args):
    """Create organized output directories for visualizations"""
    # Main output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    for dataset in args.datasets:
        os.makedirs(os.path.join(args.output_dir, dataset), exist_ok=True)
    
    return os.path.join(args.output_dir, dataset)

def analyze_by_dataset(df, args, datasets_dir):
    """Create visualizations separated by dataset"""

    datasets = df['Dataset'].unique()
    
    
    # Create plots for each dataset
    for dataset in datasets:
        dataset_df = df[df['Dataset'] == dataset] if 'Dataset' in df.columns else df
        
            
        # Common visualizations for this dataset
        create_metric_comparison_plot(
            dataset_df, 
            os.path.join(datasets_dir, 'temperature_metrics_comparison.png'),
            dataset=dataset
        )
        
        create_heatmap_visualization(
            dataset_df,
            os.path.join(datasets_dir, 'temperature_metrics_heatmap.png'),
            dataset=dataset
        )
        
        create_optimal_temperature_plot(
            dataset_df,
            os.path.join(datasets_dir, 'optimal_temperature.png'),
            dataset=dataset
        )
        
        
        create_trade_off_plot(
            dataset_df,
            os.path.join(datasets_dir, 'metrics_tradeoff.png'),
            dataset=dataset
        )
    
def main():
    parser = argparse.ArgumentParser(description="Visualize temperature evaluation results")
    parser.add_argument("--input_dir", type=str, default="/home/dazhou/ReasonEval/evaluation_results",
                       help="Directory containing aggregated CSV files")
    parser.add_argument("--output_dir", type=str, default="/home/dazhou/ReasonEval/evaluation_results/visualizations",
                       help="Directory to save visualizations")
    parser.add_argument("--models", type=str, nargs='+', 
                       default=['Abel-7B-002', 'WizardMath-7B-V1.1', 'gpt-4o-mini', 'deepseek-chat'],
                       help="Models to include in visualizations")
    parser.add_argument("--datasets", type=str, nargs='+', 
                       default=['hybrid_reasoning'],
                       help="Datasets to include in visualizations")
    parser.add_argument("--evaluators", type=str, nargs='+',
                       default=['ReasonEval_7B', 'ReasonEval_34B'],
                       help="Evaluators to include in visualizations")
    parser.add_argument("--temperatures", type=str, nargs='+',
                       help="List of temperatures to analyze")


    args = parser.parse_args()
    
    # Prepare output directories
    datasets_dir = prepare_output_directories(args)
    
    # Construct paths to aggregated files
    agg_files = []

    # Use provided models to construct paths
    for model in args.models:
        if args.datasets:
            for dataset in args.datasets:
                # Update path to include evaluator subdirectories
                for evaluator in args.evaluators:
                    potential_path = os.path.join(args.input_dir, model, dataset, evaluator, "aggregated.csv")
                    if os.path.exists(potential_path):
                        agg_files.append(potential_path)
    
    if not agg_files:
        print("No aggregated files found. Please check your input directory and model names.")
        return
    
    try:
        # Load and combine data
        df = load_aggregated_results(agg_files)
        
        # Filter by temperature if specified
        if args.temperatures:
            temps = [float(t) for t in args.temperatures]
            df = df[df['Temperature'].isin(temps)]
        
        # Analyze data by dataset
        analyze_by_dataset(df, args, datasets_dir)
        
        
        print("Visualization complete!")
        
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
