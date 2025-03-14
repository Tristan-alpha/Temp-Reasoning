import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

def load_aggregated_results(file_paths):
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
            dfs.append(df)
        else:
            print(f"Warning: File not found: {path}")
    
    if not dfs:
        raise ValueError("No valid data files found")
        
    return pd.concat(dfs, ignore_index=True)

def create_metric_comparison_plot(df, output_path, figsize=(15, 10)):
    """Create plots comparing metrics across models and temperatures"""
    plt.figure(figsize=figsize)
    
    # Sort temperatures for proper line ordering
    all_temps = sorted(df['Temperature'].unique())
    
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
    ax3.set_title('Correctness Score vs Temperature', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Temperature', fontsize=12)
    ax3.set_ylabel('Shepherd Score', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.set_xticks(all_temps)
    
    # Plot 4: All metrics normalized and combined
    ax4 = plt.subplot(gs[1, 1])
    
    # Normalize each metric to 0-1 scale for better comparison
    metrics = ['Solution_Level_Validity', 'Solution_Level_Redundancy', 'Solution_Score_Shepherd']
    models = df['Model'].unique()
    
    normalized_data = []
    for model in models:
        model_data = df[df['Model'] == model]
        for metric in metrics:
            values = model_data[metric].values
            min_val = values.min()
            max_val = values.max()
            range_val = max_val - min_val if max_val > min_val else 1.0
            
            for temp, value in zip(model_data['Temperature'], values):
                norm_value = (value - min_val) / range_val
                normalized_data.append({
                    'Model': model,
                    'Temperature': temp,
                    'Metric': metric,
                    'Value': norm_value
                })
    
    norm_df = pd.DataFrame(normalized_data)
    sns.lineplot(data=norm_df, x='Temperature', y='Value', hue='Metric', 
                 style='Model', markers=True, dashes=False, ax=ax4)
    ax4.set_title('Normalized Metrics Comparison', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Temperature', fontsize=12)
    ax4.set_ylabel('Normalized Score', fontsize=12)
    ax4.grid(True, linestyle='--', alpha=0.7)
    ax4.set_xticks(all_temps)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created metric comparison plot: {output_path}")

def create_model_comparison_plot(df, output_dir, figsize=(14, 8)):
    """Create bar charts comparing models at each temperature"""
    # Get unique temperatures sorted
    temperatures = sorted(df['Temperature'].unique())
    
    for temp in temperatures:
        temp_data = df[df['Temperature'] == temp]
        
        if len(temp_data) < 2:  # Skip if only one model available for this temperature
            continue
            
        plt.figure(figsize=figsize)
        
        # Set up a 1x3 grid of subplots
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot validity scores
        sns.barplot(data=temp_data, x='Model', y='Solution_Level_Validity', ax=axes[0])
        axes[0].set_title(f'Validity at Temp={temp}')
        axes[0].set_xlabel('')
        axes[0].set_ylim(0.5, 0.85)  # Adjusted based on the data provided
        
        # Plot redundancy scores
        sns.barplot(data=temp_data, x='Model', y='Solution_Level_Redundancy', ax=axes[1])
        axes[1].set_title(f'Redundancy at Temp={temp}')
        axes[1].set_xlabel('')
        axes[1].set_ylim(0.1, 0.25)  # Adjusted based on the data provided
        
        # Plot shepherd scores
        sns.barplot(data=temp_data, x='Model', y='Solution_Score_Shepherd', ax=axes[2])
        axes[2].set_title(f'Correctness at Temp={temp}')
        axes[2].set_xlabel('')
        axes[2].set_ylim(0.45, 0.7)  # Adjusted based on the data provided
        
        # Add value labels on top of bars
        for ax in axes:
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.3f}', 
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='bottom', fontsize=9, rotation=0)
        
        plt.suptitle(f'Model Comparison at Temperature {temp}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(output_dir, f'model_comparison_temp_{temp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created model comparison for temperature {temp}: {output_path}")

def create_heatmap_visualization(df, output_path, figsize=(12, 8)):
    """Create heatmap visualization for temperature vs metrics"""
    # Pivot data to create a matrix suitable for heatmap
    models = df['Model'].unique()
    
    plt.figure(figsize=figsize)
    
    # Set up a grid with 3 rows (one for each metric)
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    metrics = ['Solution_Level_Validity', 'Solution_Level_Redundancy', 'Solution_Score_Shepherd']
    titles = ['Validity Score', 'Redundancy Score', 'Correctness Score']
    
    # Create a heatmap for each metric
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        # Pivot the data
        pivot_data = df.pivot(index='Model', columns='Temperature', values=metric)
        
        # Plot heatmap
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', ax=axes[i], cbar_kws={'label': metric})
        axes[i].set_title(f'{title} by Temperature', fontsize=12, fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created heatmap visualization: {output_path}")

def create_optimal_temperature_plot(df, output_path, figsize=(10, 6)):
    """Create plot to identify optimal temperature for each model"""
    plt.figure(figsize=figsize)
    
    # Get unique models
    models = df['Model'].unique()
    
    # Find peak performance points for each model
    peak_points = []
    
    for model in models:
        model_data = df[df['Model'] == model]
        
        # Find temperature with best shepherd score (correctness)
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
                 marker='o', linewidth=2, label=f'{model} (Correctness)')
    
    # Add peak points with annotations
    for point in peak_points:
        plt.scatter(point['Best_Temperature'], point['Shepherd_Score'], 
                   color='red', s=100, zorder=5)
        plt.annotate(f"{point['Model']}\nTemp={point['Best_Temperature']}\nScore={point['Shepherd_Score']:.3f}", 
                    (point['Best_Temperature'], point['Shepherd_Score']),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('Optimal Temperature for Each Model', fontsize=16, fontweight='bold')
    plt.xlabel('Temperature', fontsize=14)
    plt.ylabel('Correctness Score', fontsize=14)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created optimal temperature plot: {output_path}")

def create_trade_off_plot(df, output_path, figsize=(12, 10)):
    """Create scatter plot showing trade-offs between metrics"""
    plt.figure(figsize=figsize)
    
    # Set up a grid with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Validity vs Correctness
    sns.scatterplot(data=df, x='Solution_Level_Validity', y='Solution_Score_Shepherd', 
                    hue='Model', size='Temperature', sizes=(50, 200), ax=axes[0])
    
    for _, row in df.iterrows():
        axes[0].annotate(f"{row['Temperature']}", 
                       (row['Solution_Level_Validity'], row['Solution_Score_Shepherd']),
                       fontsize=9, ha='center')
    
    axes[0].set_title('Validity vs Correctness', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Validity Score', fontsize=12)
    axes[0].set_ylabel('Correctness Score', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Redundancy vs Correctness
    sns.scatterplot(data=df, x='Solution_Level_Redundancy', y='Solution_Score_Shepherd', 
                    hue='Model', size='Temperature', sizes=(50, 200), ax=axes[1])
    
    for _, row in df.iterrows():
        axes[1].annotate(f"{row['Temperature']}", 
                       (row['Solution_Level_Redundancy'], row['Solution_Score_Shepherd']),
                       fontsize=9, ha='center')
    
    axes[1].set_title('Redundancy vs Correctness', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Redundancy Score', fontsize=12)
    axes[1].set_ylabel('Correctness Score', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created trade-off plot: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize temperature evaluation results")
    parser.add_argument("--input_dir", type=str, default="/home/dazhou/ReasonEval/evaluation_results",
                       help="Directory containing aggregated CSV files")
    parser.add_argument("--output_dir", type=str, default="/home/dazhou/ReasonEval/evaluation_results/visualizations",
                       help="Directory to save visualizations")
    parser.add_argument("--models", type=str, nargs='+', 
                       default=['Abel-7B-002', 'WizardMath-7B-V1.1'],
                       help="Models to include in visualizations")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Construct paths to aggregated files
    agg_files = [os.path.join(args.input_dir, f"{model}_aggregated.csv") for model in args.models]
    
    try:
        # Load and combine data
        df = load_aggregated_results(agg_files)
        
        # Create various visualizations
        create_metric_comparison_plot(df, 
                                     os.path.join(args.output_dir, 'temperature_metrics_comparison.png'))
        
        create_model_comparison_plot(df, args.output_dir)
        
        create_heatmap_visualization(df,
                                    os.path.join(args.output_dir, 'temperature_metrics_heatmap.png'))
        
        create_optimal_temperature_plot(df,
                                      os.path.join(args.output_dir, 'optimal_temperature.png'))
        
        create_trade_off_plot(df,
                             os.path.join(args.output_dir, 'metrics_tradeoff.png'))
        
        # Create a summary table
        summary = []
        for model in df['Model'].unique():
            model_data = df[df['Model'] == model]
            best_temp_idx = model_data['Solution_Score_Shepherd'].idxmax()
            best_temp_row = model_data.loc[best_temp_idx]
            
            summary.append({
                'Model': model,
                'Best Temperature': best_temp_row['Temperature'],
                'Best Correctness Score': best_temp_row['Solution_Score_Shepherd'],
                'Corresponding Validity': best_temp_row['Solution_Level_Validity'],
                'Corresponding Redundancy': best_temp_row['Solution_Level_Redundancy'],
            })
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(os.path.join(args.output_dir, 'optimal_temperature_summary.csv'), index=False)
        print(f"Summary table saved to: {os.path.join(args.output_dir, 'optimal_temperature_summary.csv')}")
        
        print("Visualization complete!")
        
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
