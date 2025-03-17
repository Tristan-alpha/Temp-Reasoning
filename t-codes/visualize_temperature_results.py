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
    ax3.set_title('Shepherd Score vs Temperature', fontsize=14, fontweight='bold')
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
        axes[2].set_title(f'Shepherd at Temp={temp}')
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
    titles = ['Validity Score', 'Redundancy Score', 'Shepherd Score']
    
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
    plt.title('Optimal Temperature for Each Model', fontsize=16, fontweight='bold')
    plt.xlabel('Temperature', fontsize=14)
    plt.ylabel('Shepherd Score', fontsize=14)
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
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created trade-off plot: {output_path}")

def create_score_distribution_plots(input_dir, output_dir, models):
    """
    Create plots showing the distribution of scores across ranges for each model and temperature.
    
    Args:
        input_dir: Directory containing the detailed CSV files
        output_dir: Directory to save the visualizations
        models: List of model names to analyze
    """
    # Define the score ranges
    ranges = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    range_labels = ['0-25%', '25-50%', '50-75%', '75-100%']
    
    # Load detailed data for each model
    model_data = {}
    for model in models:
        detailed_csv = os.path.join(input_dir, f"{model}_detailed.csv")
        if os.path.exists(detailed_csv):
            model_data[model] = pd.read_csv(detailed_csv)
        else:
            print(f"Warning: Detailed CSV for {model} not found at {detailed_csv}")
    
    if not model_data:
        print("No detailed data found. Cannot create distribution plots.")
        return
    
    # Define metrics to analyze
    metrics = {
        'Solution_Level_Validity': 'Validity Score',
        'Solution_Level_Redundancy': 'Redundancy Score',
        'Solution_Score_Shepherd': 'Shepherd Score'
    }
    
    # For each metric, create a distribution plot
    for metric_key, metric_name in metrics.items():
        # Get all temperatures across all models
        all_temps = set()
        for df in model_data.values():
            all_temps.update(df['Temperature'].unique())
        all_temps = sorted(all_temps)
        
        # Set up the plot grid based on number of temperatures
        n_temps = len(all_temps)
        n_cols = min(3, n_temps)
        n_rows = (n_temps + n_cols - 1) // n_cols  # ceiling division
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        elif n_rows == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # For each temperature, create a subplot
        for i, temp in enumerate(all_temps):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Prepare data for this temperature
            dist_data = []
            
            for model, df in model_data.items():
                temp_df = df[df['Temperature'] == temp]
                if temp_df.empty:
                    continue
                
                # Calculate distribution across ranges
                counts = [0] * len(ranges)
                for j, (low, high) in enumerate(ranges):
                    counts[j] = ((temp_df[metric_key] >= low) & (temp_df[metric_key] < high)).sum()
                
                total = len(temp_df)
                percentages = [count / total * 100 for count in counts] if total > 0 else [0] * len(ranges)
                
                for j, pct in enumerate(percentages):
                    dist_data.append({
                        'Model': model,
                        'Range': range_labels[j],
                        'Percentage': pct
                    })
            
            if not dist_data:
                ax.text(0.5, 0.5, f"No data for temperature {temp}", 
                        ha='center', va='center', fontsize=12)
                ax.axis('off')
                continue
                
            # Create the distribution plot for this temperature
            dist_df = pd.DataFrame(dist_data)
            
            # Pivot the data for stacked bars
            pivot_df = dist_df.pivot(index='Model', columns='Range', values='Percentage')
            pivot_df = pivot_df.reindex(columns=range_labels)  # ensure correct order
            
            # Plot stacked bar chart
            pivot_df.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
            
            ax.set_title(f'Temperature = {temp}', fontsize=14, fontweight='bold')
            ax.set_ylabel('Percentage (%)')
            ax.set_ylim(0, 100)
            
            # Add percentage labels on each segment
            for bar_container in ax.containers:
                # Only label if the segment is large enough to be visible
                ax.bar_label(bar_container, fmt='%.1f%%', label_type='center',
                           labels=[f"{v:.1f}%" if v > 5 else "" for v in bar_container.datavalues])
            
            ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        # Add a common legend for all subplots
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, title='Score Range', loc='upper center', 
                 bbox_to_anchor=(0.5, 0.05), ncol=4)
        
        # Adjust subplot legends
        for ax in axes:
            if ax.get_legend():
                ax.get_legend().remove()
        
        plt.suptitle(f'Distribution of {metric_name} Across Score Ranges', fontsize=18, fontweight='bold')
        plt.tight_layout(rect=[0, 0.05, 1, 0.97])  # Adjust for the common legend
        
        # Save the figure
        output_path = os.path.join(output_dir, f'{metric_key}_distribution.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created distribution plot for {metric_name}: {output_path}")
    
    # Create an all-in-one heatmap visualization
    create_distribution_heatmaps(model_data, output_dir)

def create_distribution_heatmaps(model_data, output_dir):
    """
    Create heatmaps showing distribution of scores across ranges for each model and temperature.
    
    Args:
        model_data: Dictionary of DataFrames with detailed data for each model
        output_dir: Directory to save the visualizations
    """
    # Define the score ranges and metrics
    ranges = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    range_labels = ['0-25%', '25-50%', '50-75%', '75-100%']
    
    metrics = {
        'Solution_Level_Validity': 'Validity Score',
        'Solution_Level_Redundancy': 'Redundancy Score', 
        'Solution_Score_Shepherd': 'Shepherd Score'
    }
    
    # Create a single figure for all heatmaps
    n_models = len(model_data)
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(n_models, n_metrics, figsize=(5*n_metrics, 4*n_models))
    
    if n_models == 1:
        axes = axes.reshape(1, -1)
    
    # For each model and metric, create a heatmap
    for i, (model, df) in enumerate(model_data.items()):
        temperatures = sorted(df['Temperature'].unique())
        
        for j, (metric_key, metric_name) in enumerate(metrics.items()):
            ax = axes[i, j]
            
            # Prepare data for the heatmap
            heatmap_data = []
            
            for temp in temperatures:
                temp_df = df[df['Temperature'] == temp]
                if temp_df.empty:
                    continue
                
                # Calculate distribution across ranges
                row_data = {'Temperature': temp}
                
                for k, (low, high) in enumerate(ranges):
                    count = ((temp_df[metric_key] >= low) & (temp_df[metric_key] < high)).sum()
                    total = len(temp_df)
                    percentage = (count / total * 100) if total > 0 else 0
                    row_data[range_labels[k]] = percentage
                
                heatmap_data.append(row_data)
            
            if not heatmap_data:
                ax.text(0.5, 0.5, "No data available", 
                       ha='center', va='center', fontsize=12)
                ax.axis('off')
                continue
            
            # Convert to DataFrame and pivot
            heatmap_df = pd.DataFrame(heatmap_data)
            heatmap_df = heatmap_df.set_index('Temperature')
            
            # Plot heatmap
            sns.heatmap(heatmap_df, annot=True, fmt='.1f', cmap='viridis', 
                      cbar_kws={'label': 'Percentage (%)'}, ax=ax)
            
            # Set title and labels
            if i == 0:
                ax.set_title(metric_name, fontsize=14, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f"{model}\nTemperature", fontsize=12)
            else:
                ax.set_ylabel("")
                
            ax.set_xlabel("Score Range")
    
    plt.suptitle('Distribution of Scores Across Ranges (percentage in each range)', 
               fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save the figure
    output_path = os.path.join(output_dir, 'score_distribution_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created score distribution heatmap: {output_path}")

def create_combined_model_comparison(df, output_path, figsize=(15, 10)):
    """Create plots comparing both models together on the same charts"""
    plt.figure(figsize=figsize)
    
    # Sort temperatures for proper line ordering
    all_temps = sorted(df['Temperature'].unique())
    models = df['Model'].unique()
    
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
    
    # Plot 4: Bar chart comparing models at each temperature
    ax4 = plt.subplot(gs[1, 1])
    
    # Create a melted dataframe for the bar chart
    bar_data = []
    for model in models:
        model_data = df[df['Model'] == model]
        for _, row in model_data.iterrows():
            bar_data.append({
                'Model': model,
                'Temperature': row['Temperature'],
                'Shepherd Score': row['Solution_Score_Shepherd']
            })
    
    bar_df = pd.DataFrame(bar_data)
    sns.barplot(x='Temperature', y='Shepherd Score', hue='Model', data=bar_df, ax=ax4)
    ax4.set_title('Model Comparison by Temperature', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Temperature', fontsize=12)
    ax4.set_ylabel('Shepherd Score', fontsize=12)
    
    plt.suptitle(f'Direct Comparison of {" vs ".join(models)}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created combined model comparison: {output_path}")

def create_radar_comparison(df, output_path, figsize=(12, 10)):
    """Create radar charts to compare models across all metrics at different temperatures"""
    # Metrics to include in the radar chart
    metrics = ['Solution_Level_Validity', 'Solution_Level_Redundancy', 'Solution_Score_Shepherd']
    metric_labels = ['Validity', 'Redundancy', 'Shepherd']
    
    # Get unique models and temperatures
    models = df['Model'].unique()
    temperatures = sorted(df['Temperature'].unique())
    
    # Number of metrics
    N = len(metrics)
    
    # Create figure
    fig = plt.figure(figsize=figsize)
    
    # Calculate number of rows and columns for subplots
    n_temps = len(temperatures)
    n_cols = min(3, n_temps)
    n_rows = (n_temps + n_cols - 1) // n_cols
    
    # Create subplots for each temperature
    for i, temp in enumerate(temperatures):
        # Filter data for this temperature
        temp_data = df[df['Temperature'] == temp]
        
        # Calculate angles for the radar chart
        angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # Close the loop
        
        # Add subplot
        ax = fig.add_subplot(n_rows, n_cols, i+1, polar=True)
        
        # Plot each model
        for model in models:
            model_data = temp_data[temp_data['Model'] == model]
            if not model_data.empty:
                # Get values for each metric
                values = model_data[metrics].values[0].tolist()
                values += values[:1]  # Close the loop
                
                # Plot the model
                ax.plot(angles, values, linewidth=2, label=model)
                ax.fill(angles, values, alpha=0.1)
        
        # Set labels and title
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        ax.set_title(f'Temperature: {temp}', fontsize=12)
        
        # Add legend to the first subplot
        if i == 0:
            ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.suptitle('Model Comparison Radar Charts by Temperature', fontsize=16, fontweight='bold', y=1.02)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created radar chart comparison: {output_path}")

def create_metric_scatter_matrix(df, output_path, figsize=(15, 15)):
    """Create a scatter matrix showing relationships between metrics with model differentiation"""
    # Select metrics for comparison
    metrics = ['Solution_Level_Validity', 'Solution_Level_Redundancy', 'Solution_Score_Shepherd']
    metric_labels = ['Validity', 'Redundancy', 'Shepherd']
    
    # Create a new DataFrame with clearer column names
    plot_df = df[['Model', 'Temperature'] + metrics].copy()
    plot_df.columns = ['Model', 'Temperature'] + metric_labels
    
    # Create the pair plot with different colors for models and sizes for temperature
    g = sns.pairplot(
        plot_df, 
        hue='Model', 
        vars=metric_labels,
        height=4,
        plot_kws={'alpha': 0.8, 'edgecolor': 'w', 'linewidth': 0.5}
    )
    
    # Add temperature information as text annotations
    for ax in g.axes.flat:
        if ax.get_xlabel() and ax.get_ylabel():
            x_label = ax.get_xlabel()
            y_label = ax.get_ylabel()
            if x_label in metric_labels and y_label in metric_labels:
                for model in plot_df['Model'].unique():
                    model_data = plot_df[plot_df['Model'] == model]
                    for _, row in model_data.iterrows():
                        ax.text(row[x_label], row[y_label], f"{row['Temperature']}", 
                               fontsize=8, ha='center', va='center')
    
    g.fig.suptitle('Metric Relationships Across Models and Temperatures', fontsize=16, y=1.02)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created metric scatter matrix: {output_path}")

def create_temperature_boxplot_comparison(df, output_path, figsize=(15, 10)):
    """Create boxplots comparing metrics at each temperature across models"""
    plt.figure(figsize=figsize)
    
    # Sort temperatures for proper line ordering
    all_temps = sorted(df['Temperature'].unique())
    
    # Create a grid with 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot 1: Validity comparison
    sns.boxplot(data=df, x='Temperature', y='Solution_Level_Validity', 
                hue='Model', ax=axes[0])
    axes[0].set_title('Validity Score Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Temperature', fontsize=12)
    axes[0].set_ylabel('Validity Score', fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].set_xticks(range(len(all_temps)))
    axes[0].set_xticklabels([str(t) for t in all_temps], rotation=45)
    
    # Plot 2: Redundancy comparison
    sns.boxplot(data=df, x='Temperature', y='Solution_Level_Redundancy', 
                hue='Model', ax=axes[1])
    axes[1].set_title('Redundancy Score Distribution', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Temperature', fontsize=12)
    axes[1].set_ylabel('Redundancy Score', fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].set_xticks(range(len(all_temps)))
    axes[1].set_xticklabels([str(t) for t in all_temps], rotation=45)
    
    # Plot 3: Shepherd comparison
    sns.boxplot(data=df, x='Temperature', y='Solution_Score_Shepherd', 
                hue='Model', ax=axes[2])
    axes[2].set_title('Shepherd Score Distribution', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Temperature', fontsize=12)
    axes[2].set_ylabel('Shepherd Score', fontsize=12)
    axes[2].grid(True, linestyle='--', alpha=0.7)
    axes[2].set_xticks(range(len(all_temps)))
    axes[2].set_xticklabels([str(t) for t in all_temps], rotation=45)
    
    # Use a shared legend for the entire figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=2)
    
    # Remove individual legends to avoid duplication
    for ax in axes:
        if ax.get_legend():
            ax.get_legend().remove()
    
    plt.suptitle('Model Performance Distribution Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.1, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created temperature boxplot comparison: {output_path}")

def create_model_difference_plot(df, output_path, figsize=(15, 8)):
    """Create plots showing the difference between models across temperatures"""
    # Get unique models
    models = df['Model'].unique()
    
    if len(models) != 2:
        print("Model difference plot requires exactly 2 models")
        return
        
    # Get sorted temperatures
    temperatures = sorted(df['Temperature'].unique())
    
    # Prepare data for comparison
    model1_data = df[df['Model'] == models[0]].sort_values('Temperature')
    model2_data = df[df['Model'] == models[1]].sort_values('Temperature')
    
    # Calculate differences (model1 - model2)
    diff_data = []
    
    for temp in temperatures:
        m1_row = model1_data[model1_data['Temperature'] == temp]
        m2_row = model2_data[model2_data['Temperature'] == temp]
        
        if len(m1_row) == 0 or len(m2_row) == 0:
            continue
            
        diff_data.append({
            'Temperature': temp,
            'Validity_Diff': float(m1_row['Solution_Level_Validity']) - float(m2_row['Solution_Level_Validity']),
            'Redundancy_Diff': float(m1_row['Solution_Level_Redundancy']) - float(m2_row['Solution_Level_Redundancy']),
            'Shepherd_Diff': float(m1_row['Solution_Score_Shepherd']) - float(m2_row['Solution_Score_Shepherd'])
        })
    
    diff_df = pd.DataFrame(diff_data)
    
    # Plot differences
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Validity difference
    sns.barplot(x='Temperature', y='Validity_Diff', data=diff_df, ax=axes[0])
    axes[0].axhline(y=0, color='r', linestyle='-', alpha=0.3)
    axes[0].set_title(f'Validity: {models[0]} vs {models[1]}', fontsize=13)
    axes[0].set_ylabel('Score Difference')
    
    # Redundancy difference
    sns.barplot(x='Temperature', y='Redundancy_Diff', data=diff_df, ax=axes[1])
    axes[1].axhline(y=0, color='r', linestyle='-', alpha=0.3)
    axes[1].set_title(f'Redundancy: {models[0]} vs {models[1]}', fontsize=13)
    axes[1].set_ylabel('Score Difference')
    
    # Shepherd difference
    sns.barplot(x='Temperature', y='Shepherd_Diff', data=diff_df, ax=axes[2])
    axes[2].axhline(y=0, color='r', linestyle='-', alpha=0.3)
    axes[2].set_title(f'Shepherd: {models[0]} vs {models[1]}', fontsize=13)
    axes[2].set_ylabel('Score Difference')
    
    # Add value annotations on bars
    for ax in axes:
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2., height + (0.005 if height >= 0 else -0.02),
                   f'{height:.3f}', ha='center', fontsize=9)
    
    plt.suptitle(f'Performance Difference: {models[0]} - {models[1]}', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created model difference plot: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize temperature evaluation results")
    parser.add_argument("--input_dir", type=str, default="/home/dazhou/ReasonEval/evaluation_results",
                       help="Directory containing aggregated CSV files")
    parser.add_argument("--output_dir", type=str, default="/home/dazhou/ReasonEval/evaluation_results/visualizations",
                       help="Directory to save visualizations")
    parser.add_argument("--models", type=str, nargs='+', 
                       default=['Abel-7B-002', 'WizardMath-7B-V1.1'],
                       help="Models to include in visualizations")
    parser.add_argument("--detailed", action="store_true", 
                       help="Create detailed distribution visualizations")
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
        
        # Add new visualizations that directly compare models
        create_combined_model_comparison(df,
                                        os.path.join(args.output_dir, 'combined_model_comparison.png'))
        
        create_radar_comparison(df,
                               os.path.join(args.output_dir, 'radar_model_comparison.png'))
        
        create_metric_scatter_matrix(df,
                                   os.path.join(args.output_dir, 'metric_scatter_matrix.png'))
        
        create_temperature_boxplot_comparison(df,
                                           os.path.join(args.output_dir, 'temperature_boxplot_comparison.png'))
        
        if len(args.models) == 2:
            create_model_difference_plot(df, os.path.join(args.output_dir, 'model_difference_plot.png'))
        
        # Create score distribution plots
        create_score_distribution_plots(args.input_dir, args.output_dir, args.models)
        
        # Create a summary table
        summary = []
        for model in df['Model'].unique():
            model_data = df[df['Model'] == model]
            best_temp_idx = model_data['Solution_Score_Shepherd'].idxmax()
            best_temp_row = model_data.loc[best_temp_idx]
            
            summary.append({
                'Model': model,
                'Best Temperature': best_temp_row['Temperature'],
                'Best Shepherd Score': best_temp_row['Solution_Score_Shepherd'],
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
