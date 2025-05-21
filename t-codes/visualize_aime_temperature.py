#!/usr/bin/env python3
import os
import argparse
import sys
from visualize_temperature_results import (
    load_aggregated_results, 
    create_metric_comparison_plot,
    create_heatmap_visualization, 
    create_optimal_temperature_plot,
    create_trade_off_plot,
    create_score_distribution_plots,
    prepare_output_directories,
    analyze_by_dataset
)

def main():
    """
    Main function to visualize temperature evaluation results specifically for
    Qwen models on the AIME dataset.
    """
    parser = argparse.ArgumentParser(description="Visualize temperature evaluation results for Qwen on AIME dataset")
    parser.add_argument("--input_dir", type=str, default="/home/dazhou/ReasonEval/evaluation_results",
                       help="Directory containing aggregated CSV files")
    parser.add_argument("--output_dir", type=str, default="/home/dazhou/ReasonEval/evaluation_results/aime_visualizations",
                       help="Directory to save visualizations")
    parser.add_argument("--models", type=str, nargs='+', 
                       default=['Qwen3-32B'],
                       help="Models to include in visualizations (default: Qwen3-32B)")
    parser.add_argument("--datasets", type=str, nargs='+', 
                       default=['aime'],
                       help="Datasets to include in visualizations (default: aime)")
    parser.add_argument("--evaluators", type=str, nargs='+',
                       default=['ReasonEval_7B', 'ReasonEval_34B'],
                       help="Evaluation models to include (default: ReasonEval_7B and ReasonEval_34B)")
    parser.add_argument("--temperatures", type=float, nargs='+',
                       default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
                       help="Temperatures to analyze")
    parser.add_argument("--detailed", action="store_true", 
                       help="Create detailed distribution visualizations")
    parser.add_argument("--custom_output_name", type=str, default="",
                       help="Custom name for output files")

    args = parser.parse_args()
    
    # Prepare output directories
    dataset_dir = prepare_output_directories(args)
    
    # Construct paths to aggregated files
    agg_files = []

    # Use provided models to construct paths
    for model in args.models:
        for dataset in args.datasets:
            for evaluator in args.evaluators:
                potential_path = os.path.join(args.input_dir, model, dataset, evaluator, "aggregated.csv")
                if os.path.exists(potential_path):
                    agg_files.append(potential_path)
                    print(f"Found aggregated file: {potential_path}")
                else:
                    print(f"Warning: Aggregated file not found: {potential_path}")
    
    if not agg_files:
        print("No aggregated files found. Please check your input directory and model names.")
        return
    
    try:
        # Load and combine data
        df = load_aggregated_results(agg_files, 
                                    output_path=os.path.join(args.output_dir, f"{args.custom_output_name or 'aime'}_summary.csv"))
        
        # Filter by specified temperatures if provided
        if args.temperatures:
            df = df[df['Temperature'].isin(args.temperatures)]
            if df.empty:
                print(f"No data found for specified temperatures: {args.temperatures}")
                return
        
        # Create visual output directory by model
        for model in args.models:
            model_df = df[df['Model'] == model]
            if model_df.empty:
                continue
                
            model_dir = os.path.join(args.output_dir, model)
            os.makedirs(model_dir, exist_ok=True)
            
            # Create combined plots for this model
            create_metric_comparison_plot(
                model_df, 
                os.path.join(model_dir, f'{args.custom_output_name or "aime"}_temperature_metrics_comparison.png')
            )
            
            create_heatmap_visualization(
                model_df,
                os.path.join(model_dir, f'{args.custom_output_name or "aime"}_temperature_metrics_heatmap.png')
            )
            
            create_optimal_temperature_plot(
                model_df,
                os.path.join(model_dir, f'{args.custom_output_name or "aime"}_optimal_temperature.png')
            )
            
            create_trade_off_plot(
                model_df,
                os.path.join(model_dir, f'{args.custom_output_name or "aime"}_metrics_tradeoff.png')
            )
        
        # Analyze data by dataset
        analyze_by_dataset(df, args, dataset_dir)
        
        # Create score distribution plots if detailed flag is set
        if args.detailed:
            detailed_dir = os.path.join(args.output_dir, 'distributions')
            os.makedirs(detailed_dir, exist_ok=True)
            
            create_score_distribution_plots(
                args.input_dir, 
                detailed_dir,
                args.models,
                args.datasets
            )
        
        print("Visualization complete!")
        print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
