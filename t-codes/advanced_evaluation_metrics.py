import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# import nltk
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk.corpus import stopwords
# import spacy
import warnings
warnings.filterwarnings('ignore')

# # Download necessary NLTK resources
# try:
#     nltk.data.find('tokenizers/punkt')
#     nltk.data.find('corpora/stopwords')
# except LookupError:
#     nltk.download('punkt')
#     nltk.download('stopwords')

# # Load spaCy model for more advanced NLP tasks
# try:
#     nlp = spacy.load("en_core_web_sm")
# except:
#     print("Installing spaCy model...")
#     import subprocess
#     subprocess.call("python -m spacy download en_core_web_sm", shell=True)
#     nlp = spacy.load("en_core_web_sm")

def load_detailed_results(input_file):
    """
    Load detailed results CSV file
    
    Args:
        input_file: Path to the detailed CSV file
        
    Returns:
        DataFrame with detailed results
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File not found: {input_file}")
        
    return pd.read_csv(input_file)

def load_original_solutions(results_dir, model_name, temperature):
    """
    Load the original solutions with reasoning steps
    
    Args:
        results_dir: Directory containing the original JSON result files
        model_name: Name of the model
        temperature: Temperature value
        
    Returns:
        List of dictionaries with original solution data
    """
    filename = os.path.join(results_dir, f"{model_name}_temperature_{temperature}.json")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Original results file not found: {filename}")
        
    with open(filename, 'r') as f:
        results = json.load(f)
    
    return results

def extract_steps(original_data):
    """
    Extract reasoning steps from original data
    
    Args:
        original_data: List of dictionaries with original solution data
        
    Returns:
        Dictionary mapping question UUIDs to lists of reasoning steps
    """
    steps_dict = {}
    
    for item in original_data:
        uuid = item.get("uuid", "unknown")
        steps = item.get("model_output_steps", [])
        
        # Filter out empty steps
        valid_steps = [step for step in steps if step and not step.isspace()]
        
        if valid_steps:
            steps_dict[uuid] = valid_steps
    
    return steps_dict

def find_math_operations(text):
    """Find mathematical operations in text"""
    # Look for common math operations: +, -, *, /, =, <, >, ≤, ≥, etc.
    basic_ops = len(re.findall(r'[\+\-\*\/=<>]', text))
    
    # Look for mathematical functions
    math_funcs = len(re.findall(r'\b(sin|cos|tan|log|ln|sqrt|pow|exp|floor|ceil|round|abs|max|min)\b', text))
    
    # Look for numbers with operations
    num_ops = len(re.findall(r'(\d+\s*[\+\-\*\/]\s*\d+)', text))
    
    return basic_ops + math_funcs + num_ops

def count_uncertainty_markers(text):
    """Count uncertainty/hedging markers in text"""
    uncertainty_words = [
        'probably', 'might', 'may', 'could', 'perhaps', 'possibly', 'likely', 
        'unlikely', 'maybe', 'appears', 'seems', 'approximately', 'around', 
        'roughly', 'estimate', 'guess', 'not sure', 'uncertain'
    ]
    
    count = 0
    text_lower = text.lower()
    
    for word in uncertainty_words:
        count += text_lower.count(word)
    
    return count

def count_confidence_expressions(text):
    """Count confidence/assertive expressions in text"""
    confidence_words = [
        'clearly', 'definitely', 'certainly', 'absolutely', 'obviously', 'undoubtedly',
        'must', 'will', 'always', 'never', 'exactly', 'precisely', 'indeed', 
        'without doubt', 'guaranteed', 'surely', 'undeniable', 'conclusively'
    ]
    
    count = 0
    text_lower = text.lower()
    
    for word in confidence_words:
        count += text_lower.count(word)
    
    return count

def detect_hallucination_markers(text):
    """
    Detect potential hallucination markers in text
    
    This is a simplified heuristic and may not catch all hallucinations
    """
    # Look for phrases that often indicate made-up facts
    hallucination_markers = [
        'according to', 'based on', 'as stated in', 'as mentioned in',
        'studies show', 'research indicates', 'experts say', 'it is known that',
        'it is well established'
    ]
    
    count = 0
    text_lower = text.lower()
    
    for marker in hallucination_markers:
        count += text_lower.count(marker)
    
    return count

def detect_self_contradictions(steps):
    """
    Detect potential self-contradictions between steps
    
    This is a simplified heuristic that looks for negation patterns
    """
    if len(steps) < 2:
        return 0
    
    # Process steps with spaCy for deeper analysis
    doc_list = list(nlp.pipe(steps))
    
    # Extract key statements from each step
    statements = []
    for doc in doc_list:
        step_statements = []
        for sent in doc.sents:
            # Look for sentences with mathematical content
            if any(token.text.isdigit() or token.text in "+-*/=" for token in sent):
                step_statements.append(sent.text)
        statements.append(step_statements)
    
    # Check for contradictions (very simplified approach)
    contradiction_count = 0
    
    # Compare each step with previous steps
    for i in range(1, len(statements)):
        current_text = " ".join(statements[i]).lower()
        
        for j in range(i):
            prev_text = " ".join(statements[j]).lower()
            
            # Check for pattern where step contains negation of earlier statement
            if "not " in current_text and any(stmt in current_text for stmt in statements[j]):
                contradiction_count += 1
                break
            
            # Look for direct numerical contradictions
            current_nums = re.findall(r'(\w+)\s*=\s*(\d+\.?\d*)', current_text)
            prev_nums = re.findall(r'(\w+)\s*=\s*(\d+\.?\d*)', prev_text)
            
            for var1, val1 in current_nums:
                for var2, val2 in prev_nums:
                    if var1 == var2 and val1 != val2:
                        contradiction_count += 1
                        break
    
    return contradiction_count

def calculate_structural_metrics(detailed_df, steps_dict):
    """
    Calculate structural metrics
    
    Args:
        detailed_df: DataFrame with detailed results
        steps_dict: Dictionary mapping UUIDs to reasoning steps
        
    Returns:
        DataFrame with structural metrics by temperature and model
    """
    metrics = []
    
    # Group by model and temperature
    for (model, temp), group in detailed_df.groupby(['Model', 'Temperature']):
        # Get UUIDs for this group
        uuids = group['Question_UUID'].tolist()
        
        step_counts = []
        step_lengths_words = []
        step_lengths_chars = []
        solution_lengths = []
        step_complexity = []
        
        # Process each solution
        for uuid in uuids:
            if uuid in steps_dict:
                steps = steps_dict[uuid]
                
                # Step count
                step_count = len(steps)
                step_counts.append(step_count)
                
                # Step length
                step_lengths_words.extend([len(step.split()) for step in steps])
                step_lengths_chars.extend([len(step) for step in steps])
                
                # Solution length
                solution_length = sum(len(step) for step in steps)
                solution_lengths.append(solution_length)
                
                # Step complexity
                complexity = [find_math_operations(step) for step in steps]
                step_complexity.extend(complexity)
        
        # Calculate averages
        metrics.append({
            'Model': model,
            'Temperature': temp,
            'Avg_Step_Count': np.mean(step_counts) if step_counts else 0,
            'Avg_Step_Length_Words': np.mean(step_lengths_words) if step_lengths_words else 0,
            'Avg_Step_Length_Chars': np.mean(step_lengths_chars) if step_lengths_chars else 0,
            'Avg_Solution_Length': np.mean(solution_lengths) if solution_lengths else 0,
            'Avg_Step_Complexity': np.mean(step_complexity) if step_complexity else 0,
        })
    
    return pd.DataFrame(metrics)

def calculate_diversity_metrics(detailed_df, steps_dict):
    """
    Calculate diversity and consistency metrics
    
    Args:
        detailed_df: DataFrame with detailed results
        steps_dict: Dictionary mapping UUIDs to reasoning steps
        
    Returns:
        DataFrame with diversity metrics by temperature and model
    """
    metrics = []
    
    # Group by model and temperature
    for (model, temp), group in detailed_df.groupby(['Model', 'Temperature']):
        # Get UUIDs for this group
        uuids = group['Question_UUID'].tolist()
        
        lexical_diversity_scores = []
        step_similarity_scores = []
        
        # Process each solution
        for uuid in uuids:
            if uuid in steps_dict:
                steps = steps_dict[uuid]
                
                if len(steps) < 1:
                    continue
                
                # # Lexical diversity
                # all_text = " ".join(steps).lower()
                # tokens = word_tokenize(all_text)
                # # Remove stopwords
                # stop_words = set(stopwords.words('english'))
                # filtered_tokens = [w for w in tokens if w.isalnum() and w not in stop_words]
                
                # # Calculate lexical diversity (unique tokens / total tokens)
                # if filtered_tokens:
                #     lexical_diversity = len(set(filtered_tokens)) / len(filtered_tokens)
                #     lexical_diversity_scores.append(lexical_diversity)
                
                # Step-to-step similarity
                if len(steps) > 1:
                    # Use CountVectorizer to convert steps to token counts
                    vectorizer = CountVectorizer().fit(steps)
                    step_vectors = vectorizer.transform(steps).toarray()
                    
                    # Calculate similarities between consecutive steps
                    similarities = []
                    for i in range(len(steps)-1):
                        sim = cosine_similarity([step_vectors[i]], [step_vectors[i+1]])[0][0]
                        similarities.append(sim)
                    
                    # Average similarity score
                    avg_similarity = np.mean(similarities)
                    step_similarity_scores.append(avg_similarity)
        
        # # Calculate averages
        # metrics.append({
        #     'Model': model,
        #     'Temperature': temp,
        #     'Lexical_Diversity': np.mean(lexical_diversity_scores) if lexical_diversity_scores else 0,
        #     'Step_Similarity': np.mean(step_similarity_scores) if step_similarity_scores else 0,
        # })

        # Calculate averages
        metrics.append({
            'Model': model,
            'Temperature': temp,
            'Step_Similarity': np.mean(step_similarity_scores) if step_similarity_scores else 0,
        })
    
    return pd.DataFrame(metrics)

def calculate_error_metrics(detailed_df, steps_dict):
    """
    Calculate error analysis metrics
    
    Args:
        detailed_df: DataFrame with detailed results
        steps_dict: Dictionary mapping UUIDs to reasoning steps
        
    Returns:
        DataFrame with error metrics by temperature and model
    """
    metrics = []
    
    # Group by model and temperature
    for (model, temp), group in detailed_df.groupby(['Model', 'Temperature']):
        # Get UUIDs for this group
        uuids = group['Question_UUID'].tolist()
        
        hallucination_rates = []
        self_contradiction_rates = []
        
        # Process each solution
        for uuid in uuids:
            if uuid in steps_dict:
                steps = steps_dict[uuid]
                
                if not steps:
                    continue
                
                # Hallucination rate
                hallucination_count = sum(detect_hallucination_markers(step) for step in steps)
                hallucination_rate = hallucination_count / len(steps)
                hallucination_rates.append(hallucination_rate)
                
                # # Self-contradiction rate
                # contradiction_count = detect_self_contradictions(steps)
                # contradiction_rate = contradiction_count / max(1, len(steps) - 1)
                # self_contradiction_rates.append(contradiction_rate)
        
        # # Calculate averages
        # metrics.append({
        #     'Model': model,
        #     'Temperature': temp,
        #     'Hallucination_Rate': np.mean(hallucination_rates) if hallucination_rates else 0,
        #     'Self_Contradiction_Rate': np.mean(self_contradiction_rates) if self_contradiction_rates else 0,
        # })

         # Calculate averages
        metrics.append({
            'Model': model,
            'Temperature': temp,
            'Hallucination_Rate': np.mean(hallucination_rates) if hallucination_rates else 0,
        })
    
    return pd.DataFrame(metrics)

def calculate_reasoning_pattern_metrics(detailed_df, steps_dict):
    """
    Calculate reasoning pattern metrics
    
    Args:
        detailed_df: DataFrame with detailed results
        steps_dict: Dictionary mapping UUIDs to reasoning steps
        
    Returns:
        DataFrame with reasoning pattern metrics by temperature and model
    """
    metrics = []
    
    # Group by model and temperature
    for (model, temp), group in detailed_df.groupby(['Model', 'Temperature']):
        # Get UUIDs for this group
        uuids = group['Question_UUID'].tolist()
        
        abstract_concrete_ratios = []
        uncertainty_counts = []
        confidence_counts = []
        
        # Process each solution
        for uuid in uuids:
            if uuid in steps_dict:
                steps = steps_dict[uuid]
                
                if not steps:
                    continue
                
                # Count math operations (concrete reasoning)
                concrete = sum(find_math_operations(step) for step in steps)
                
                # Count words as proxy for abstract reasoning
                total_words = sum(len(step.split()) for step in steps)
                
                # Calculate abstract-to-concrete ratio
                # Higher ratio means more abstract, lower means more concrete calculations
                if concrete > 0:
                    abstract_concrete_ratio = (total_words - concrete) / concrete
                    abstract_concrete_ratios.append(abstract_concrete_ratio)
                
                # Uncertainty markers
                uncertainty = sum(count_uncertainty_markers(step) for step in steps)
                uncertainty_counts.append(uncertainty / len(steps))
                
                # Confidence expressions
                confidence = sum(count_confidence_expressions(step) for step in steps)
                confidence_counts.append(confidence / len(steps))
        
        # Calculate averages
        metrics.append({
            'Model': model,
            'Temperature': temp,
            'Abstract_Concrete_Ratio': np.mean(abstract_concrete_ratios) if abstract_concrete_ratios else 0,
            'Uncertainty_Markers': np.mean(uncertainty_counts) if uncertainty_counts else 0,
            'Confidence_Expressions': np.mean(confidence_counts) if confidence_counts else 0,
        })
    
    return pd.DataFrame(metrics)

def plot_metrics(metrics_df, output_dir, metric_category):
    """
    Plot metrics by temperature for all models together
    
    Args:
        metrics_df: DataFrame with metrics
        output_dir: Directory to save plots
        metric_category: Category name for the metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if we have data to plot
    if metrics_df.empty:
        print(f"Warning: No data to plot for {metric_category} metrics")
        return
    
    # Exclude Model and Temperature columns
    metric_columns = [col for col in metrics_df.columns if col not in ['Model', 'Temperature']]
    
    # Create subplots
    n_metrics = len(metric_columns)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    plt.figure(figsize=(15, 5 * n_rows))
    
    for i, metric in enumerate(metric_columns):
        plt.subplot(n_rows, n_cols, i+1)
        
        # Check if this metric column exists and has non-null values
        if metric not in metrics_df.columns or metrics_df[metric].isna().all():
            plt.text(0.5, 0.5, f"No data for {metric}", ha='center', va='center')
            continue
        
        # Plot the data, using hue='Model' to distinguish different models
        sns.lineplot(data=metrics_df, x='Temperature', y=metric, 
                     hue='Model', marker='o', linewidth=2)
        
        plt.title(f'{metric} vs Temperature', fontsize=14, fontweight='bold')
        plt.xlabel('Temperature', fontsize=12)
        plt.ylabel(metric.replace('_', ' '), fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add data point values as text annotations
        for model in metrics_df['Model'].unique():
            model_data = metrics_df[metrics_df['Model'] == model]
            for _, row in model_data.iterrows():
                plt.annotate(f"{row[metric]:.2f}", 
                           (row['Temperature'], row[metric]),
                           textcoords="offset points", 
                           xytext=(0,5), 
                           ha='center', 
                           fontsize=8)
    
    # Add a main title for the entire figure
    title = f"{metric_category.capitalize()} Metrics for All Models"
    plt.suptitle(title, fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the suptitle
    plt.savefig(os.path.join(output_dir, f'{metric_category}_metrics.png'))
    plt.close()
    
    print(f"Created {metric_category} metrics plot: {metric_category}_metrics.png")
    
    # Also save individual plots for each metric
    for metric in metric_columns:
        plt.figure(figsize=(10, 6))
        
        # Check if this metric column exists and has non-null values
        if metric not in metrics_df.columns or metrics_df[metric].isna().all():
            plt.text(0.5, 0.5, f"No data for {metric}", ha='center', va='center')
            plt.savefig(os.path.join(output_dir, f'{metric}.png'))
            plt.close()
            continue
            
        # Plot with model differentiation
        sns.lineplot(data=metrics_df, x='Temperature', y=metric, 
                     hue='Model', marker='o', linewidth=2)
        
        # Add data point values as text
        for model in metrics_df['Model'].unique():
            model_data = metrics_df[metrics_df['Model'] == model]
            for _, row in model_data.iterrows():
                plt.annotate(f"{row[metric]:.2f}", 
                           (row['Temperature'], row[metric]),
                           textcoords="offset points", 
                           xytext=(0,5), 
                           ha='center', 
                           fontsize=9)
        
        metric_title = f'{metric.replace("_", " ")} vs Temperature'
        plt.title(metric_title, fontsize=14, fontweight='bold')
        plt.xlabel('Temperature', fontsize=12)
        plt.ylabel(metric.replace('_', ' '), fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric}.png'))
        plt.close()
        
        print(f"Created individual metric plot: {metric}.png")

def plot_correlation_matrix(all_metrics, output_dir):
    """
    Plot correlation matrix between different metrics
    
    Args:
        all_metrics: List of DataFrames with metrics
        output_dir: Directory to save plots
    """
    # Combine all metrics
    combined = all_metrics[0]
    for df in all_metrics[1:]:
        combined = pd.merge(combined, df, on=['Model', 'Temperature'])
    
    # Calculate correlation matrix
    corr_columns = [col for col in combined.columns if col not in ['Model', 'Temperature']]
    corr_matrix = combined[corr_columns].corr()
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Matrix Between Metrics")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_correlation.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Calculate advanced evaluation metrics")
    parser.add_argument("--input_dir", type=str, default="/home/dazhou/ReasonEval/evaluation_results",
                       help="Directory containing detailed CSV files")
    parser.add_argument("--results_dir", type=str, default="/home/dazhou/ReasonEval/temperature_study",
                       help="Directory containing original JSON result files")
    parser.add_argument("--output_dir", type=str, default="/home/dazhou/ReasonEval/evaluation_results/advanced_metrics",
                       help="Directory to save analysis results")
    parser.add_argument("--models", type=str, nargs='+', 
                       default=['Abel-7B-002', 'WizardMath-7B-V1.1'],
                       help="Models to analyze")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load detailed results and original solutions
    all_steps = {}
    
    for model in args.models:
        # Load detailed CSV
        detailed_csv = os.path.join(args.input_dir, f"{model}_detailed.csv")
        
        try:
            detailed_df = load_detailed_results(detailed_csv)
        except FileNotFoundError:
            print(f"Skipping {model}, detailed CSV not found")
            continue
            
        # Get unique temperatures
        temperatures = detailed_df['Temperature'].unique()
        
        # Load original solutions for each temperature
        model_steps = {}
        for temp in temperatures:
            try:
                original_data = load_original_solutions(args.results_dir, model, temp)
                steps_dict = extract_steps(original_data)
                
                # Add steps to the model's dictionary
                model_steps.update(steps_dict)
            except FileNotFoundError:
                print(f"Could not find original solutions for {model} at temperature {temp}")
        
        all_steps[model] = model_steps
    
    results = []
    
    # Calculate metrics for each model
    for model in args.models:
        if model not in all_steps:
            continue
            
        # Load detailed CSV
        detailed_csv = os.path.join(args.input_dir, f"{model}_detailed.csv")
        detailed_df = load_detailed_results(detailed_csv)
        
        # Get steps dict for this model
        steps_dict = all_steps[model]
        
        # Calculate metrics
        structural_metrics = calculate_structural_metrics(detailed_df, steps_dict)
        diversity_metrics = calculate_diversity_metrics(detailed_df, steps_dict)
        error_metrics = calculate_error_metrics(detailed_df, steps_dict)
        reasoning_metrics = calculate_reasoning_pattern_metrics(detailed_df, steps_dict)
        
        # Save metrics CSVs for each model
        structural_metrics.to_csv(os.path.join(args.output_dir, f'{model}_structural_metrics.csv'), index=False)
        diversity_metrics.to_csv(os.path.join(args.output_dir, f'{model}_diversity_metrics.csv'), index=False)
        error_metrics.to_csv(os.path.join(args.output_dir, f'{model}_error_metrics.csv'), index=False)
        reasoning_metrics.to_csv(os.path.join(args.output_dir, f'{model}_reasoning_metrics.csv'), index=False)
        
        # Skip individual model plotting, only combine metrics for collective analysis
        results.append((structural_metrics, diversity_metrics, error_metrics, reasoning_metrics))
    
    # Create combined metrics for all models
    if results:
        all_structural = pd.concat([r[0] for r in results])
        all_diversity = pd.concat([r[1] for r in results])
        all_error = pd.concat([r[2] for r in results])
        all_reasoning = pd.concat([r[3] for r in results])
        
        # Save combined metrics
        all_structural.to_csv(os.path.join(args.output_dir, 'all_structural_metrics.csv'), index=False)
        all_diversity.to_csv(os.path.join(args.output_dir, 'all_diversity_metrics.csv'), index=False)
        all_error.to_csv(os.path.join(args.output_dir, 'all_error_metrics.csv'), index=False)
        all_reasoning.to_csv(os.path.join(args.output_dir, 'all_reasoning_metrics.csv'), index=False)
        
        # Only plot combined metrics for all models
        plot_metrics(all_structural, args.output_dir, 'structural')
        plot_metrics(all_diversity, args.output_dir, 'diversity')
        plot_metrics(all_error, args.output_dir, 'error')
        plot_metrics(all_reasoning, args.output_dir, 'reasoning')
        
        # Plot correlation matrix
        plot_correlation_matrix([all_structural, all_diversity, all_error, all_reasoning], args.output_dir)
    
    print("Advanced metrics calculation complete!")

if __name__ == "__main__":
    main()
