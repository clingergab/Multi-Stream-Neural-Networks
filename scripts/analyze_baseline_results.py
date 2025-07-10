#!/usr/bin/env python3
"""
MCResNet Baseline Grid Search Results Analyzer

This script analyzes the results from the baseline grid search and creates
visualizations to understand which hyperparameters work best.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

def load_latest_results(results_dir: str = "mcresnet_baseline_grid_search"):
    """Load the most recent grid search results."""
    results_path = Path(results_dir)
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory {results_dir} not found")
    
    # Find the most recent results file
    json_files = list(results_path.glob("baseline_grid_search_results_*.json"))
    if not json_files:
        raise FileNotFoundError(f"No results files found in {results_dir}")
    
    latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading results from: {latest_file}")
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    return results, latest_file

def create_results_dataframe(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert results to pandas DataFrame for analysis."""
    data = []
    
    for result in results:
        if 'error' not in result['metrics']:
            row = result['parameters'].copy()
            row.update(result['metrics'])
            row['combination_id'] = result['combination_id']
            row['duration_minutes'] = result.get('duration_minutes', 0)
            data.append(row)
    
    return pd.DataFrame(data)

def plot_parameter_effects(df: pd.DataFrame, save_dir: str = "plots"):
    """Create plots showing parameter effects."""
    Path(save_dir).mkdir(exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Learning rate vs accuracy
    sns.boxplot(data=df, x='learning_rate', y='final_val_accuracy', ax=axes[0, 0])
    axes[0, 0].set_title('Learning Rate vs Validation Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Batch size vs accuracy
    sns.boxplot(data=df, x='batch_size', y='final_val_accuracy', ax=axes[0, 1])
    axes[0, 1].set_title('Batch Size vs Validation Accuracy')
    
    # Plot 3: Optimizer comparison
    sns.boxplot(data=df, x='optimizer', y='final_val_accuracy', ax=axes[0, 2])
    axes[0, 2].set_title('Optimizer vs Validation Accuracy')
    
    # Plot 4: Weight decay vs accuracy
    sns.boxplot(data=df, x='weight_decay', y='final_val_accuracy', ax=axes[1, 0])
    axes[1, 0].set_title('Weight Decay vs Validation Accuracy')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 5: Scheduler vs accuracy
    sns.boxplot(data=df, x='scheduler', y='final_val_accuracy', ax=axes[1, 1])
    axes[1, 1].set_title('Scheduler vs Validation Accuracy')
    
    # Plot 6: Transform (augmentation) vs accuracy
    sns.boxplot(data=df, x='transform', y='final_val_accuracy', ax=axes[1, 2])
    axes[1, 2].set_title('Data Augmentation vs Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/parameter_effects.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    
    # Convert categorical variables to numeric for correlation
    df_numeric = df.copy()
    df_numeric['optimizer_num'] = df_numeric['optimizer'].map({'sgd': 0, 'adamw': 1})
    df_numeric['scheduler_num'] = df_numeric['scheduler'].map({'cosine': 0, 'oneCycle': 1})
    df_numeric['transform_num'] = df_numeric['transform'].astype(int)
    
    # Select numeric columns for correlation
    numeric_cols = ['learning_rate', 'batch_size', 'weight_decay', 'optimizer_num', 
                   'scheduler_num', 'transform_num', 'final_val_accuracy', 'final_val_loss']
    corr_matrix = df_numeric[numeric_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5)
    plt.title('Parameter Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f"{save_dir}/correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {save_dir}/")

def analyze_parameter_importance(df: pd.DataFrame) -> Dict[str, float]:
    """Analyze which parameters have the most impact on performance."""
    param_importance = {}
    
    categorical_params = ['optimizer', 'scheduler', 'transform']
    numerical_params = ['learning_rate', 'batch_size', 'weight_decay']
    
    # For categorical parameters, calculate range of mean accuracies
    for param in categorical_params:
        param_means = df.groupby(param)['final_val_accuracy'].mean()
        param_importance[param] = param_means.max() - param_means.min()
    
    # For numerical parameters, calculate correlation with accuracy
    for param in numerical_params:
        correlation = abs(df[param].corr(df['final_val_accuracy']))
        param_importance[param] = correlation
    
    return param_importance

def main():
    """Main analysis function."""
    print("MCResNet Baseline Grid Search Results Analysis")
    print("=" * 50)
    
    try:
        # Load results
        results, results_file = load_latest_results()
        print(f"Loaded {len(results)} experiment combinations")
        
        # Convert to DataFrame
        df = create_results_dataframe(results)
        print(f"Successfully processed {len(df)} combinations")
        print(f"Failed combinations: {len(results) - len(df)}")
        
        if len(df) == 0:
            print("No successful experiments found!")
            return
        
        # Summary statistics
        print("\n" + "=" * 50)
        print("SUMMARY STATISTICS")
        print("=" * 50)
        print(f"Best validation accuracy: {df['final_val_accuracy'].max():.4f}")
        print(f"Best validation loss: {df['final_val_loss'].min():.4f}")
        print(f"Average accuracy: {df['final_val_accuracy'].mean():.4f} ± {df['final_val_accuracy'].std():.4f}")
        print(f"Average loss: {df['final_val_loss'].mean():.4f} ± {df['final_val_loss'].std():.4f}")
        print(f"Average training time: {df['duration_minutes'].mean():.1f} ± {df['duration_minutes'].std():.1f} minutes")
        
        # Top configurations
        print("\n" + "=" * 50)
        print("TOP 5 CONFIGURATIONS BY ACCURACY")
        print("=" * 50)
        top_5 = df.nlargest(5, 'final_val_accuracy')
        for i, (idx, row) in enumerate(top_5.iterrows(), 1):
            print(f"{i}. Accuracy: {row['final_val_accuracy']:.4f}, Loss: {row['final_val_loss']:.4f}")
            print(f"   LR: {row['learning_rate']}, Batch: {row['batch_size']}, Opt: {row['optimizer']}")
            print(f"   WD: {row['weight_decay']}, Sched: {row['scheduler']}, Aug: {row['transform']}")
            print(f"   Duration: {row['duration_minutes']:.1f} min")
            print()
        
        # Parameter importance
        importance = analyze_parameter_importance(df)
        print("=" * 50)
        print("PARAMETER IMPORTANCE (higher = more impact)")
        print("=" * 50)
        for param, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"{param:15}: {score:.4f}")
        
        # Create visualizations
        plot_parameter_effects(df)
        
        # Save summary to CSV
        summary_file = "grid_search_summary.csv"
        df.to_csv(summary_file, index=False)
        print(f"\nDetailed results saved to: {summary_file}")
        
        # Save top configurations to JSON
        top_configs = {
            'top_5_by_accuracy': [
                {
                    'rank': i+1,
                    'parameters': row.to_dict()
                }
                for i, (_, row) in enumerate(top_5.iterrows())
            ],
            'parameter_importance': importance,
            'summary_stats': {
                'best_accuracy': float(df['final_val_accuracy'].max()),
                'best_loss': float(df['final_val_loss'].min()),
                'mean_accuracy': float(df['final_val_accuracy'].mean()),
                'std_accuracy': float(df['final_val_accuracy'].std()),
                'total_successful': len(df),
                'total_failed': len(results) - len(df)
            }
        }
        
        with open('top_configurations.json', 'w') as f:
            json.dump(top_configs, f, indent=2)
        
        print("Analysis completed successfully!")
        print("Files generated:")
        print("- grid_search_summary.csv")
        print("- top_configurations.json")
        print("- plots/parameter_effects.png")
        print("- plots/correlation_matrix.png")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
