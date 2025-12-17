"""
Visualization module
Creates publication-quality figures for model comparison
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import FIGURES_DIR

sns.set_style('whitegrid')
sns.set_palette('husl')


def plot_model_comparison(results_df, run_id, metric='R² Test'):
    """
    Create bar chart comparing models by metric
    
    Args:
        results_df: Results dataframe
        run_id: Run identifier for filename
        metric: Metric to plot
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Sort by metric
    df_sorted = results_df.sort_values(metric, ascending=True)
    
    colors = sns.color_palette('RdYlGn', len(df_sorted))
    
    ax.barh(df_sorted['Model'], df_sorted[metric], color=colors)
    ax.set_xlabel(metric, fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title(f'Model Comparison: {metric}', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, (idx, row) in enumerate(df_sorted.iterrows()):
        ax.text(row[metric], i, f' {row[metric]:.3f}', 
                va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    fig_path = FIGURES_DIR / f'comparison_{metric.replace(" ", "_").lower()}_{run_id}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved figure: {fig_path}")
    return fig_path


def plot_overfitting_analysis(results_df, run_id):
    """
    Create scatter plot showing train vs test R² (overfitting analysis)
    
    Args:
        results_df: Results dataframe
        run_id: Run identifier
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.scatter(results_df['R² Train'], results_df['R² Test'], 
               s=100, alpha=0.6, edgecolors='black', linewidth=1.5)
    
    # Add diagonal line (perfect generalization)
    lim_min = min(results_df['R² Train'].min(), results_df['R² Test'].min()) - 0.05
    lim_max = max(results_df['R² Train'].max(), results_df['R² Test'].max()) + 0.05
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 
            'r--', linewidth=2, label='Perfect Generalization', alpha=0.7)
    
    # Add labels for top models
    for idx, row in results_df.head(5).iterrows():
        ax.annotate(row['Model'], 
                   (row['R² Train'], row['R² Test']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
    
    ax.set_xlabel('R² Train', fontsize=12, fontweight='bold')
    ax.set_ylabel('R² Test', fontsize=12, fontweight='bold')
    ax.set_title('Overfitting Analysis: Train vs Test R²', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    fig_path = FIGURES_DIR / f'overfitting_analysis_{run_id}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved figure: {fig_path}")
    return fig_path


def plot_prediction_scatter(model, scaler, X_test, y_test, model_name, run_id):
    """
    Create scatter plot of predicted vs actual L/D values
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        X_test, y_test: Test data
        model_name: Model name
        run_id: Run identifier
    """
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    ax.scatter(y_test, y_pred, alpha=0.5, s=20)
    
    # Perfect prediction line
    lim_min = min(y_test.min(), y_pred.min()) - 1
    lim_max = max(y_test.max(), y_pred.max()) + 1
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 
            'r--', linewidth=2, label='Perfect Prediction')
    
    # Calculate R²
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test, y_pred)
    
    ax.set_xlabel('Actual L/D', fontsize=12, fontweight='bold')
    ax.set_ylabel('Predicted L/D', fontsize=12, fontweight='bold')
    ax.set_title(f'Prediction Quality: {model_name}\nR² = {r2:.4f}', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('=', '')
    fig_path = FIGURES_DIR / f'predictions_{safe_name}_{run_id}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved figure: {fig_path}")
    return fig_path


def plot_regularization_impact(results_df, run_id, model_family='Lasso'):
    """
    Plot impact of regularization parameter on performance
    
    Args:
        results_df: Results dataframe
        run_id: Run identifier
        model_family: Model family name (Lasso, Ridge, MLP)
    """
    family_results = results_df[results_df['Model'].str.contains(model_family)]
    
    if len(family_results) < 2:
        print(f"⚠ Not enough {model_family} results for regularization plot")
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Extract alpha values from model names
    alphas = []
    r2_train = []
    r2_test = []
    gaps = []
    
    for idx, row in family_results.iterrows():
        try:
            alpha_str = row['Model'].split('α=')[1].split(')')[0]
            alphas.append(float(alpha_str))
            r2_train.append(row['R² Train'])
            r2_test.append(row['R² Test'])
            gaps.append(row['Overfitting Gap'])
        except:
            continue
    
    if not alphas:
        print(f"⚠ Could not parse alpha values for {model_family}")
        return None
    
    # Sort by alpha
    sorted_data = sorted(zip(alphas, r2_train, r2_test, gaps))
    alphas, r2_train, r2_test, gaps = zip(*sorted_data)
    
    # Plot 1: R² vs Alpha
    ax1.plot(alphas, r2_train, 'o-', linewidth=2, markersize=8, label='Train R²')
    ax1.plot(alphas, r2_test, 's-', linewidth=2, markersize=8, label='Test R²')
    ax1.set_xscale('log')
    ax1.set_xlabel('Regularization α', fontsize=12, fontweight='bold')
    ax1.set_ylabel('R² Score', fontsize=12, fontweight='bold')
    ax1.set_title(f'{model_family}: R² vs Regularization', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Overfitting Gap vs Alpha
    ax2.plot(alphas, gaps, 'o-', linewidth=2, markersize=8, color='red')
    ax2.axhline(y=0, color='green', linestyle='--', linewidth=2, alpha=0.5, label='No Overfitting')
    ax2.set_xscale('log')
    ax2.set_xlabel('Regularization α', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Overfitting Gap', fontsize=12, fontweight='bold')
    ax2.set_title(f'{model_family}: Overfitting vs Regularization', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    fig_path = FIGURES_DIR / f'regularization_{model_family.lower()}_{run_id}.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved figure: {fig_path}")
    return fig_path


def create_all_visualizations(results_df, best_model, best_scaler, X_test, y_test, best_name, run_id):
    """
    Generate all visualizations for a run
    
    Args:
        results_df: Results dataframe
        best_model: Best performing model
        best_scaler: Scaler for best model
        X_test, y_test: Test data
        best_name: Name of best model
        run_id: Run identifier
    """
    print("\nGenerating visualizations...")
    
    plot_model_comparison(results_df, run_id, 'R² Test')
    plot_model_comparison(results_df, run_id, 'MAE Test')
    plot_overfitting_analysis(results_df, run_id)
    plot_prediction_scatter(best_model, best_scaler, X_test, y_test, best_name, run_id)
    
    # Regularization plots if applicable
    for family in ['Lasso', 'Ridge', 'MLP']:
        plot_regularization_impact(results_df, run_id, family)
    
    print("✓ All visualizations complete")
