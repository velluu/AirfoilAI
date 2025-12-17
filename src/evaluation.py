"""
Model evaluation and metrics module
Comprehensive evaluation with multiple metrics and comparison tables
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
from pathlib import Path
from src.config import METRICS_DIR, TABLES_DIR


def evaluate_model(model, scaler, X_train, y_train, X_test, y_test, model_name="Model"):
    """
    Comprehensive model evaluation
    
    Args:
        model: Trained model
        scaler: Fitted StandardScaler
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_name: Name for display
        
    Returns:
        Dictionary with all metrics
    """
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # R² Score
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    
    # Adjusted R²
    n_samples = len(y_test)
    n_features = X_test.shape[1]
    adj_r2_test = 1 - (1 - r2_test) * (n_samples - 1) / (n_samples - n_features - 1)
    
    # MAE and RMSE
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # MAPE (%)
    mape_test = np.mean(np.abs((y_test - y_test_pred) / (y_test + 1e-8))) * 100
    
    # Overfitting gap
    gap = r2_train - r2_test
    
    return {
        'Model': model_name,
        'R² Train': r2_train,
        'R² Test': r2_test,
        'Adj R² Test': adj_r2_test,
        'MAE Train': mae_train,
        'MAE Test': mae_test,
        'RMSE Train': rmse_train,
        'RMSE Test': rmse_test,
        'MAPE% Test': mape_test,
        'Overfitting Gap': gap
    }


def create_comparison_table(results_list, run_id):
    """
    Create comprehensive comparison table
    
    Args:
        results_list: List of evaluation result dictionaries
        run_id: Unique run identifier
        
    Returns:
        DataFrame with sorted results
    """
    df = pd.DataFrame(results_list)
    
    # Sort by R² Test descending
    df = df.sort_values('R² Test', ascending=False).reset_index(drop=True)
    
    # Add rank column
    df.insert(0, 'Rank', range(1, len(df) + 1))
    
    # Save to CSV
    table_path = TABLES_DIR / f'model_comparison_{run_id}.csv'
    df.to_csv(table_path, index=False, float_format='%.4f')
    
    print(f"\n✓ Saved comparison table: {table_path}")
    
    return df


def save_metrics_log(results_df, run_id, metadata=None):
    """
    Save detailed metrics log to ideas/metrics folder
    
    Args:
        results_df: Results dataframe
        run_id: Unique run identifier
        metadata: Optional dict with run information
    """
    log_path = METRICS_DIR / f'metrics_{run_id}.txt'
    
    with open(log_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("AIRFOILAI - MODEL COMPARISON METRICS\n")
        f.write("="*80 + "\n")
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        if metadata:
            f.write("\nRun Configuration:\n")
            for key, value in metadata.items():
                f.write(f"  {key}: {value}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("MODEL COMPARISON RESULTS\n")
        f.write("="*80 + "\n\n")
        
        # Write full table
        f.write(results_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
        
        f.write("\n\n" + "="*80 + "\n")
        f.write("TOP 3 MODELS\n")
        f.write("="*80 + "\n\n")
        
        # Write top 3 with details
        for idx, row in results_df.head(3).iterrows():
            f.write(f"{row['Rank']}. {row['Model']}\n")
            f.write(f"   R² Test: {row['R² Test']:.4f}\n")
            f.write(f"   Adj R² Test: {row['Adj R² Test']:.4f}\n")
            f.write(f"   MAE Test: {row['MAE Test']:.4f}\n")
            f.write(f"   MAPE% Test: {row['MAPE% Test']:.2f}%\n")
            f.write(f"   Overfitting Gap: {row['Overfitting Gap']:.4f}\n")
            f.write("\n")
        
        f.write("="*80 + "\n")
        f.write("KEY INSIGHTS\n")
        f.write("="*80 + "\n\n")
        
        best_model = results_df.iloc[0]
        worst_model = results_df.iloc[-1]
        
        f.write(f"• Best performing model: {best_model['Model']}\n")
        f.write(f"  - Achieved R² = {best_model['R² Test']:.4f} on test set\n")
        f.write(f"  - MAPE = {best_model['MAPE% Test']:.2f}%\n")
        f.write(f"  - Overfitting gap = {best_model['Overfitting Gap']:.4f}\n\n")
        
        f.write(f"• Baseline (Linear Regression): ")
        linear_row = results_df[results_df['Model'] == 'Linear Regression']
        if not linear_row.empty:
            f.write(f"R² = {linear_row.iloc[0]['R² Test']:.4f}\n")
        else:
            f.write("Not evaluated\n")
        
        f.write(f"\n• Improvement over baseline: ")
        if not linear_row.empty:
            improvement = best_model['R² Test'] - linear_row.iloc[0]['R² Test']
            f.write(f"{improvement:.4f} ({improvement/linear_row.iloc[0]['R² Test']*100:.1f}%)\n")
        else:
            f.write("N/A\n")
        
        f.write(f"\n• Models with low overfitting (gap < 0.05):\n")
        low_overfit = results_df[results_df['Overfitting Gap'] < 0.05]
        for _, row in low_overfit.iterrows():
            f.write(f"  - {row['Model']}: gap = {row['Overfitting Gap']:.4f}\n")
    
    print(f"✓ Saved metrics log: {log_path}")
    return log_path


def print_evaluation_summary(results_df):
    """Print formatted summary to console"""
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80 + "\n")
    
    # Print top 5
    print(results_df[['Rank', 'Model', 'R² Test', 'MAE Test', 'MAPE% Test', 'Overfitting Gap']].head(5).to_string(index=False))
    
    print("\n" + "-"*80)
    print(f"Best Model: {results_df.iloc[0]['Model']}")
    print(f"  R² Test: {results_df.iloc[0]['R² Test']:.4f}")
    print(f"  MAE Test: {results_df.iloc[0]['MAE Test']:.4f}")
    print(f"  MAPE: {results_df.iloc[0]['MAPE% Test']:.2f}%")
    print("="*80 + "\n")
