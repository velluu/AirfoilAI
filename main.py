"""
Main Pipeline - AirfoilAI Model Comparison Study
Complete workflow from data loading to final results
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.config import ensure_directories, get_run_id
from src.data_loader import load_airfrans_data, get_dataset_statistics
from src.feature_extraction import extract_dataset_features, prepare_train_test_split
from src.models import ModelRegistry, train_model_with_scaling
from src.evaluation import evaluate_model, create_comparison_table, save_metrics_log, print_evaluation_summary
from src.visualization import create_all_visualizations


def main():
    """Execute complete ML comparison pipeline"""
    
    print("="*80)
    print("AIRFOILAI - COMPREHENSIVE ML MODEL COMPARISON")
    print("="*80)
    print()
    
    # Generate unique run ID
    run_id = get_run_id()
    print(f"Run ID: {run_id}\n")
    
    # Ensure all directories exist
    ensure_directories()
    
    # ========================================================================
    # STEP 1: LOAD DATA
    # ========================================================================
    print("STEP 1: Loading AirfRANS dataset...")
    print("-" * 80)
    
    train_data, test_data = load_airfrans_data()
    
    train_stats = get_dataset_statistics(train_data)
    test_stats = get_dataset_statistics(test_data)
    
    print("\nDataset Statistics:")
    print(f"  Training: {train_stats['total_samples']} samples")
    print(f"    - NACA-3: {train_stats['naca_3_series']}")
    print(f"    - NACA-4: {train_stats['naca_4_series']}")
    print(f"  Test: {test_stats['total_samples']} samples")
    print(f"    - NACA-3: {test_stats['naca_3_series']}")
    print(f"    - NACA-4: {test_stats['naca_4_series']}")
    print()
    
    # ========================================================================
    # STEP 2: FEATURE EXTRACTION
    # ========================================================================
    print("STEP 2: Extracting features and computing L/D ratios...")
    print("-" * 80)
    
    df_train = extract_dataset_features(train_data, "training set")
    df_test = extract_dataset_features(test_data, "test set")
    
    X_train, X_test, y_train, y_test = prepare_train_test_split(df_train, df_test)
    
    # ========================================================================
    # STEP 3: TRAIN AND EVALUATE ALL MODELS
    # ========================================================================
    print("\nSTEP 3: Training and evaluating all models...")
    print("-" * 80)
    
    models = ModelRegistry.get_all_baseline_models()
    results_list = []
    trained_models = []
    
    for model, model_name in models:
        print(f"\nTraining: {model_name}")
        
        try:
            trained_model, scaler, X_test_scaled = train_model_with_scaling(
                model, X_train, y_train, X_test
            )
            
            results = evaluate_model(
                trained_model, scaler, 
                X_train, y_train, X_test, y_test, 
                model_name
            )
            
            results_list.append(results)
            trained_models.append((trained_model, scaler, model_name))
            
            print(f"  R² Test: {results['R² Test']:.4f}")
            print(f"  MAE Test: {results['MAE Test']:.4f}")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            continue
    
    # ========================================================================
    # STEP 4: CREATE COMPARISON TABLE
    # ========================================================================
    print("\nSTEP 4: Creating comparison tables and metrics...")
    print("-" * 80)
    
    results_df = create_comparison_table(results_list, run_id)
    
    # Save detailed metrics log
    metadata = {
        'Train Samples': len(X_train),
        'Test Samples': len(X_test),
        'Features': X_train.shape[1],
        'Models Evaluated': len(results_list)
    }
    save_metrics_log(results_df, run_id, metadata)
    
    # Print summary
    print_evaluation_summary(results_df)
    
    # ========================================================================
    # STEP 5: CREATE VISUALIZATIONS
    # ========================================================================
    print("\nSTEP 5: Generating visualizations...")
    print("-" * 80)
    
    # Get best model
    best_idx = results_df.iloc[0]['Rank'] - 1
    best_model, best_scaler, best_name = trained_models[best_idx]
    
    create_all_visualizations(
        results_df, best_model, best_scaler, 
        X_test, y_test, best_name, run_id
    )
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\nRun ID: {run_id}")
    print(f"Best Model: {results_df.iloc[0]['Model']}")
    print(f"R² Test: {results_df.iloc[0]['R² Test']:.4f}")
    print(f"\nResults saved in:")
    print(f"  - Metrics: ideas/metrics/metrics_{run_id}.txt")
    print(f"  - Tables: results/tables/model_comparison_{run_id}.csv")
    print(f"  - Figures: results/figures/*_{run_id}.png")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
