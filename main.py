from src.config import generate_run_id, DATA_DIR, FIGURES_DIR, TABLES_DIR, METRICS_DIR
from src.data_loader import load_airfrans_data
from src.feature_extraction import extract_features_from_simulations
from src.models import ModelTrainer
from src.evaluation import ModelEvaluator
from src.visualization import create_comparison_plots, plot_learning_curves, plot_feature_importance, plot_prediction_scatter
from sklearn.preprocessing import StandardScaler
import pandas as pd

def main():
    run_id = generate_run_id()
    print(f"\n=== AirfoilAI Pipeline Started ===")
    print(f"Run ID: {run_id}\n")
    
    print("Loading AirfRANS data...")
    (train_sims, train_names), (test_sims, test_names) = load_airfrans_data(DATA_DIR)
    
    print("Extracting features...")
    train_df = extract_features_from_simulations(train_sims, train_names)
    test_df = extract_features_from_simulations(test_sims, test_names)
    
    feature_cols = ['Uinf', 'AoA', 'NACA_1', 'NACA_2', 'NACA_3', 'NACA_4',
                    'mean_pressure', 'std_pressure', 'mean_velocity', 'max_velocity']
    
    X_train = train_df[feature_cols]
    y_train = train_df['L_D']
    X_test = test_df[feature_cols]
    y_test = test_df['L_D']
    
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
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training models...")
    trainer = ModelTrainer()
    models = trainer.train_all_models(X_train_scaled, y_train)
    
    print("Evaluating models...")
    evaluator = ModelEvaluator()
    results = {}
    for name, model in models.items():
        results[name] = evaluator.evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
    
    comparison_table = evaluator.create_comparison_table(results)
    comparison_table.to_csv(TABLES_DIR / f'model_comparison_{run_id}.csv', index=False)
    
    evaluator.save_metrics_log(results, METRICS_DIR / f'metrics_{run_id}.txt')
    
    print("Creating visualizations...")
    create_comparison_plots(comparison_table, FIGURES_DIR, run_id)
    plot_learning_curves(models, X_train_scaled, y_train, FIGURES_DIR, run_id)
    plot_feature_importance(models, feature_cols, FIGURES_DIR, run_id)
    plot_prediction_scatter(models, X_test_scaled, y_test, FIGURES_DIR, run_id)
    
    print(f"\\nResults saved with run_id: {run_id}")
    print("="*80)

if __name__ == "__main__":
    main()
