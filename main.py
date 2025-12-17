from src.config import generate_run_id, DATA_DIR, FIGURES_DIR, TABLES_DIR, METRICS_DIR
from src.data_loader import load_airfrans_data
from src.feature_extraction import extract_features_from_simulations
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import pandas as pd
import numpy as np

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
    
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nTraining models...")
    models = {
        'Linear Regression': LinearRegression(),
        'Lasso': Lasso(alpha=0.1),
        'Ridge': Ridge(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
        'Decision Tree (depth=5)': DecisionTreeRegressor(max_depth=5, random_state=42),
        'Decision Tree (depth=10)': DecisionTreeRegressor(max_depth=10, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42),
        'MLP': MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    }
    
    results = []
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train_scaled, y_train)
        
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        results.append({
            'Model': name,
            'R² Train': r2_train,
            'R² Test': r2_test,
            'MAE Test': mae_test,
            'RMSE Test': rmse_test,
            'Gap': r2_train - r2_test
        })
    
    results_df = pd.DataFrame(results).sort_values('R² Test', ascending=False)
    results_df.to_csv(TABLES_DIR / f'model_comparison_{run_id}.csv', index=False)
    
    print("\n" + "="*80)
    print("Results:")
    print("="*80)
    print(results_df.to_string(index=False))
    
    with open(METRICS_DIR / f'metrics_{run_id}.txt', 'w') as f:
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Train samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n\n")
        f.write(results_df.to_string(index=False))
    
    print(f"\n✓ Results saved with run_id: {run_id}")
    print("="*80)

if __name__ == "__main__":
    main()
