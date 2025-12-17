from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import DATA_DIR, METRICS_DIR, MODELS_DIR, RAW_DATASET_DIR, generate_run_id
from src.evaluation import create_comparison_table, save_metrics_log
from src.models import ModelRegistry
from src.tabular_data import load_airfrans_tabular_split

def main():
    run_id = generate_run_id()
    print(f"\n=== AirfoilAI Pipeline Started ===")
    print(f"Run ID: {run_id}\n")

    csv_path = DATA_DIR / "airfrans_dataset.csv"
    manifest_path = RAW_DATASET_DIR / "manifest.json"
    task = "full"

    print("Loading tabular AirfRANS dataset...")
    train_df, test_df = load_airfrans_tabular_split(csv_path, manifest_path, task=task)
    print(f"✓ Train rows: {len(train_df)}")
    print(f"✓ Test rows:  {len(test_df)}")

    feature_cols = ["param1", "param2", "param3", "param4", "param5", "param6"]
    target_col = "L_D"

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    preprocessor = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    print("\nTraining models...")
    results = []
    best = {"name": None, "pipeline": None, "r2_test": -np.inf}

    for model, model_name in ModelRegistry.get_all_baseline_models():
        print(f"  Training {model_name}...")
        pipeline = Pipeline(steps=[("pre", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)

        y_train_pred = np.asarray(pipeline.predict(X_train)).reshape(-1)
        y_test_pred = np.asarray(pipeline.predict(X_test)).reshape(-1)

        y_train_np = np.asarray(y_train).reshape(-1)
        y_test_np = np.asarray(y_test).reshape(-1)

        r2_train = float(r2_score(y_train_np, y_train_pred))
        r2_test = float(r2_score(y_test_np, y_test_pred))

        n_samples = len(y_test)
        n_features = X_test.shape[1]
        adj_r2_test = float(
            1 - (1 - r2_test) * (n_samples - 1) / (n_samples - n_features - 1)
        )

        mae_train = float(mean_absolute_error(y_train_np, y_train_pred))
        mae_test = float(mean_absolute_error(y_test_np, y_test_pred))
        rmse_train = float(np.sqrt(mean_squared_error(y_train_np, y_train_pred)))
        rmse_test = float(np.sqrt(mean_squared_error(y_test_np, y_test_pred)))

        mape_test = float(
            np.mean(np.abs((y_test_np - y_test_pred) / (y_test_np + 1e-8))) * 100
        )
        overfit_gap = float(r2_train - r2_test)

        results.append(
            {
                "Model": model_name,
                "R² Train": r2_train,
                "R² Test": r2_test,
                "Adj R² Test": adj_r2_test,
                "MAE Train": mae_train,
                "MAE Test": mae_test,
                "RMSE Train": rmse_train,
                "RMSE Test": rmse_test,
                "MAPE% Test": mape_test,
                "Overfitting Gap": overfit_gap,
            }
        )

        if r2_test > best["r2_test"]:
            best = {"name": model_name, "pipeline": pipeline, "r2_test": r2_test}

    results_df = create_comparison_table(results, run_id)
    print("\n" + "=" * 80)
    print("Results:")
    print("=" * 80)
    print(results_df.to_string(index=False))

    model_path = MODELS_DIR / f"best_model_{run_id}.joblib"
    joblib.dump(best["pipeline"], model_path)
    print(f"\n✓ Saved best model: {model_path}")

    save_metrics_log(
        results_df,
        run_id,
        metadata={
            "task": task,
            "csv": str(csv_path),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "features": ",".join(feature_cols),
            "target": target_col,
            "best_model": best["name"],
        },
    )

    print(f"\n✓ Results saved with run_id: {run_id}")
    print("=" * 80)

if __name__ == "__main__":
    main()
