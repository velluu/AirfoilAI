# AirfoilAI - ML Model Comparison for L/D Prediction

Comprehensive machine learning comparison study for predicting airfoil lift-to-drag (L/D) ratios using the AirfRANS CFD dataset.

## Project Structure

```
AirfoilAI/
├── main.py                 # Main pipeline script
├── src/
│   ├── config.py          # Configuration and paths
│   ├── data_loader.py     # AirfRANS data loading
│   ├── feature_extraction.py  # Feature engineering and L/D computation
│   ├── models.py          # Model registry (Linear, Trees, RF, XGBoost, MLP)
│   ├── evaluation.py      # Metrics and comparison tables
│   └── visualization.py   # Publication-quality figures
├── data/
│   ├── raw/              # Raw AirfRANS data
│   └── processed/        # Processed .pkl files
├── results/
│   ├── figures/          # Model comparison plots
│   └── tables/           # CSV comparison tables
├── ideas/
│   ├── metrics/          # Detailed run logs (.txt)
│   └── *.md              # Implementation guides
└── models/               # Saved model checkpoints
```

## Quick Start

### 1. Setup Environment

```bash
conda env create -f environment.yml
conda activate airfoilai
```

### 2. Prepare Data

Place AirfRANS processed data files in `data/processed/`:
- `dataset_full_train.pkl`
- `dataset_full_test.pkl`

### 3. Run Complete Pipeline

```bash
python main.py
```

This will:
1. Load AirfRANS CFD simulations (800 train, 200 test)
2. Extract features (Reynolds, AoA, NACA parameters) and compute L/D
3. Train 9 models: Linear, Lasso, Ridge, ElasticNet, DecisionTree, RandomForest, GradientBoosting, XGBoost, MLP
4. Generate comparison tables and metrics
5. Create visualizations

## Output Files

Each run generates unique timestamped files:

- **Metrics Log**: `ideas/metrics/metrics_YYYYMMDD_HHMMSS.txt`
- **Comparison Table**: `results/tables/model_comparison_YYYYMMDD_HHMMSS.csv`
- **Figures**: `results/figures/*_YYYYMMDD_HHMMSS.png`

## Models Compared

| Category | Models | Key Focus |
|----------|--------|-----------|
| **Linear** | Linear Regression, Lasso, Ridge, ElasticNet | Baseline, L1/L2 regularization |
| **Trees** | Decision Tree, Random Forest | Nonlinearity, ensemble learning |
| **Boosting** | Gradient Boosting, XGBoost | Sequential learning, performance |
| **Neural** | MLP (64→32→1) | Deep learning, early stopping |

## Requirements

Python 3.11+, scikit-learn, xgboost, torch, numpy, pandas, matplotlib, seaborn