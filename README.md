# AirfoilAI - ML Model Comparison for L/D Prediction

Comprehensive machine learning comparison study for predicting airfoil lift-to-drag (L/D) ratios using the PALMO (OVERFLOW Machine Learning Airfoil Performance) dataset from NASA.

## Project Structure

```
AirfoilAI/
├── main.py                 # Main pipeline script
├── src/
│   ├── config.py          # Configuration and paths
│   ├── data_loader.py     # PALMO data loading
│   ├── feature_extraction.py  # Feature preparation (log_Re transform)
│   ├── models.py          # Model registry (Linear, Trees, RF, XGBoost, MLP)
│   ├── evaluation.py      # Metrics and comparison tables
│   └── visualization.py   # Publication-quality figures
├── data/
│   └── raw/              # PALMO coefficient files
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

Download PALMO coefficient files from NASA and place in `data/raw/`:
- `PALMO_NACA_4_series_cl.txt`
- `PALMO_NACA_4_series_cd.txt`
- `PALMO_NACA_4_series_cm.txt`

**Source**: https://ntrs.nasa.gov/citations/20240014546 (NASA Technical Report)

### 3. Run Complete Pipeline

```bash
python main.py
```

This will:
1. Load PALMO dataset (52,480 total CFD simulations)
   - Train: 12 airfoils × 3,280 conditions = 39,360 samples
   - Test: 4 airfoils × 3,280 conditions = 13,120 samples
2. Prepare features: camber, camber_pos, thickness, Mach, log_Re, alpha → L/D
3. Train 9 models: Linear, Lasso, Ridge, ElasticNet, DecisionTree, RandomForest, GradientBoosting, XGBoost, MLP
4. Evaluate with 6 metrics: R², Adj R², MAE, RMSE, MAPE, train-test gap
5. Generate comparison visualizations

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

## Dataset: PALMO

- **16 NACA 4-series airfoils**: 12 training + 4 test (unseen geometries)
- **52,480 CFD simulations**: NASA OVERFLOW RANS with Spalart-Allmaras turbulence
- **Conditions**: 10 Mach (0.25-0.90) × 8 Re (75k-8M) × 41 AoA (-20° to +20°)
- **Features**: Camber, camber position, thickness, Mach, Reynolds, angle of attack
- **Target**: L/D = Cl / Cd (lift-to-drag ratio)
- **Source**: High-fidelity data for ML surrogate validation from NASA Rotorcraft Systems Engineering