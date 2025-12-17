# AirfoilAI — ML Model Comparison for L/D Prediction

Comprehensive machine learning study comparing 9 regression methods for predicting **lift-to-drag ratio (L/D)** of **NACA 4-digit airfoils** using the [AirfRANS](https://github.com/Extrality/AirfRANS) CFD dataset.

---

## Project Structure

```
AirfoilAI/
├── main.py                     # Quick baseline pipeline
├── notebooks/
│   ├── documentation.ipynb     # Dataset exploration & feature engineering
│   └── methods/                # One notebook per ML method (01–09)
│       ├── 01_linear_regression.ipynb
│       ├── 02_lasso.ipynb
│       ├── 03_ridge.ipynb
│       ├── 04_elastic_net.ipynb
│       ├── 05_decision_tree.ipynb
│       ├── 06_random_forest.ipynb
│       ├── 07_gradient_boosting.ipynb
│       ├── 08_xgboost.ipynb
│       └── 09_mlp.ipynb
├── src/
│   ├── config.py               # Paths & run-ID generation
│   ├── build_dataset.py        # Extract tabular features from AirfRANS
│   ├── tabular_data.py         # Train/test split via manifest
│   ├── models.py               # Model registry helpers
│   ├── evaluation.py           # Metrics & comparison tables
│   └── visualization.py        # Plotting utilities
├── data/
│   ├── Dataset/                # Raw AirfRANS simulations (downloaded)
│   └── processed/              # airfrans_dataset.csv (generated)
├── models/                     # Saved .joblib / .pkl model files
├── results/
│   ├── figures/
│   └── tables/
├── ideas/                      # Planning docs & implementation guides
├── environment.yml
└── requirements.txt
```

---

## Quick Start

### 1. Create Environment

```bash
conda env create -f environment.yml
conda activate airfoilai
```

### 2. Download AirfRANS Dataset

```bash
python src/download_airfrans.py
```

This fetches the full AirfRANS dataset (~1.5 GB) into `data/Dataset/`.

### 3. Build Tabular CSV

```bash
python src/build_dataset.py
```

Extracts **489 NACA 4-digit** samples with features:  
`Reynolds`, `angle_of_attack_rad`, `camber`, `camber_pos`, `thickness` → target `L_D`.

### 4. Explore Notebooks

Open any notebook in `notebooks/methods/` to:

- Run hyperparameter sweeps  
- View learning-rate / regularization curves  
- See final adjusted R² and RMSE  

---

## Dataset: AirfRANS (NACA 4-Digit Subset)

| Property | Value |
|----------|-------|
| **Samples** | 489 (378 train / 111 test) |
| **Source** | AirfRANS OpenFOAM RANS simulations |
| **Features** | Reynolds (2–6 M), AoA (−5° to +15°), camber, camber position, thickness |
| **Target** | L/D = Cl / Cd |

---

## Methods & Results Summary

| Method | Test Adj R² | Test RMSE | Notes |
|--------|-------------|-----------|-------|
| Linear Regression (SGD) | 0.5849 | 21.28 | Baseline; no regularization |
| Lasso | 0.5877 | 21.21 | L1; all 5 features retained |
| Ridge | 0.5886 | 21.18 | L2; α ≈ 25.8 |
| Elastic Net | 0.5888 | 21.18 | L1+L2; α ≈ 0.48, ratio 0.9 |
| Decision Tree | 0.9192 | 9.39 | depth=20, leaf=1; high variance |
| Random Forest | 0.8805 | 11.42 | n=25, depth=15 |
| Gradient Boosting | 0.9668 | 6.02 | n=500, lr=0.10 |
| XGBoost | 0.9642 | 6.25 | n=500, depth=4, λ=1 |
| **MLP** | **0.9913** | **3.07** | (64,32), α=3.7e-5, lr=0.037 |

> All metrics are **adjusted R²** on the held-out test set (111 samples).

---

## Requirements

- Python 3.10+
- scikit-learn, xgboost, numpy, pandas, matplotlib, seaborn
- airfrans (for dataset download)

Install via:

```bash
pip install -r requirements.txt
```

---

## License

MIT