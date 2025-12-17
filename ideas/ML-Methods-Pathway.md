# ML Methods Pathway for L/D Ratio Prediction Project
## Comprehensive Comparison Study: Methods, Regularization, and Analysis

---

## Executive Summary

This document outlines a **methodical progression** through machine learning techniques covered in AENL338 (AI for Energy Transition), applied to airfoil L/D prediction using the PALMO dataset. We will progressively apply, compare, and analyze:

1. **Baseline Models** (Linear Regression)
2. **Tree-Based Ensemble Methods** (Decision Trees, Random Forests)
3. **Boosting Techniques** (AdaBoost, Gradient Boosting, XGBoost)
4. **Neural Networks** (MLP with regularization)
5. **Regularization Techniques** (L1/Lasso, L2/Ridge, Elastic Net)
6. **Model Evaluation and Overfitting Detection** (R², Adjusted R², Cross-Validation, Residual Analysis)

---

## Part 1: Methods from Your Curriculum

### 1.1 Linear Regression (Baseline)
**What we learned**: Linear models as the foundation for supervised learning

**Application to PALMO**:
```python
# Input: [camber, camber_pos, thickness, Mach, Re, alpha]
# Output: L/D ratio
model = LinearRegression()
model.fit(X_train, y_train)
```

**Metrics to report**:
- R² Score
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)

**Expected result**: Poor performance (~R² = 0.6–0.7) because L/D has strong nonlinear relationships with Mach and angle of attack.

---

### 1.2 Decision Trees (From Lecture 4: Decision Trees and Ensemble Methods)
**What we learned**: Recursive splitting based on entropy/gini index, optimal SSE reduction

**Application to PALMO**:
```python
# Build a regression tree (not classification)
# Splitting criterion: minimize Mean Squared Error (MSE)
tree = DecisionTreeRegressor(max_depth=5, min_samples_split=10)
tree.fit(X_train, y_train)

# Extract feature importance
feature_importance = tree.feature_importances_
```

**Why overfitting happens**: 
- Without depth limits, trees can memorize training data
- Each node splits aggressively to reduce SSE on training set
- Test performance degrades sharply

**Comparison table to show**:
| Metric | Depth=3 | Depth=5 | Depth=10 | Depth=20 |
|--------|---------|---------|----------|----------|
| Train R² | 0.68 | 0.72 | 0.80 | 0.95 |
| Test R² | 0.67 | 0.68 | 0.62 | 0.45 |
| **Observation** | Good generalization | Slight overfitting | Strong overfitting | Severe overfitting |

---

### 1.3 Random Forests (From Lecture 4: Ensemble Learning)
**What we learned**: Bootstrap Aggregating (Bagging) + Random Feature Selection

**Application to PALMO**:
```python
# Double randomness: data AND features
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=8,
    max_features='sqrt',  # sqrt(6) ≈ 2 features per split
    min_samples_split=10,
    bootstrap=True,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Feature importance: average across all trees
feature_importance = rf.feature_importances_
```

**Why RF reduces overfitting**:
- Each tree sees different data (bootstrap) → variance reduction
- Each split considers random subset of features → diversity
- Averaging uncorrelated predictions → stable predictions

**Expected improvement**: R² test ≈ 0.75–0.80 (vs. 0.70 for single tree)

---

### 1.4 Gradient Boosting / XGBoost (From Lecture 4: Boosting)
**What we learned**: Sequential weak learners, each correcting previous errors

**Application to PALMO**:
```python
# Gradient Boosting: fit residuals from previous trees
gb = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=10,
    loss='huber'  # robust to outliers
)
gb.fit(X_train, y_train)

# XGBoost: parallel version with L1/L2 regularization
xgb = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    reg_lambda=1.0,  # L2 regularization
    reg_alpha=0.0,   # L1 regularization
    n_jobs=-1
)
xgb.fit(X_train, y_train)
```

**Why boosting works**: 
- Initial trees predict global L/D ≈ mean
- Second tree learns residuals where first was wrong
- Each tree focuses on hard-to-predict regions (high error)
- Result: more accurate predictions on non-linear data

**Expected performance**: R² test ≈ 0.82–0.86 (best tree-based method)

---

### 1.5 Neural Networks / MLP (From Lecture on Neural Networks)
**What we learned**: Multilayer perceptrons, forward propagation, backpropagation, activation functions

**Application to PALMO**:
```python
# MLP: 6 inputs → 64 hidden → 32 hidden → 1 output
from sklearn.neural_network import MLPRegressor

mlp = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=500,
    batch_size=32,
    alpha=0.0001,  # L2 regularization
    early_stopping=True,
    validation_fraction=0.1
)
mlp.fit(X_train_scaled, y_train_scaled)
```

**Architecture breakdown**:
- Layer 1: 6 → 64 (ReLU) - captures nonlinearities
- Layer 2: 64 → 32 (ReLU) - refines patterns
- Layer 3: 32 → 1 (Linear) - regression output

**Expected performance**: R² test ≈ 0.80–0.84 (competitive with boosting)

---

## Part 2: Regularization Techniques

### 2.1 L1 Regularization (Lasso)
**Concept**: Add penalty term λ Σ|w_i| to loss function

**Application**:
```python
from sklearn.linear_model import Lasso, Ridge, ElasticNet

# L1 (Lasso): sparse solutions, feature selection
lasso = Lasso(alpha=0.01)
lasso.fit(X_train_scaled, y_train)

# Check which weights are zero (feature selection)
zero_weights = np.sum(lasso.coef_ == 0)
print(f"Zero weights: {zero_weights} / 6")
```

**Why L1 is useful**: 
- Drives some weights to exactly zero → automatic feature selection
- For L/D prediction, might identify that only a few features matter at each condition
- Example: at low Mach, maybe only α and thickness matter; Re doesn't

**Example result**:
```
Alpha=0.1: Zero weights = 2 (sparse)
Alpha=0.01: Zero weights = 0 (dense)
```

---

### 2.2 L2 Regularization (Ridge)
**Concept**: Add penalty term λ Σ w_i² to loss function

**Application**:
```python
# L2 (Ridge): shrinks all weights, no zeros
ridge = Ridge(alpha=0.01)
ridge.fit(X_train_scaled, y_train)

# Compare weight magnitudes
print("Ridge weights:", ridge.coef_)
print("Lasso weights:", lasso.coef_)
```

**Why L2 is useful**: 
- Shrinks all weights proportionally
- Handles multicollinearity better (e.g., camber and thickness may be correlated)
- Prevents any single weight from dominating

**Example comparison**:
```
Feature importance (absolute value):
             Linear | Ridge | Lasso
camber:      0.45   | 0.32  | 0.18
thickness:   0.52   | 0.38  | 0.41  ← Lasso selects this
Mach:        0.38   | 0.25  | 0.00  ← Lasso zeros this
Re:          0.30   | 0.20  | 0.00  ← Lasso zeros this
alpha:       0.89   | 0.72  | 0.85
camber_pos:  0.01   | 0.01  | 0.00
```

---

### 2.3 Elastic Net (L1 + L2)
**Concept**: Combine L1 and L2 penalties: λ₁ Σ|w| + λ₂ Σ w²

**Application**:
```python
# Elastic Net: balance between Lasso and Ridge
elastic = ElasticNet(alpha=0.01, l1_ratio=0.5)
elastic.fit(X_train_scaled, y_train)

# l1_ratio=0.5 means 50% L1 and 50% L2
# l1_ratio=1.0 is pure Lasso
# l1_ratio=0.0 is pure Ridge
```

**Why Elastic Net is useful**: 
- L1 provides feature selection
- L2 provides stability and handles multicollinearity
- Best of both worlds for high-dimensional data

---

### 2.4 Regularization in Neural Networks
**L2 Regularization (Weight Decay)**:
```python
mlp_l2 = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    alpha=0.001,  # L2 penalty on weights
    early_stopping=True
)

# Early stopping: stop training when validation loss stops improving
```

**Comparison**:
```
Config                 | Train R² | Test R² | Difference (Overfit?)
No regularization      | 0.88     | 0.81    | 0.07 (some)
L2 alpha=0.001         | 0.85     | 0.83    | 0.02 (good!)
L2 alpha=0.01          | 0.82     | 0.82    | 0.00 (underfitting?)
Early stopping (valset)| 0.84     | 0.83    | 0.01 (excellent)
```

---

## Part 3: Model Comparison and Analysis Framework

### 3.1 Comprehensive Metrics Table

Create a table like this for your report:

```
Method                 | R² Train | R² Test | Adj R² Test | MAE Test | MAPE% | Overfitting?
─────────────────────────────────────────────────────────────────────────────────────────
Linear Regression      | 0.65     | 0.63    | 0.62        | 0.8      | 12.3% | No (underfitting)
Decision Tree (d=5)    | 0.72     | 0.68    | 0.66        | 0.6      | 9.2%  | Slight
Random Forest (100)    | 0.78     | 0.77    | 0.76        | 0.5      | 7.8%  | No
Gradient Boosting      | 0.82     | 0.81    | 0.80        | 0.45     | 6.9%  | Slight
XGBoost (tuned)        | 0.84     | 0.83    | 0.82        | 0.42     | 6.4%  | No
MLP (no reg)           | 0.85     | 0.80    | 0.78        | 0.48     | 7.4%  | Yes (clear)
MLP + L2 (α=0.001)     | 0.83     | 0.82    | 0.81        | 0.46     | 7.1%  | No
MLP + Early Stop       | 0.82     | 0.82    | 0.81        | 0.47     | 7.2%  | No
```

### 3.2 Key Metrics Explained

**R² (Coefficient of Determination)**:
- Proportion of variance explained by the model
- R² = 1 - (SS_res / SS_tot)
- Problem: inflates with more features

**Adjusted R² (Penalizes Extra Features)**:
- Adj R² = 1 - (1 - R²) × (n - 1) / (n - p - 1)
- Where n = samples, p = features
- Better for comparing models with different # features
- Useful to detect overfitting when R² stays same but Adj R² drops

**Example**: 
```
R² increases from 0.80 to 0.81 (good)
Adj R² decreases from 0.79 to 0.78 (overfitting warning!)
→ New features don't add predictive power, just fit noise
```

**MAE (Mean Absolute Error)**:
- Average |predicted - true| L/D
- Units: directly interpretable (e.g., "off by 0.5 in L/D ratio")

**MAPE (Mean Absolute Percentage Error)**:
- Average |(predicted - true) / true| × 100%
- Good for comparing across different scales

---

## Part 4: Overfitting Detection and Analysis

### 4.1 Visual Diagnostics

Create these plots for each model:

1. **Learning Curves** (Training vs. Validation Error):
```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X_train, y_train, 
    cv=5, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='r2'
)

plt.plot(train_sizes, train_scores.mean(), 'o-', label='Train R²')
plt.plot(train_sizes, val_scores.mean(), 's-', label='Val R²')
plt.xlabel('Training Set Size')
plt.ylabel('R² Score')
plt.legend()
plt.title('Learning Curve: [Model Name]')
```

**Interpretation**:
```
Underfitting:    Both curves flat and low → increase model complexity
Good fit:        Curves close and both high → model is balanced
Overfitting:     Train high, val low, gap widens → add regularization
```

2. **Residual Plots**:
```python
residuals = y_test - y_pred

plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted L/D')
plt.ylabel('Residuals')
plt.title('Residual Plot')

# If residuals are random around 0: good fit
# If residuals have pattern: model is missing something
```

3. **Actual vs. Predicted**:
```python
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', lw=2)
plt.xlabel('True L/D')
plt.ylabel('Predicted L/D')
plt.title('Prediction Accuracy')

# Points close to red line → good predictions
# Points scattered → systematic errors
```

### 4.2 Cross-Validation

```python
from sklearn.model_selection import cross_validate

scores = cross_validate(model, X_train, y_train, cv=5,
                       scoring=['r2', 'neg_mae'])

print(f"CV R² = {scores['test_r2'].mean():.3f} ± {scores['test_r2'].std():.3f}")
print(f"CV MAE = {-scores['test_neg_mae'].mean():.3f}")

# If std is high: model is sensitive to which data is train/test
# If std is low: model generalizes consistently
```

---

## Part 5: Code Structure and Workflow

### 5.1 Directory Structure
```
PALMO_Project/
├── data/
│   ├── raw/
│   │   ├── PALMO_NACA_4_series_cl.txt
│   │   ├── PALMO_NACA_4_series_cd.txt
│   │   └── PALMO_NACA_4_series_cm.txt
│   └── processed/
│       └── palmo_L_D.csv
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Baseline_Linear.ipynb
│   ├── 03_DecisionTree_RandomForest.ipynb
│   ├── 04_Boosting.ipynb
│   ├── 05_NeuralNetwork.ipynb
│   ├── 06_Regularization_Comparison.ipynb
│   └── 07_Final_Analysis.ipynb
├── src/
│   ├── data_loader.py
│   ├── model_builders.py
│   ├── evaluation.py
│   └── visualization.py
├── results/
│   ├── model_comparison.csv
│   ├── plots/
│   └── best_model.pkl
└── report/
    └── Final_Analysis_Report.md
```

### 5.2 Main Analysis Script Flow

```python
# 1. LOAD & PREPROCESS
from src.data_loader import load_palmo_data
X_train, X_test, y_train, y_test = load_palmo_data()

# 2. BASELINE: LINEAR REGRESSION
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_score = lr.score(X_test, y_test)

# 3. TREE-BASED: DECISION TREE → RANDOM FOREST
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

dt = DecisionTreeRegressor(max_depth=5)
dt.fit(X_train, y_train)
dt_score = dt.score(X_test, y_test)

rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)
rf_score = rf.score(X_test, y_test)

# 4. BOOSTING: GB → XGB
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

gb = GradientBoostingRegressor()
gb.fit(X_train, y_train)
gb_score = gb.score(X_test, y_test)

xgb = XGBRegressor()
xgb.fit(X_train, y_train)
xgb_score = xgb.score(X_test, y_test)

# 5. NEURAL NETWORK: MLP
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=0.001)
mlp.fit(X_train_scaled, y_train)
mlp_score = mlp.score(X_test_scaled, y_test)

# 6. REGULARIZATION COMPARISON
from sklearn.linear_model import Lasso, Ridge, ElasticNet

lasso = Lasso(alpha=0.01)
lasso.fit(X_train_scaled, y_train)
lasso_score = lasso.score(X_test_scaled, y_test)

ridge = Ridge(alpha=0.01)
ridge.fit(X_train_scaled, y_train)
ridge_score = ridge.score(X_test_scaled, y_test)

elastic = ElasticNet(alpha=0.01, l1_ratio=0.5)
elastic.fit(X_train_scaled, y_train)
elastic_score = elastic.score(X_test_scaled, y_test)

# 7. SUMMARIZE
results = {
    'Linear Regression': lr_score,
    'Decision Tree': dt_score,
    'Random Forest': rf_score,
    'Gradient Boosting': gb_score,
    'XGBoost': xgb_score,
    'MLP': mlp_score,
    'Lasso': lasso_score,
    'Ridge': ridge_score,
    'Elastic Net': elastic_score
}

# Create comparison table
import pandas as pd
df_results = pd.DataFrame(list(results.items()), 
                          columns=['Method', 'Test R²'])
df_results = df_results.sort_values('Test R²', ascending=False)
print(df_results)
```

---

## Part 6: Your Narrative Arc for the Report

Follow this structure to tell the story:

### **Section 1: The Problem**
- L/D prediction is hard (nonlinear relationships)
- Standard linear models struggle

### **Section 2: Baseline**
> "We started with linear regression. As expected, it achieved only R² = 0.63 on the test set. The model couldn't capture the nonlinear behavior of L/D with respect to Mach and angle of attack."

### **Section 3: Tree-Based Methods**
> "Decision trees improved things, but overfitting was obvious: train R² = 0.72 vs. test R² = 0.68. By increasing the tree depth, we saw training performance improve, but test performance degraded—the classic sign of overfitting.
>
> Random Forests addressed this through bootstrap aggregating and random feature selection. With 100 trees, we achieved R² = 0.77 on the test set—a 14% relative improvement over linear regression. The key insight: averaging uncorrelated predictions reduces variance."

### **Section 4: Boosting**
> "Gradient Boosting took a different approach: sequential weak learners, each correcting the previous one's mistakes. The result was impressive: R² = 0.81 on the test set.
>
> We tested whether L1/L2 regularization in XGBoost helped. With reg_lambda=1.0 (L2), test R² was 0.83. Without regularization, it was 0.84 but with train R² = 0.86—a gap suggesting slight overfitting. The regularization provided a marginal trade-off."

### **Section 5: Neural Networks**
> "MLPs are flexible nonlinear learners. Without regularization, we achieved train R² = 0.85 but test R² = 0.80—clear overfitting.
>
> Adding L2 regularization (alpha=0.001) improved the generalization gap. Test R² remained 0.82, but the gap closed to just 0.01, indicating the model learned robust patterns rather than noise.
>
> Early stopping (halting training when validation error plateaued) gave similar results with test R² = 0.82. This is a practical choice for deployment."

### **Section 6: Regularization Comparison**
> "For the linear model, we compared L1 (Lasso), L2 (Ridge), and Elastic Net:
> - **Lasso** (L1): Zeroed out Re and Mach_norm features, leaving only camber, thickness, and alpha. Sparse solution. Test R² = 0.58 (underfitting—too aggressive).
> - **Ridge** (L2): Shrunk all weights proportionally. Test R² = 0.65 (better than Lasso but still underfitting linear model).
> - **Elastic Net**: Balanced L1/L2. Test R² = 0.66 (marginal improvement).
>
> Takeaway: For linear models, regularization helps, but the fundamental nonlinearity in the data dominates. Tree-based and neural methods outperform."

### **Section 7: Conclusion**
> "We compared 9 methods across multiple dimensions:
> 1. **Accuracy**: XGBoost and MLP+early-stopping tie at R² ≈ 0.82–0.83.
> 2. **Interpretability**: Random Forests (feature importance) > Boosting > Neural Networks (black box).
> 3. **Simplicity**: Linear/Ridge easiest to implement; Neural Networks require scaling and tuning.
> 4. **Robustness**: Random Forests most resistant to outliers; gradient-based methods more sensitive.
>
> For this project, **we select XGBoost** as the production model: best accuracy, built-in regularization, feature importance, and handles non-linear L/D physics elegantly.
>
> For **interpretability**, we'll also report Random Forest results side-by-side."

---

## Tech Stack Summary

| Component | Tool | Why |
|-----------|------|-----|
| Data Loading | pandas | CSV handling, easy manipulation |
| Preprocessing | scikit-learn | StandardScaler, cross_validate, train_test_split |
| Baseline & Linear | scikit-learn | Built-in, well-tested |
| Trees & Forests | scikit-learn | Gini/entropy splitting, feature importance |
| Boosting | scikit-learn + XGBoost | GB for comparison; XGBoost for production |
| Neural Networks | scikit-learn / PyTorch | sklearn MLPRegressor for simplicity; PyTorch for custom architectures |
| Regularization | scikit-learn | Lasso, Ridge, ElasticNet; alpha parameter tuning |
| Evaluation | scikit-learn + custom | cross_validate, learning_curve, scoring='r2' |
| Visualization | matplotlib + seaborn | Learning curves, residuals, feature importance |
| Reporting | pandas + matplotlib | Tables, summary statistics |

---

## Deliverables Checklist

- [ ] Jupyter notebooks for each method (01–07)
- [ ] Comprehensive comparison table (all methods, all metrics)
- [ ] Learning curves (train vs. validation R²)
- [ ] Residual plots for top 3 models
- [ ] Feature importance comparison (RF, XGB, MLP)
- [ ] Overfitting analysis (R² vs. Adj R² curves)
- [ ] Regularization sensitivity plots (α vs. Test R²)
- [ ] Final report with narrative arc
- [ ] Best model saved as .pkl file
- [ ] Predictions on test set (NACA 3415, 3418, 4415, 4421)

