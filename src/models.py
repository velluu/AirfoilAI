"""
Model training framework
Implements all ML methods from AENL338 course with regularization options
"""
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import numpy as np


class ModelRegistry:
    """Registry of all models to compare"""
    
    @staticmethod
    def get_linear_baseline():
        """Linear Regression - Baseline"""
        return LinearRegression(), "Linear Regression"
    
    @staticmethod
    def get_lasso(alpha=0.1):
        """Lasso (L1 Regularization)"""
        return Lasso(alpha=alpha, random_state=42), f"Lasso (α={alpha})"
    
    @staticmethod
    def get_ridge(alpha=1.0):
        """Ridge (L2 Regularization)"""
        return Ridge(alpha=alpha, random_state=42), f"Ridge (α={alpha})"
    
    @staticmethod
    def get_elastic_net(alpha=0.1, l1_ratio=0.5):
        """Elastic Net (L1 + L2)"""
        return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42), f"ElasticNet (α={alpha}, ratio={l1_ratio})"
    
    @staticmethod
    def get_decision_tree(max_depth=5):
        """Decision Tree"""
        return DecisionTreeRegressor(max_depth=max_depth, random_state=42), f"Decision Tree (depth={max_depth})"
    
    @staticmethod
    def get_random_forest(n_estimators=100, max_depth=None):
        """Random Forest"""
        depth_str = max_depth if max_depth else "None"
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features='sqrt',
            min_samples_split=10,
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        ), f"Random Forest (n={n_estimators}, depth={depth_str})"
    
    @staticmethod
    def get_gradient_boosting(n_estimators=100, learning_rate=0.1, max_depth=3):
        """Gradient Boosting"""
        return GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        ), f"Gradient Boosting (n={n_estimators}, lr={learning_rate})"
    
    @staticmethod
    def get_xgboost(n_estimators=100, learning_rate=0.1, max_depth=6, reg_lambda=1.0):
        """XGBoost with L2 Regularization"""
        return xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            reg_lambda=reg_lambda,
            random_state=42,
            n_jobs=-1
        ), f"XGBoost (n={n_estimators}, λ={reg_lambda})"
    
    @staticmethod
    def get_mlp(hidden_layers=(64, 32), alpha=0.0001, max_iter=500):
        """Multi-Layer Perceptron (Neural Network)"""
        return MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            alpha=alpha,
            max_iter=max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        ), f"MLP {hidden_layers} (α={alpha})"
    
    @staticmethod
    def get_all_baseline_models():
        """Get all baseline models for initial comparison"""
        return [
            ModelRegistry.get_linear_baseline(),
            ModelRegistry.get_lasso(),
            ModelRegistry.get_ridge(),
            ModelRegistry.get_elastic_net(),
            ModelRegistry.get_decision_tree(max_depth=5),
            ModelRegistry.get_random_forest(n_estimators=100),
            ModelRegistry.get_gradient_boosting(),
            ModelRegistry.get_xgboost(),
            ModelRegistry.get_mlp()
        ]
    
    @staticmethod
    def get_regularization_comparison_models():
        """Get models for regularization analysis"""
        models = []
        
        # Linear models with varying regularization
        for alpha in [0.01, 0.1, 1.0, 10.0]:
            models.append(ModelRegistry.get_lasso(alpha))
            models.append(ModelRegistry.get_ridge(alpha))
        
        # Tree depths for overfitting analysis
        for depth in [3, 5, 10, 20]:
            models.append(ModelRegistry.get_decision_tree(depth))
        
        # Neural networks with varying regularization
        for alpha in [0.0, 0.0001, 0.001, 0.01]:
            models.append(ModelRegistry.get_mlp(alpha=alpha))
        
        return models


def train_model_with_scaling(model, X_train, y_train, X_test=None):
    """
    Train model with feature scaling
    
    Args:
        model: sklearn-compatible model
        X_train: Training features
        y_train: Training target
        X_test: Optional test features to scale
        
    Returns:
        (trained_model, scaler, X_test_scaled)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model.fit(X_train_scaled, y_train)
    
    X_test_scaled = scaler.transform(X_test) if X_test is not None else None
    
    return model, scaler, X_test_scaled
