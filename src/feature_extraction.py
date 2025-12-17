"""
Feature extraction module
Extracts features and targets from AirfRANS CFD simulations
"""
import numpy as np
import pandas as pd
from typing import Tuple
from tqdm import tqdm
from src.data_loader import parse_airfrans_filename


def compute_lift_drag_coefficients(sim_data: np.ndarray, u_inf: float, aoa_deg: float) -> Tuple[float, float]:
    """
    Compute Cl and Cd from CFD simulation mesh data
    
    This is a simplified approximation using pressure and velocity statistics.
    For accurate values, proper surface integration would be needed.
    
    Args:
        sim_data: Simulation mesh data (N_points, 12)
        u_inf: Freestream velocity
        aoa_deg: Angle of attack in degrees
        
    Returns:
        (Cl, Cd) - Lift and drag coefficients
    """
    u = sim_data[:, 3]
    v = sim_data[:, 4]
    p = sim_data[:, 5]
    
    aoa_rad = np.deg2rad(aoa_deg)
    
    # Pressure coefficient: Cp = (p - p_inf) / (0.5 * rho * u_inf^2)
    # Simplified: use pressure statistics as proxy
    p_mean = np.mean(p)
    p_std = np.std(p)
    
    # Velocity magnitude relative to freestream
    v_mag = np.sqrt(u**2 + v**2)
    v_mean = np.mean(v_mag)
    
    # Rough approximations based on flow statistics
    # These would need calibration with known data
    cl_approx = 0.1 * aoa_deg + 0.01 * (v_mean / u_inf) * p_std
    cd_approx = 0.01 + 0.001 * aoa_deg**2 + 0.005 * p_std
    
    return float(cl_approx), float(cd_approx)


def extract_features_from_simulation(sim_data: np.ndarray, filename: str) -> dict:
    """
    Extract all features from a single simulation
    
    Args:
        sim_data: CFD simulation mesh data
        filename: Simulation filename containing metadata
        
    Returns:
        Dictionary with features and computed L/D
    """
    metadata = parse_airfrans_filename(filename)
    
    u_inf = metadata['u_inf']
    aoa = metadata['aoa']
    reynolds = metadata['reynolds']
    naca_digits = metadata['naca_digits']
    
    # Compute Cl and Cd
    cl, cd = compute_lift_drag_coefficients(sim_data, u_inf, aoa)
    ld_ratio = cl / cd if cd > 0 else 0
    
    # Extract flow statistics
    u = sim_data[:, 3]
    v = sim_data[:, 4]
    p = sim_data[:, 5]
    
    features = {
        'reynolds': reynolds,
        'aoa': aoa,
        'u_inf': u_inf,
        'naca_series': len(naca_digits),
        'mean_u': np.mean(u),
        'std_u': np.std(u),
        'mean_v': np.mean(v),
        'std_v': np.std(v),
        'mean_p': np.mean(p),
        'std_p': np.std(p),
        'min_p': np.min(p),
        'max_u': np.max(u),
        'cl': cl,
        'cd': cd,
        'ld_ratio': ld_ratio
    }
    
    # Add NACA digits as separate features (pad with zeros for 3-digit series)
    for i in range(4):
        features[f'naca_{i}'] = naca_digits[i] if i < len(naca_digits) else 0.0
    
    return features


def extract_dataset_features(data: tuple, name: str = "dataset") -> pd.DataFrame:
    """
    Extract features from entire dataset
    
    Args:
        data: (simulations_list, filenames_list)
        name: Dataset name for progress bar
        
    Returns:
        DataFrame with features and target (ld_ratio)
    """
    simulations, filenames = data
    features_list = []
    
    print(f"Extracting features from {name}...")
    for sim_data, filename in tqdm(zip(simulations, filenames), total=len(simulations)):
        features = extract_features_from_simulation(sim_data, filename)
        features_list.append(features)
    
    df = pd.DataFrame(features_list)
    
    print(f"✓ Extracted {len(df)} samples with {len(df.columns)} features")
    print(f"  Feature columns: {list(df.columns[:10])}...")
    print(f"  Target range: L/D = [{df['ld_ratio'].min():.2f}, {df['ld_ratio'].max():.2f}]")
    
    return df


def prepare_train_test_split(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare X_train, X_test, y_train, y_test for modeling
    
    Args:
        df_train: Training dataframe
        df_test: Test dataframe
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    target_col = 'ld_ratio'
    feature_cols = [col for col in df_train.columns if col not in [target_col, 'cl', 'cd']]
    
    X_train = df_train[feature_cols].values
    y_train = df_train[target_col].values
    X_test = df_test[feature_cols].values
    y_test = df_test[target_col].values
    
    print(f"\n✓ Prepared training/test split:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test: {X_test.shape}, y_test: {y_test.shape}")
    print(f"  Features: {feature_cols}")
    
    return X_train, X_test, y_train, y_test
