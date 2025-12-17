"""
Data loading module for AirfRANS dataset
Handles loading pickled simulation data and extracting metadata
"""
import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, List
from src.config import PROCESSED_DATA_DIR


def load_airfrans_data() -> Tuple[tuple, tuple]:
    """
    Load processed AirfRANS dataset
    
    Returns:
        train: (simulations_list, filenames_list)
        test: (simulations_list, filenames_list)
    """
    train_path = PROCESSED_DATA_DIR / 'dataset_full_train.pkl'
    test_path = PROCESSED_DATA_DIR / 'dataset_full_test.pkl'
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Processed data not found. Run data setup first.\n"
            f"Expected files:\n  {train_path}\n  {test_path}"
        )
    
    train = joblib.load(train_path)
    test = joblib.load(test_path)
    
    print(f"✓ Loaded {len(train[0])} training simulations")
    print(f"✓ Loaded {len(test[0])} test simulations")
    
    return train, test


def parse_airfrans_filename(filename: str) -> dict:
    """
    Parse AirfRANS filename to extract simulation parameters
    
    Format: airFoil2D_SST_{Uinf}_{AoA}_{naca_digits...}
    
    Args:
        filename: Simulation filename
        
    Returns:
        Dictionary with u_inf, aoa, reynolds, naca_digits
    """
    parts = filename.replace('airFoil2D_SST_', '').split('_')
    u_inf = float(parts[0])
    aoa = float(parts[1])
    reynolds = u_inf / 1.56e-5  # kinematic viscosity
    naca_digits = [float(p) for p in parts[2:]]
    
    return {
        'u_inf': u_inf,
        'aoa': aoa,
        'reynolds': reynolds,
        'naca_digits': naca_digits
    }


def get_dataset_statistics(data: tuple) -> dict:
    """
    Get basic statistics about the dataset
    
    Args:
        data: (simulations_list, filenames_list)
        
    Returns:
        Dictionary with dataset statistics
    """
    simulations, filenames = data
    
    naca_3_count = 0
    naca_4_count = 0
    
    for fname in filenames:
        parts = fname.replace('airFoil2D_SST_', '').split('_')
        naca_count = len(parts[2:])
        if naca_count == 3:
            naca_3_count += 1
        elif naca_count == 4:
            naca_4_count += 1
    
    return {
        'total_samples': len(simulations),
        'naca_3_series': naca_3_count,
        'naca_4_series': naca_4_count,
        'avg_mesh_points': int(np.mean([sim.shape[0] for sim in simulations])),
        'flow_variables': simulations[0].shape[1]
    }
