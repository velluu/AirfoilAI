import pickle
from pathlib import Path
from typing import Tuple, List, Any

def load_airfrans_data(data_dir: Path) -> Tuple[Tuple[List[Any], List[str]], Tuple[List[Any], List[str]]]:
    train_path = data_dir / 'dataset_full_train.pkl'
    test_path = data_dir / 'dataset_full_test.pkl'
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"AirfRANS data not found in {data_dir}")
    
    with open(train_path, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
    
    train_simulations, train_filenames = train_data
    test_simulations, test_filenames = test_data
    
    print(f"✓ Loaded {len(train_simulations)} training simulations")
    print(f"✓ Loaded {len(test_simulations)} test simulations")
    
    return (train_simulations, train_filenames), (test_simulations, test_filenames)
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


