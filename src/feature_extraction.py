import numpy as np
import pandas as pd
from typing import List, Any
import re

def parse_filename(filename: str) -> dict:
    pattern = r'Uinf_([\d.]+)_AoA_([\d.-]+)_NACA(\d)(\d)(\d)(\d?)'
    match = re.search(pattern, filename)
    if not match:
        return None
    
    uinf = float(match.group(1))
    aoa = float(match.group(2))
    naca_digits = [int(d) if d else 0 for d in match.groups()[2:]]
    
    return {
        'Uinf': uinf,
        'AoA': aoa,
        'NACA_1': naca_digits[0],
        'NACA_2': naca_digits[1],
        'NACA_3': naca_digits[2],
        'NACA_4': naca_digits[3] if len(naca_digits) > 3 else 0
    }

def compute_flow_statistics(simulation: Any) -> dict:
    return {
        'mean_pressure': np.mean(simulation[:, 0]),
        'std_pressure': np.std(simulation[:, 0]),
        'mean_velocity': np.mean(np.sqrt(simulation[:, 1]**2 + simulation[:, 2]**2)),
        'max_velocity': np.max(np.sqrt(simulation[:, 1]**2 + simulation[:, 2]**2))
    }

def extract_features_from_simulations(simulations: List[Any], filenames: List[str]) -> pd.DataFrame:
    features_list = []
    
    for sim, fname in zip(simulations, filenames):
        params = parse_filename(fname)
        if params is None:
            continue
        
        flow_stats = compute_flow_statistics(sim)
        
        cl = 0.1 * params['AoA']
        cd = 0.01 + 0.0001 * params['AoA']**2
        
        if cd < 0.0001:
            continue
        
        features = {**params, **flow_stats, 'L_D': cl / cd}
        features_list.append(features)
    
    return pd.DataFrame(features_list)
