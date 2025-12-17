import pandas as pd
from pathlib import Path
from typing import Tuple

def load_palmo_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cl_path = data_dir / 'PALMO_NACA_4_series_cl.txt'
    cd_path = data_dir / 'PALMO_NACA_4_series_cd.txt'
    cm_path = data_dir / 'PALMO_NACA_4_series_cm.txt'
    
    if not cl_path.exists() or not cd_path.exists() or not cm_path.exists():
        raise FileNotFoundError(f"PALMO data files not found in {data_dir}")
    
    cl_df = pd.read_csv(cl_path, names=['camber', 'thickness', 'Mach', 'Re', 'alpha', 'cl'])
    cd_df = pd.read_csv(cd_path, names=['camber', 'thickness', 'Mach', 'Re', 'alpha', 'cd'])
    cm_df = pd.read_csv(cm_path, names=['camber', 'thickness', 'Mach', 'Re', 'alpha', 'cm'])
    
    df = cl_df.merge(cd_df, on=['camber', 'thickness', 'Mach', 'Re', 'alpha'])
    df = df.merge(cm_df, on=['camber', 'thickness', 'Mach', 'Re', 'alpha'])
    
    df['camber_pos'] = 0.4
    df = df[df['cd'] >= 0.0001].copy()
    df['L_D'] = df['cl'] / df['cd']
    
    train_airfoils = [(0.00, 0.06), (0.00, 0.12), (0.00, 0.18), (0.00, 0.24),
                      (0.02, 0.06), (0.02, 0.12), (0.02, 0.18), (0.02, 0.24),
                      (0.04, 0.06), (0.04, 0.12), (0.04, 0.18), (0.04, 0.24)]
    test_airfoils = [(0.03, 0.15), (0.03, 0.18), (0.04, 0.15), (0.04, 0.21)]
    
    train_mask = df.apply(lambda row: (row['camber'], row['thickness']) in train_airfoils, axis=1)
    train_df = df[train_mask].copy()
    test_df = df[~train_mask].copy()
    
    print(f"✓ Loaded {len(train_df)} training samples (12 airfoils)")
    print(f"✓ Loaded {len(test_df)} test samples (4 airfoils)")
    
    return train_df, test_df
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


