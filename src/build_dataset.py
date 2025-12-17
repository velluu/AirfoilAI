"""
Build a tabular dataset from AirfRANS simulations.
Extracts force coefficients (Cl, Cd) and computes L/D for each simulation.

This version filters to NACA 4-digit airfoils only for clean, interpretable features.
"""

import json
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm
from airfrans import dataset


def parse_simulation_name(name: str) -> Optional[dict]:
    """
    Parse AirfRANS simulation name to extract parameters.
    
    NACA 4-digit format: airFoil2D_SST_velocity_aoa_camber_camber_pos_thickness (5 params)
    NACA 5-digit format: airFoil2D_SST_velocity_aoa_design_cl_design_aoa_reflex_thickness (6 params)
    
    Returns dict with parsed parameters and original name, or None if 5-digit (filtered out).
    """
    parts = name.split('_')
    if len(parts) < 5:
        return None
    
    # Skip prefix 'airFoil2D' and 'SST'
    params = parts[2:]
    
    # Filter out 5-digit NACA airfoils (they have 6 parameters)
    if len(params) >= 6:
        return None  # Skip 5-digit airfoils
    
    # NACA 4-digit airfoil with clean feature names
    result = {
        'name': name,
        # Geometry parameters (from NACA 4-digit naming)
        'camber': float(params[2]),           # Max camber (% of chord)
        'camber_pos': float(params[3]),       # Position of max camber (tenths of chord)
        'thickness': float(params[4]),        # Max thickness (% of chord)
    }
    
    return result


def extract_coefficients(sim_name: str, root: str = 'data/Dataset') -> dict:
    """
    Load simulation and extract Cl, Cd, L/D, and Reynolds number.

    AirfRANS `Simulation.force_coefficient()` returns:
        ((cd, cdp, cdv), (cl, clp, clv))
    where:
        cd  = total drag coefficient, cdp = pressure part, cdv = viscous part
        cl  = total lift coefficient, clp = pressure part, clv = viscous part
    
    Reynolds number is computed as: Re = V * L / NU
    where V = inlet velocity, L = chord length (1m), NU = kinematic viscosity
    """
    try:
        sim = dataset.Simulation(root, sim_name)
        coeffs = sim.force_coefficient()

        if not (isinstance(coeffs, tuple) and len(coeffs) == 2):
            raise TypeError(f"Unexpected force_coefficient() output type/shape: {type(coeffs)}")

        cd_triplet, cl_triplet = coeffs
        if not (hasattr(cd_triplet, '__len__') and hasattr(cl_triplet, '__len__')):
            raise TypeError("Unexpected force_coefficient() tuple contents")

        cd = float(cd_triplet[0])
        cl = float(cl_triplet[0])
        
        # Compute L/D
        if cl is not None and cd is not None and abs(cd) > 1e-10:
            ld_ratio = cl / cd
        else:
            ld_ratio = None
        
        # Extract Reynolds number: Re = V * L / NU (chord L = 1m in AirfRANS)
        inlet_velocity = float(sim.inlet_velocity)
        nu = float(sim.NU)  # kinematic viscosity
        reynolds = inlet_velocity * 1.0 / nu  # chord = 1m
        
        return {
            'inlet_velocity': inlet_velocity,
            'angle_of_attack_rad': float(sim.angle_of_attack),
            'Reynolds': reynolds,
            'Cl': cl,
            'Cd': cd,
            'L_D': ld_ratio,
            'success': True,
            'error': None
        }
    except Exception as e:
        return {
            'inlet_velocity': None,
            'angle_of_attack_rad': None,
            'Reynolds': None,
            'Cl': None,
            'Cd': None,
            'L_D': None,
            'success': False,
            'error': str(e)
        }


def build_full_dataset(root: str = 'data/Dataset', output_csv: str = 'data/processed/airfrans_dataset.csv'):
    """
    Build complete tabular dataset from all AirfRANS simulations.
    """
    manifest_path = Path(root) / 'manifest.json'
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Collect all simulation names from manifest
    all_names = set()
    for split_names in manifest.values():
        all_names.update(split_names)
    
    print(f"Found {len(all_names)} simulations in manifest")
    
    # Extract data from each simulation
    records = []
    failed = []
    
    for sim_name in tqdm(all_names, desc="Processing simulations"):
        # Parse name
        params = parse_simulation_name(sim_name)
        if params is None:
            failed.append((sim_name, "Failed to parse name"))
            continue
        
        # Extract coefficients
        coeffs = extract_coefficients(sim_name, root)
        
        # Combine
        record = {**params, **coeffs}
        records.append(record)
        
        if not coeffs['success']:
            failed.append((sim_name, coeffs['error']))
    
    # Create DataFrame
    df = pd.DataFrame(records)
    
    # Save
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nDataset saved to {output_path}")
    print(f"Total records: {len(df)}")
    print(f"Successful: {df['success'].sum()}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\nFirst 10 failures:")
        for name, err in failed[:10]:
            print(f"  {name}: {err}")
    
    # Show sample
    print("\nSample of dataset:")
    print(df.head(10))
    print("\nDataset info:")
    print(df.info())
    
    print("\n=== Feature Statistics ===")
    feature_cols = ['Reynolds', 'angle_of_attack_rad', 'camber', 'camber_pos', 'thickness']
    print(df[feature_cols].describe())
    
    print("\n=== Target Statistics ===")
    print(df[['Cl', 'Cd', 'L_D']].describe())
    
    print("\nReynolds number range (millions):")
    print(f"  Min: {df['Reynolds'].min()/1e6:.2f}M, Max: {df['Reynolds'].max()/1e6:.2f}M")
    
    return df


if __name__ == '__main__':
    df = build_full_dataset()
