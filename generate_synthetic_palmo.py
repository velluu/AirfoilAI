import numpy as np
import pandas as pd
from pathlib import Path

output_dir = Path("data/raw")
output_dir.mkdir(parents=True, exist_ok=True)

print("Generating synthetic PALMO dataset matching NASA specifications...")

airfoils = [
    (0.00, 0.06), (0.00, 0.12), (0.00, 0.18), (0.00, 0.24),
    (0.02, 0.06), (0.02, 0.12), (0.02, 0.18), (0.02, 0.24),
    (0.04, 0.06), (0.04, 0.12), (0.04, 0.18), (0.04, 0.24),
    (0.03, 0.15), (0.03, 0.18), (0.04, 0.15), (0.04, 0.21)
]

mach_numbers = [0.25, 0.35, 0.45, 0.55, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
reynolds_numbers = [75000, 125000, 250000, 500000, 1000000, 2000000, 4000000, 8000000]
alphas = np.arange(-20, 21, 1)

print(f"Conditions: {len(mach_numbers)} Mach × {len(reynolds_numbers)} Re × {len(alphas)} α = {len(mach_numbers) * len(reynolds_numbers) * len(alphas)} per airfoil")
print(f"Total: {len(airfoils)} airfoils × {len(mach_numbers) * len(reynolds_numbers) * len(alphas)} = {len(airfoils) * len(mach_numbers) * len(reynolds_numbers) * len(alphas)} simulations")

def compute_cl(camber, thickness, mach, re, alpha):
    cl_linear = 0.11 * alpha * (1 - 0.1 * mach)
    cl_camber = 0.5 * camber * (1 - 0.05 * mach)
    cl_thickness = -0.02 * thickness * abs(alpha) / 10
    re_factor = 1 + 0.05 * np.log10(re / 500000)
    stall_factor = 1 / (1 + np.exp(0.5 * (abs(alpha) - 12 - 5 * thickness)))
    return (cl_linear + cl_camber + cl_thickness) * re_factor * stall_factor

def compute_cd(camber, thickness, mach, re, alpha):
    cd_min = 0.006 + 0.01 * thickness + 0.003 * camber
    cd_induced = 0.0001 * alpha**2 * (1 + 0.2 * mach)
    cd_wave = 0.02 * np.exp(10 * (mach - 0.75)) if mach > 0.7 else 0
    re_factor = (500000 / re) ** 0.1
    stall_penalty = 0.05 * np.exp(0.3 * (abs(alpha) - 15))
    return (cd_min + cd_induced + cd_wave) * re_factor + stall_penalty

def compute_cm(camber, thickness, mach, alpha):
    cm_camber = -0.05 * camber * (1 + 0.1 * alpha / 10)
    cm_alpha = -0.002 * alpha
    cm_thickness = 0.001 * thickness
    return cm_camber + cm_alpha + cm_thickness + np.random.normal(0, 0.002)

data_cl, data_cd, data_cm = [], [], []

for camber, thickness in airfoils:
    for mach in mach_numbers:
        for re in reynolds_numbers:
            for alpha in alphas:
                cl = compute_cl(camber, thickness, mach, re, alpha)
                cd = compute_cd(camber, thickness, mach, re, alpha)
                cm = compute_cm(camber, thickness, mach, alpha)
                
                data_cl.append([camber, thickness, mach, re, alpha, cl])
                data_cd.append([camber, thickness, mach, re, alpha, cd])
                data_cm.append([camber, thickness, mach, re, alpha, cm])

df_cl = pd.DataFrame(data_cl, columns=['camber', 'thickness', 'Mach', 'Re', 'alpha', 'cl'])
df_cd = pd.DataFrame(data_cd, columns=['camber', 'thickness', 'Mach', 'Re', 'alpha', 'cd'])
df_cm = pd.DataFrame(data_cm, columns=['camber', 'thickness', 'Mach', 'Re', 'alpha', 'cm'])

df_cl.to_csv(output_dir / 'PALMO_NACA_4_series_cl.txt', index=False, header=False)
df_cd.to_csv(output_dir / 'PALMO_NACA_4_series_cd.txt', index=False, header=False)
df_cm.to_csv(output_dir / 'PALMO_NACA_4_series_cm.txt', index=False, header=False)

print(f"\n✓ Generated {len(df_cl)} total samples")
print(f"✓ Saved to data/raw/")
print(f"  - PALMO_NACA_4_series_cl.txt ({len(df_cl)} rows)")
print(f"  - PALMO_NACA_4_series_cd.txt ({len(df_cd)} rows)")
print(f"  - PALMO_NACA_4_series_cm.txt ({len(df_cm)} rows)")

print("\nSample L/D statistics:")
df_cl['cd'] = df_cd['cd']
df_cl['L_D'] = df_cl['cl'] / df_cl['cd'].clip(lower=0.0001)
print(f"  L/D range: [{df_cl['L_D'].min():.2f}, {df_cl['L_D'].max():.2f}]")
print(f"  Mean L/D: {df_cl['L_D'].mean():.2f}")
print("\nDataset ready for training!")
