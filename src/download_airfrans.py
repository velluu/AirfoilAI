from airfrans import dataset
from pathlib import Path
import pickle

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

print("Downloading AirfRANS dataset (this may take several minutes)...")
print("\nFull dataset info:")
print("  - 1000 CFD simulations (800 train, 200 test)")
print("  - NACA 3 and 4 series airfoils")
print("  - ~180k mesh points per simulation")
print("  - 12 flow variables per point\n")

dataset.download(root='data', file_name='Dataset', unzip=True, OpenFOAM=False)

print("\n✓ Dataset downloaded to data/Dataset/")
print("\nReady to run: python main.py")
