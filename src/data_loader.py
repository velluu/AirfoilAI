# Usage: from src.data_loader import setup_data; setup_data() | Downloads AirfRANS, saves to data/raw and data/processed

import airfrans as af
import joblib
from pathlib import Path

def setup_data():
    project_root = Path(__file__).resolve().parent.parent
    raw_path = project_root / 'data' / 'raw'
    processed_path = project_root / 'data' / 'processed'
    
    raw_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)
    
    train_pkl = processed_path / 'dataset_full_train.pkl'
    test_pkl = processed_path / 'dataset_full_test.pkl'
    
    if train_pkl.exists() and test_pkl.exists():
        print("Data already processed")
        return
    
    af.dataset.download(root=str(raw_path), file_name='Dataset', unzip=True)
    
    train = af.dataset.load(root=str(raw_path / 'Dataset'), task='full', train=True)
    test = af.dataset.load(root=str(raw_path / 'Dataset'), task='full', train=False)
    
    joblib.dump(train, train_pkl)
    joblib.dump(test, test_pkl)
    
    zip_file = raw_path / 'Dataset.zip'
    if zip_file.exists():
        zip_file.unlink()
    
    print("Setup complete")

if __name__ == "__main__":
    setup_data()
