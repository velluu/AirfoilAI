"""
Configuration module for AirfoilAI project
"""
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'
TABLES_DIR = RESULTS_DIR / 'tables'
METRICS_DIR = PROJECT_ROOT / 'ideas' / 'metrics'
MODELS_DIR = PROJECT_ROOT / 'models'

RANDOM_STATE = 42
TEST_SIZE = 0.2

def get_run_id():
    """Generate unique run ID based on timestamp"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def ensure_directories():
    """Create all necessary directories"""
    for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                     RESULTS_DIR, FIGURES_DIR, TABLES_DIR, METRICS_DIR, MODELS_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)
