from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
RESULTS_DIR = PROJECT_ROOT / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'
TABLES_DIR = RESULTS_DIR / 'tables'
METRICS_DIR = PROJECT_ROOT / 'ideas' / 'metrics'
MODELS_DIR = PROJECT_ROOT / 'models'

for dir_path in [DATA_DIR, FIGURES_DIR, TABLES_DIR, METRICS_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

def generate_run_id() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')

TRAIN_AIRFOILS = ['NACA 0006', 'NACA 0012', 'NACA 0018', 'NACA 0024',
                  'NACA 2406', 'NACA 2412', 'NACA 2418', 'NACA 2424',
                  'NACA 4406', 'NACA 4412', 'NACA 4418', 'NACA 4424']

TEST_AIRFOILS = ['NACA 3415', 'NACA 3418', 'NACA 4415', 'NACA 4421']
