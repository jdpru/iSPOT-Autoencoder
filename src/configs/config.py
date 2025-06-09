from pathlib import Path

SEED = 42

######################### PATHS #########################

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Directories
DATA_DIR = PROJECT_ROOT/'data'
FIGURES_DIR = PROJECT_ROOT/'figures'
LOGS_DIR = PROJECT_ROOT/'logs'
MODELS_DIR = PROJECT_ROOT/'models'
DIRECTORIES = (DATA_DIR, FIGURES_DIR, LOGS_DIR, MODELS_DIR)

# HDF5 data files
TRAIN_DATA_FILE = DATA_DIR / 'train_data_response.csv'
TEST_DATA_FILE = DATA_DIR / 'test_data_response.csv'