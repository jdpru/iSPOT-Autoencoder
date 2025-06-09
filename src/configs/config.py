from pathlib import Path
import torch

SEED = 42

# Device configuration
def get_device():
    if torch.cuda.is_available():
        print("\n CUDA available, using GPU \n")
        return torch.device('cuda')
    else:
        print("\n CUDA not available, using CPU \n")
        return torch.device('cpu')

DEVICE = get_device()

######################### PATHS #########################

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Directories
DATA_DIR = PROJECT_ROOT/'data'
FIGURES_DIR = PROJECT_ROOT/'figures'
LOGS_DIR = PROJECT_ROOT/'logs'
MODELS_DIR = PROJECT_ROOT/'models'
CHECKPOINTS_DIR = PROJECT_ROOT/'checkpoints'
DIRECTORIES = (DATA_DIR, FIGURES_DIR, LOGS_DIR, MODELS_DIR, CHECKPOINTS_DIR)

# HDF5 data files
TRAIN_DATA_FILE = DATA_DIR / 'train_data_response.csv'
TEST_DATA_FILE = DATA_DIR / 'test_data_response.csv'