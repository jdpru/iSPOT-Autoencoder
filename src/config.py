from pathlib import Path

SEED = 42
TARGETS = ['response', 'remission']

######################### PATHS #########################

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directories
DATA_DIR = PROJECT_ROOT/'data'
FIGURES_DIR = PROJECT_ROOT/'figures'
LOGS_DIR = PROJECT_ROOT/'logs'
MODELS_DIR = PROJECT_ROOT/'models'
DIRECTORIES = (DATA_DIR, FIGURES_DIR, LOGS_DIR, MODELS_DIR)

# HDF5 data files
TRAIN_DATA_FILE = DATA_DIR / 'dummy_train_data.h5'
TEST_DATA_FILE = DATA_DIR / 'dummy_test_data.h5'

##################### TREATMENT INFO #####################

# Treatment code constants
ALL = 0     # All drugs
ESC = 1     # Escitalopram
SER = 2     # Sertraline
VEN = 3     # Venlafaxine XR

# Short identifiers for drugs (used in file paths, keys, etc.)
DRUG_IDS = [
    'all',      # DRUG_IDS[ALL]
    'esc',      # DRUG_IDS[ESC]
    'ser',      # DRUG_IDS[SER]
    'ven'       # DRUG_IDS[VEN]
]

# Drug full names
DRUG_NAMES = [
    'All Drugs',        # DRUG_NAMES[ALL]
    'Escitalopram',     # DRUG_NAMES[ESC]
    'Sertraline',       # DRUG_NAMES[SER] 
    'Venlafaxine XR'    # DRUG_NAMES[VEN]
]

##################### TRAINING PARAMETERS ##################### 

BATCH_SIZE = 32
LEARNING_RATE = 0.001
MIN_DELTA_TRAIN_LOSS = 0.0001
N_CVFOLDS = 5
N_EPOCHS = 50

##################### HYPERPARAMETERS #####################

INPUT_DIM = 3120  # 26 electrodes * 120 timepoints
LATENT_DIM = 64
RECONSTRUCTION_WEIGHT = 1.0
PREDICTION_WEIGHT = 1.0