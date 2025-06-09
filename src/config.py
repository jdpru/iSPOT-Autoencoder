from pathlib import Path

SEED = 42
TARGETS = ['response']

######################### PATHS #########################

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directories
DATA_DIR = PROJECT_ROOT/'data'
FIGURES_DIR = PROJECT_ROOT/'figures'
LOGS_DIR = PROJECT_ROOT/'logs'
MODELS_DIR = PROJECT_ROOT/'models'
DIRECTORIES = (DATA_DIR, FIGURES_DIR, LOGS_DIR, MODELS_DIR)

# HDF5 data files
TRAIN_DATA_FILE = DATA_DIR / 'train_data_response.csv'
TEST_DATA_FILE = DATA_DIR / 'test_data_response.csv'

###################### INPUT DIMENSIONS #####################

INPUT_TYPES = ['clinical', 'eeg', 'eeg+clinical']

CLINICAL_COLS = [
    'age', 'hamd_1_1', 'hamd_2_1', 'hamd_3_1', 'hamd_4_1', 'hamd_5_1', 'hamd_6_1', 'hamd_7_1',
    'hamd_8_1', 'hamd_9_1', 'hamd_10_1', 'hamd_11_1', 'hamd_12_1', 'hamd_13_1', 'hamd_14_1',
    'hamd_15_1', 'hamd_16_1', 'hamd_17_1', 'hamd_total_1', 'hamd_emotions_1', 'hamd_behavioral_1',
    'hamd_somatic_1', 'race_Black', 'race_Hawaiian/Pacific Islander', 'race_Hispanic',
    'race_Mixed', 'race_Other', 'race_White', 'gender_MALE'
]

N_TRAIN_PATIENTS = 594
N_TEST_PATIENTS = 100
N_CLINICAL_FEATURES = len(CLINICAL_COLS)
N_ELECTRODES = 26
N_TIMEPOINTS = 116

ELECTRODES = ['C3', 'C4', 'CP3', 'CP4', 'CPz', 'Cz', 'F3', 'F4', 'F7', 'F8', 'FC3', 'FC4', 'FCz', 'Fp1', 'Fp2', 'Fz', 'O1', 'O2', 'Oz', 'P3', 'P4', 'Pz', 'T3', 'T4', 'T5', 'T6']
EEG_COLS = [f'{electrode}t{tp}' for electrode in ELECTRODES for tp in range(1, N_TIMEPOINTS+1)]

METADATA_COLS = ['patient_id', 'treatment', 'response']

INPUT_DIMENSIONS = {
    'clinical': N_CLINICAL_FEATURES,
    'eeg': N_ELECTRODES * N_TIMEPOINTS,
    'eeg+clinical': N_CLINICAL_FEATURES + (N_ELECTRODES * N_TIMEPOINTS)
}

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

AUTOENCODER_INPUT_DIM = N_ELECTRODES * N_TIMEPOINTS
LATENT_DIM = 64
RECONSTRUCTION_WEIGHT = 1.0
PREDICTION_WEIGHT = 1.0