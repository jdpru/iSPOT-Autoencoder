from src.configs import *
import numpy as np
from pathlib import Path
import os
import pandas as pd
import pickle
import shelve
import torch
import h5py

def setup():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    _create_directories()

def relative_path_str(path):
    '''
    Get the relative path from PROJECT_ROOT as a string.
    '''
    return str(path.relative_to(PROJECT_ROOT))

def _create_directories():
    '''
    Create directories for saving data and figures.
    The directories are created if they do not already exist.
    '''    
    for drug in DRUG_IDS:
        for target in TARGETS:
            for directory in DIRECTORIES[1:]:  # Skip DATA_DIR
                (directory / drug / target).mkdir(parents=True, exist_ok=True)
    
    DIRECTORIES[0].mkdir(parents=True, exist_ok=True)  # Ensure DATA_DIR exists