from src.configs import *
import numpy as np
from pathlib import Path
import os
import pandas as pd
import pickle
import shelve
import torch

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
    Create directories.
    The directories are created if they do not already exist.
    '''    
    for directory in DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)  # Ensure DATA_DIR exists

def l1_penalty(model, l1_weight, encoder_only=True):
    """
    Calculate L1 penalty for the model's parameters.
    If encoder_only is True, only apply to encoder parameters.
    """
    l1_penalty = 0.0

    if encoder_only and hasattr(model, 'encoder'):
            parameters = model.encoder.parameters()
    else:
        parameters = model.parameters()
    
    for param in parameters:
        if param.requires_grad:
            l1_penalty += torch.sum(torch.abs(param))
    
    return l1_weight * l1_penalty