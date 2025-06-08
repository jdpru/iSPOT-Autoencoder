from src.config import *
import matplotlib.pyplot as plt
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

def _check_normalization(eeg_data, dataset_name):
    """
    Check if EEG data is normalized per patient per electrode.

    Args:
        eeg_data (np.ndarray): EEG data of shape (n_patients, n_electrodes, n_timepoints).
        dataset_name (str): Name of the dataset for logging (train or test).

    Raises:
        ValueError: If the EEG data is not normalized (mean close to 0 and std close to 1).
    """
    # Mean and std across time dimension (axis=2) for each patient-electrode
    means = np.mean(eeg_data, axis=2)  # (n_patients, n_electrodes)
    stds = np.std(eeg_data, axis=2)    # (n_patients, n_electrodes)
    
    # Should be close to 0 and 1 respectively
    if not (np.allclose(means, 0, atol=1e-6) and np.allclose(stds, 1, atol=1e-6)):
        raise ValueError(f"{dataset_name} EEG data is not normalized per patient per electrode! "
                        f"Mean of means: {np.mean(means):.6f}, Mean of stds: {np.mean(stds):.6f}")

def normalize_eeg_data():
    """
    Normalize train and test EEG data per patient per electrode.
    Overwrites the original HDF5 files.
    """
    with h5py.File(TRAIN_DATA_FILE, 'r+') as f:
        eeg_data = f['eeg_data'][:]
        
        # Normalize per patient per electrode (across time axis)
        normalized_data = (eeg_data - np.mean(eeg_data, axis=2, keepdims=True)) / np.std(eeg_data, axis=2, keepdims=True)
        
        # Overwrite the dataset
        del f['eeg_data']
        f['eeg_data'] = normalized_data
        
        print(f"\n ✓✓✓ Train data normalized. ✓✓✓ \nMean: {np.mean(normalized_data):.3f}, STD: {np.std(normalized_data):.3f}, Shape: {normalized_data.shape}")
    
    with h5py.File(TEST_DATA_FILE, 'r+') as f:
        eeg_data = f['eeg_data'][:]
        
        # Normalize per patient per electrode (across time axis)
        normalized_data = (eeg_data - np.mean(eeg_data, axis=2, keepdims=True)) / np.std(eeg_data, axis=2, keepdims=True)
        
        # Overwrite the dataset
        del f['eeg_data']
        f['eeg_data'] = normalized_data
        
        print(f"\n ✓✓✓ Test data normalized. ✓✓✓ \nMean: {np.mean(normalized_data):.3f}, STD: {np.std(normalized_data):.3f}, Shape: {normalized_data.shape}")
    
def load_train_test_data():
    """
    Load train and test data from HDF5 files. All values are numpy arrays.
    Checks if EEG data is normalized per patient per electrode.

    Returns:
        tuple: (train_data, test_data) where each is a dictionary containing:
               'eeg_data', 'patient_ids', 'response', 'treatment'.
    """
    train_data = {}
    test_data = {}
    
    # Load train data
    with h5py.File(TRAIN_DATA_FILE, 'r') as f:
        train_data = {
            'eeg_data': f['eeg_data'][:],
            'patient_ids': f['patient_ids'][:],
            'response': f['response'][:],
            'treatment': f['treatment'][:]
        }
    
    # Load test data
    with h5py.File(TEST_DATA_FILE, 'r') as f:
        test_data = {
            'eeg_data': f['eeg_data'][:],
            'patient_ids': f['patient_ids'][:],
            'response': f['response'][:],
            'treatment': f['treatment'][:],
        }
    
    # Check normalization of EEG data
    _check_normalization(train_data['eeg_data'], "Train")
    _check_normalization(test_data['eeg_data'], "Test")
    print("\n ✓✓✓ EEG data is normalized (per patient per electrode) ✓✓✓ \n")

    # Print shapes of the data
    print("=" * 40 + "=" * 25)
    print("TRAIN DATA".ljust(40) + "TEST DATA")
    print("=" * 40 + "=" * 25)
    for key in train_data.keys():
        train_shape = train_data[key].shape
        test_shape = test_data[key].shape
        print(f"{key}: {train_shape}".ljust(40) + f"{key}: {test_shape}")

    return train_data, test_data