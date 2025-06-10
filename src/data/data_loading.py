import pandas as pd
from src.configs import *

def load_train_test_data():
    """
    Load train and test data from CSV files.
    
    Returns:
        dict: Dictionary containing train and test DataFrames.
    """
    train_df = pd.read_csv(TRAIN_DATA_FILE)
    test_df = pd.read_csv(TEST_DATA_FILE)

    # Ensure the data is loaded correctly
    if train_df.empty or test_df.empty:
        raise ValueError("Train or test data is empty! Please check the CSV files.")
    
    # Get clinical data
    train_clinical_data = train_df[CLINICAL_COLS].values
    test_clinical_data = test_df[CLINICAL_COLS].values

    # Get EEG data
    train_eeg_data = train_df[EEG_COLS].values
    test_eeg_data = test_df[EEG_COLS].values

    # Get response labels
    train_y = train_df['response'].values
    test_y = test_df['response'].values

    # Get treatment labels
    train_treatments = train_df['treatment'].values
    test_treatments = test_df['treatment'].values

    # Get patient IDs
    train_patient_ids = train_df['patient_id'].values
    test_patient_ids = test_df['patient_id'].values

    # Print shapes of the data
    print("=" * 65)
    print("TRAIN DATA".ljust(32) + "TEST DATA")
    print("=" * 65)
    print(f"Clinical data: {str(train_clinical_data.shape).ljust(20)} {test_clinical_data.shape}")
    print(f"EEG data: {str(train_eeg_data.shape).ljust(25)} {test_eeg_data.shape}")
    print(f"Patient IDs: {str(train_df['patient_id'].shape).ljust(22)} {test_df['patient_id'].shape}")
    print(f"Responses: {str(train_y.shape).ljust(25)} {test_y.shape}")
    print(f"Treatments: {str(train_treatments.shape).ljust(23)} {test_treatments.shape}")
    print("=" * 65)


    return {
        'train': {
            'eeg_data': train_eeg_data,
            'clinical_data': train_clinical_data,
            'response': train_y,
            'treatment': train_treatments,
            'patient_id': train_patient_ids
        },
        'test': {
            'eeg_data': test_eeg_data,
            'clinical_data': test_clinical_data,
            'response': test_y,
            'treatment': test_treatments,
            'patient_id': test_patient_ids
        }
    }
    
# def _check_eeg_normalization(eeg_data, dataset_name):
#     """
#     Check if EEG data is normalized per patient per electrode.

#     Args:
#         eeg_data (np.ndarray): EEG data - either (n_patients, N_ELECTRODES, N_TIMEPOINTS)
#         dataset_name (str): Name of the dataset for logging (train or test).
#     """

#     # Handle both shapes
#     if eeg_data.ndim == 2:  # Flattened: (n_patients, n_electrodes * n_timepoints)
#         eeg_data = eeg_data.reshape(-1, N_ELECTRODES, N_TIMEPOINTS)
    
#     # Mean and std across time dimension (axis=2) for each patient-electrode
#     means = np.mean(eeg_data, axis=2)  # (n_patients, n_electrodes)
#     stds = np.std(eeg_data, axis=2)    # (n_patients, n_electrodes)
    
#     # Should be close to 0 and 1 respectively
#     if not (np.allclose(means, 0, atol=1e-6) and np.allclose(stds, 1, atol=1e-6)):
#         raise ValueError(f"{dataset_name} EEG data is not normalized per patient per electrode! "
#                         f"Mean of means: {np.mean(means):.6f}, Mean of stds: {np.mean(stds):.6f}")

# def normalize_eeg_data():
#     """
#     Normalize train and test EEG data per patient per electrode.
#     Overwrites the original HDF5 files.
#     """
#     with h5py.File(TRAIN_DATA_FILE, 'r+') as f:
#         eeg_data = f['eeg_data'][:]
        
#         # Normalize per patient per electrode (across time axis)
#         normalized_data = (eeg_data - np.mean(eeg_data, axis=2, keepdims=True)) / np.std(eeg_data, axis=2, keepdims=True)
        
#         # Overwrite the dataset
#         del f['eeg_data']
#         f['eeg_data'] = normalized_data
        
#         print(f"\n ✓✓✓ Train data normalized. ✓✓✓ \nMean: {np.mean(normalized_data):.3f}, STD: {np.std(normalized_data):.3f}, Shape: {normalized_data.shape}")
    
#     with h5py.File(TEST_DATA_FILE, 'r+') as f:
#         eeg_data = f['eeg_data'][:]
        
#         # Normalize per patient per electrode (across time axis)
#         normalized_data = (eeg_data - np.mean(eeg_data, axis=2, keepdims=True)) / np.std(eeg_data, axis=2, keepdims=True)
        
#         # Overwrite the dataset
#         del f['eeg_data']
#         f['eeg_data'] = normalized_data
        
#         print(f"\n ✓✓✓ Test data normalized. ✓✓✓ \nMean: {np.mean(normalized_data):.3f}, STD: {np.std(normalized_data):.3f}, Shape: {normalized_data.shape}")
    
# def _check_clinical_normalization(clinical_data, dataset_name):
#     """
#     Check if continuous clinical variables are normalized.
#     For train: each feature should have mean≈0, std≈1
#     For test: just check that data looks reasonable (no extreme values)

#     Args:
#         clinical_data (np.ndarray): Clinical data - shape (n_patients, n_features)
#         dataset_name (str): Name of the dataset for logging (train or test).
#     """
#     if dataset_name.lower() == "train":
#         # Train data: each feature should be normalized (mean≈0, std≈1)
#         means = np.mean(clinical_data, axis=0)  # (n_features,)
#         stds = np.std(clinical_data, axis=0)    # (n_features,)
        
#         if not (np.allclose(means, 0, atol=1e-6) and np.allclose(stds, 1, atol=1e-6)):
#             raise ValueError(f"{dataset_name} clinical data is not normalized per feature! "
#                             f"Mean of means: {np.mean(means):.6f}, Mean of stds: {np.mean(stds):.6f}")
#     else:
#         # Test data: just check for reasonable values (normalized with train stats)
#         if np.any(np.abs(clinical_data) > 10):  # Flag extreme outliers
#             print(f"Warning: {dataset_name} clinical data contains extreme values (>10 or <-10). "
#                   f"Min: {np.min(clinical_data):.3f}, Max: {np.max(clinical_data):.3f}")
        
#         print(f"{dataset_name} clinical data stats - Mean: {np.mean(clinical_data):.3f}, "
#               f"Std: {np.std(clinical_data):.3f}")
    
    