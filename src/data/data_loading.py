import pandas as pd
from src.configs import *
from src.models.unsupervised_ae import UnsupervisedAutoencoder
from src.models.semisupervised_ae import SemiSupervisedAutoencoder
from src.models.semisupervised_rvae import SemiSupervisedRVAE
import torch
import json
from src.utils import relative_path_str

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

def load_best_model_from_hyperparam_search(model_type):
    """
    Load the best model and config from hyperparameter search results.
    
    Args:
        model_type: One of 'unsupervised_ae', 'semisupervised_ae', 'semisupervised_rvae'
    
    Returns:
        model: Loaded model with best hyperparameters
        config: Best hyperparameter configuration
    """
    # Find the most recent config file for this model type
    config_files = list(MODELS_DIR.glob(f"best_{model_type}_*_config.json"))
    if not config_files:
        raise FileNotFoundError(f"No saved model config found for {model_type}. "
                              f"Please run hyperparameter search first.")
    
    # Get the most recent config file (by timestamp)
    latest_config_file = max(config_files, key=lambda x: x.stat().st_mtime)
    
    # Load config
    with open(latest_config_file, 'r') as f:
        config_data = json.load(f)
    
    best_config = config_data['best_config']
    model_filename = config_data['model_file']
    model_path = MODELS_DIR / model_filename
    
    # Initialize model with best hyperparameters
    if model_type == 'unsupervised_ae':
        model = UnsupervisedAutoencoder(
            latent_dim=best_config['latent_dim'],
            dropout_rate=best_config['dropout_rate']
        )
    elif model_type == 'semisupervised_ae':
        model = SemiSupervisedAutoencoder(
            latent_dim=best_config['latent_dim'],
            dropout_rate=best_config['dropout_rate']
        )
    elif model_type == 'semisupervised_rvae':
        model = SemiSupervisedRVAE(
            latent_dim=best_config['latent_dim']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load trained weights
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    print(f"✓ Loaded best {model_type} model from {relative_path_str(model_path)}")
    print(f"  Best config: {best_config}")
    print(f"  Validation score: {config_data['best_val_score']:.3f}")
    
    return model, best_config

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

