from src.data.eeg_dataset import EEGDataset
from torch.utils.data import DataLoader
from src.configs import BATCH_SIZE, VAL_FRAC, SEED, DEVICE
import numpy as np
import torch

def extract_latent_features(model, data_loader):
    """
    Extract latent features from autoencoder trained on EEG data.
    
    Args:
        model: Trained autoencoder model (supervised or unsupervised)
        data_loader: DataLoader with batches containing 'eeg', 'response', and 'patient_id'
        device: torch.device to move tensors to (optional)
    
    Returns:
        latent_features: Numpy array of shape (n_samples, latent_dim)
        responses: Numpy array of shape (n_samples,)
        patient_ids: List of patient IDs
    """
    model.eval()
    latent_features = []
    responses = []
    patient_ids = []
    
    with torch.no_grad():
        for batch in data_loader:
            eeg_data = batch['eeg']
            response = batch['response']
            
            eeg_data = eeg_data.to(DEVICE)
            response = response.to(DEVICE)
            
            # Extract latent features based on model type (sup/unsup vanilla or RVAE)
            if model.name == 'UnsupervisedAutoencoder':
                _, latent = model(eeg_data)
            elif model.name == 'SemiSupervisedAutoencoder':
                _, _, latent = model(eeg_data)
            elif model.name == 'SemiSupervisedRVAE':
                _, _, latent, _ = model(eeg_data)
            
            # Collect results
            latent_features.append(latent.cpu().numpy())
            responses.append(response.cpu().numpy())
            patient_ids.extend(batch['patient_id'])
    
    return np.vstack(latent_features), np.concatenate(responses), patient_ids

def make_eeg_dataloader_from_dict(data_dict, batch_size=BATCH_SIZE, shuffle=True):
    return DataLoader(
        EEGDataset(
            data_dict['eeg_data'],
            data_dict['patient_id'],
            data_dict['response'],
            data_dict['treatment']
        ),
        batch_size=batch_size, shuffle=shuffle
    )

def train_val_split(data_dict):
    """
    Further splits the provided train data
    into train and val data dicts, shuffled and with aligned splits.

    Returns:
        train_dict, val_dict (in same format)
    """
    n_total = len(next(iter(data_dict.values())))  # grab first value to get N
    n_val = int(np.round(n_total * VAL_FRAC))

    rng = np.random.RandomState(SEED)
    shuffled_indices = rng.permutation(n_total)
    val_indices = shuffled_indices[:n_val]
    train_indices = shuffled_indices[n_val:]

    train_dict = {}
    val_dict = {}
    for k, v in data_dict.items():
        train_dict[k] = v[train_indices]
        val_dict[k] = v[val_indices]
    return train_dict, val_dict