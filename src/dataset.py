import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class EEGDataset(Dataset):
    def __init__(self, eeg_data, patient_ids, response, treatment):
        """
        Args:
            eeg_data: (n_patients, 26, 120) - EEG features
            patient_ids: (n_patients,) - Patient IDs
            response: (n_patients,) - Binary response (0/1)
            treatment: (n_patients,) - Treatment type (0/1/2)
        """
        # Flatten EEG data: (n_patients, 26*120) = (n_patients, 3120)
        self.eeg_data = torch.FloatTensor(eeg_data.reshape(eeg_data.shape[0], -1))
        self.patient_ids = patient_ids
        self.response = torch.FloatTensor(response)
        self.treatment = torch.LongTensor(treatment)
        
    def __len__(self):
        return len(self.eeg_data)
    
    def __getitem__(self, idx):
        return {
            'eeg': self.eeg_data[idx],
            'response': self.response[idx],
            'treatment': self.treatment[idx],
            'patient_id': self.patient_ids[idx]
        }

def create_dataloaders(train_eeg, train_ids, train_response, train_treatment,
                      test_eeg, test_ids, test_response, test_treatment,
                      batch_size=32, shuffle_train=True):
    """
    Create train and test dataloaders
    """
    # Create datasets
    train_dataset = EEGDataset(train_eeg, train_ids, train_response, train_treatment)
    test_dataset = EEGDataset(test_eeg, test_ids, test_response, test_treatment)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle_train,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, test_loader