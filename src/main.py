import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from src.config import *
from src.data.dataset import *
from src.data.data_loading import *
from src.utils import *
from src.experiments import *
from torch.utils.data import *

def main():
    # Load data
    data = load_train_test_data()
    train_eeg_dataloader, test_eeg_dataloader = create_eeg_dataloaders(data)

    train_clinical_data = data['train']['clinical_data']
    test_clinical_data = data['test']['clinical_data']
    train_y = data['train']['response']
    test_y = data['test']['response']

    # TO-DO: CREATE EXPERIMENT FUNCTIONS FOR EACH OF THESE

    model1 = BaselineAutoencoder()
    train_unsupervised_autoencoder(model1, train_eeg_dataloader)
    train_eeg_latent_features_unsup = extract_latent_features(train_eeg_dataloader)
    test_eeg_latent_features_unsup = extract_latent_features(test_eeg_dataloader)

    model2 = SemiSupervisedAutoencoder()
    train_semisupervised_autoencoder(model2, train_eeg_dataloader)
    train_eeg_latent_features_semisup = extract_latent_features(train_eeg_dataloader)
    test_eeg_latent_features_semisup = extract_latent_features(test_eeg_dataloader)
    
    # Train and evaluate all features with logistic regression
    train_and_evaluate_logreg(
        train_clinical_data,
        test_clinical_data,
        train_y, test_y
    )

    train_and_evaluate_logreg(
        train_eeg_latent_features_unsup, 
        test_eeg_latent_features_unsup, 
        train_y, test_y
    )

    train_and_evaluate_logreg(
        train_eeg_latent_features_semisup, 
        test_eeg_latent_features_semisup, 
        train_y, test_y
    )

    train_and_evaluate_logreg(
        np.hstack((train_eeg_latent_features_unsup, train_clinical_data)), 
        np.hstack((test_eeg_latent_features_semisup, test_clinical_data)), 
        train_y, test_y
    )

if __name__ == '__main__':
    setup()
    main()
