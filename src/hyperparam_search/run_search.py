from src.data import EEGDataset, train_val_split, make_eeg_dataloader_from_dict
from src.hyperparam_search.unsupervised_ae_search import unsupervised_ae_search
from src.hyperparam_search.semisupervised_ae_search import semisupervised_ae_search
from src.hyperparam_search.hyperparam_configs import unsup_ae_search_space, semi_ae_search_space
from src.data import load_train_test_data
from src.configs import N_EPOCHS, BATCH_SIZE, VAL_FRAC, SEED
from torch.utils.data import random_split, DataLoader
import numpy as np
import random

def run_search():
    # Load data
    all_data = load_train_test_data()
    full_train_dict = all_data['train']
    
    # Split
    train_dict, val_dict = train_val_split(full_train_dict)
    
    # Make DataLoaders
    train_loader = make_eeg_dataloader_from_dict(train_dict)
    val_loader = make_eeg_dataloader_from_dict(val_dict, shuffle=False)
    
    # Run hyperparam search for unsupervised autoencoder
    print("\n" + "=" * 50)
    print("HYPERPARAMETER SEARCH: UNSUPERVISED AUTOENCODER")
    print("=" * 50)
    best_u_model, best_u_config, best_u_score, u_results = unsupervised_ae_search(
        train_loader, val_loader, unsup_ae_search_space
    )
    
    print("\n" + "=" * 50)
    print("HYPERPARAMETER SEARCH: SEMI-SUPERVISED AUTOENCODER")
    print("=" * 50)
    # Run hyperparam search for semi-supervised autoencoder
    best_s_model, best_s_config, best_s_score, s_results = semisupervised_ae_search(
        train_loader, val_loader, semi_ae_search_space
    )

if __name__ == "__main__":
    run_search()