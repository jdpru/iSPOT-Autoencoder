import numpy as np
from src.configs.config import *
from src.models.unsupervised_ae import UnsupervisedAutoencoder
from src.models.semisupervised_ae import SemiSupervisedAutoencoder
from src.data.data_utils import extract_latent_features, make_eeg_dataloader_from_dict
from src.training import *
from src.utils import *

def run_clinical_only_experiment(train_clinical_data, test_clinical_data, train_y, test_y):
    """Run logistic regression on clinical features only"""
    print("\n" + "=" * 50)
    print("CLINICAL FEATURES ONLY EXPERIMENT")
    print("=" * 50)
    
    lr_model, predictions = train_and_evaluate_logreg(
        train_clinical_data, test_clinical_data, train_y, test_y
    )
    
    return lr_model, predictions

def run_unsupervised_ae_experiment(train_loader, test_loader):
    """Run baseline unsupervised autoencoder experiment"""
    print("\n" + "=" * 50)
    print("UNSUPERVISED AUTOENCODER EXPERIMENT")
    print("=" * 50)
    
    # Initialize and train model
    model = UnsupervisedAutoencoder()
    print("Training baseline autoencoder...")
    model = train_unsupervised_autoencoder(model, train_loader, n_epochs=N_EPOCHS)
    
    # Extract features
    print("Extracting latent features...")
    train_X, train_y = extract_latent_features(model, train_loader)
    test_X, test_y = extract_latent_features(model, test_loader)
    
    # Train and evaluate logistic regression
    print("Training and evaluating logistic regression...")
    lr_model, predictions = train_and_evaluate_logreg(
        train_X, test_X, train_y, test_y
    )
    
    return model, lr_model, predictions, (train_X, test_X)

def run_semisupervised_experiment(train_loader, test_loader):
    """Run semi-supervised autoencoder experiment"""
    print("\n" + "=" * 50)
    print("SEMI-SUPERVISED AUTOENCODER EXPERIMENT")
    print("=" * 50)
    
    # Initialize and train model
    model = SemiSupervisedAutoencoder()
    print("Training semi-supervised autoencoder...")
    model = train_semisupervised_autoencoder(
        model, train_loader, n_epochs=N_EPOCHS, 
        reconstruction_weight=RECONSTRUCTION_WEIGHT, 
        prediction_weight=PREDICTION_WEIGHT
    )
    
    # Extract features
    print("Extracting latent features...")
    train_X, train_y = extract_latent_features(model, train_loader)
    test_X, test_y = extract_latent_features(model, test_loader)
    
    # Train and evaluate logistic regression
    print("Training and evaluating logistic regression...")
    lr_model, predictions = train_and_evaluate_logreg(
        train_X, test_X, train_y, test_y
    )
    
    return model, lr_model, predictions, (train_X, test_X)

def run_unsupervised_ae_plus_clinical_experiment(train_loader, test_loader, 
                                               train_clinical_data, test_clinical_data):
    """Run unsupervised autoencoder + clinical features experiment"""
    print("\n" + "=" * 50)
    print("UNSUPERVISED AUTOENCODER + CLINICAL FEATURES EXPERIMENT")
    print("=" * 50)
    
    # Initialize and train model
    model = UnsupervisedAutoencoder()
    print("Training baseline autoencoder...")
    model = train_unsupervised_autoencoder(model, train_loader, n_epochs=N_EPOCHS)
    
    # Extract EEG latent features
    print("Extracting latent features...")
    train_eeg_features, train_y = extract_latent_features(model, train_loader)
    test_eeg_features, test_y = extract_latent_features(model, test_loader)
    
    # Combine with clinical features
    print("Combining EEG latent features with clinical features...")
    train_combined = np.hstack((train_eeg_features, train_clinical_data))
    test_combined = np.hstack((test_eeg_features, test_clinical_data))
    
    # Train and evaluate logistic regression
    print("Training and evaluating logistic regression...")
    lr_model, predictions = train_and_evaluate_logreg(
        train_combined, test_combined, train_y, test_y
    )
    
    return model, lr_model, predictions

def run_semisupervised_plus_clinical_experiment(train_loader, test_loader, 
                                              train_clinical_data, test_clinical_data):
    """Run semi-supervised autoencoder + clinical features experiment"""
    print("\n" + "=" * 50)
    print("SEMI-SUPERVISED AUTOENCODER + CLINICAL FEATURES EXPERIMENT")
    print("=" * 50)
    
    # Initialize and train model
    model = SemiSupervisedAutoencoder()
    print("Training semi-supervised autoencoder...")
    model = train_semisupervised_autoencoder(
        model, train_loader, n_epochs=N_EPOCHS, 
        reconstruction_weight=RECONSTRUCTION_WEIGHT, 
        prediction_weight=PREDICTION_WEIGHT
    )
    
    # Extract EEG latent features
    print("Extracting latent features...")
    train_eeg_features, train_y = extract_latent_features(model, train_loader)
    test_eeg_features, test_y = extract_latent_features(model, test_loader)
    
    # Combine with clinical features
    print("Combining EEG latent features with clinical features...")
    train_combined = np.hstack((train_eeg_features, train_clinical_data))
    test_combined = np.hstack((test_eeg_features, test_clinical_data))
    
    # Train and evaluate logistic regression
    print("Training and evaluating logistic regression...")
    lr_model, predictions = train_and_evaluate_logreg(
        train_combined, test_combined, train_y, test_y
    )
    
    return model, lr_model, predictions

def run_all_experiments(data, train_eeg_dataloader, test_eeg_dataloader):
    """Run all experimental pipelines and return results"""
    
    # Extract data components
    train_clinical_data = data['train']['clinical_data']
    test_clinical_data = data['test']['clinical_data']
    train_y = data['train']['response']
    test_y = data['test']['response']
    
    results = {}
    
    # 1. Clinical features only
    results['clinical_only'] = run_clinical_only_experiment(
        train_clinical_data, test_clinical_data, train_y, test_y
    )
    
    # 2. Unsupervised autoencoder EEG features
    results['unsupervised_ae'] = run_unsupervised_ae_experiment(
        train_eeg_dataloader, test_eeg_dataloader
    )
    
    # 3. Semi-supervised autoencoder EEG features
    results['semisupervised_ae'] = run_semisupervised_experiment(
        train_eeg_dataloader, test_eeg_dataloader
    )
    
    # 4. Unsupervised autoencoder EEG + clinical features
    results['unsupervised_ae_clinical'] = run_unsupervised_ae_plus_clinical_experiment(
        train_eeg_dataloader, test_eeg_dataloader, 
        train_clinical_data, test_clinical_data
    )
    
    # 5. Semi-supervised autoencoder EEG + clinical features
    results['semisupervised_ae_clinical'] = run_semisupervised_plus_clinical_experiment(
        train_eeg_dataloader, test_eeg_dataloader, 
        train_clinical_data, test_clinical_data
    )
    
    return results