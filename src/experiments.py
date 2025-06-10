import numpy as np
from src.configs.config import *
from src.models.unsupervised_ae import UnsupervisedAutoencoder
from src.models.semisupervised_ae import SemiSupervisedAutoencoder
from src.data import extract_latent_features, load_best_model_from_hyperparam_search
from src.training import *
from src.utils import *
from src.plotting import plot_confusion_matrix
import pickle

def run_clinical_only_experiment(train_clinical_data, test_clinical_data, train_y, test_y):
    """Run logistic regression on clinical features only"""
    print("\n" + "=" * 50)
    print("CLINICAL FEATURES ONLY EXPERIMENT")
    print("=" * 50)
    
    lr_model, roc_auc, test_predictions = train_and_evaluate_logreg(
        train_clinical_data, test_clinical_data, train_y, test_y
    )
    
    return lr_model, roc_auc, test_predictions

def run_unsupervised_ae_experiment(train_loader, test_loader):
    """Run unsupervised autoencoder experiment with pre-trained model"""
    print("\n" + "=" * 50)
    print("UNSUPERVISED AUTOENCODER EXPERIMENT")
    print("=" * 50)
    
    # Load pre-trained model from hyperparameter search
    model, best_config = load_best_model_from_hyperparam_search('unsupervised_ae')
    
    # Extract features
    print("Extracting latent features...")
    train_X, train_y, _ = extract_latent_features(model, train_loader)
    test_X, test_y, _ = extract_latent_features(model, test_loader)
    
    # Train and evaluate logistic regression
    print("Training and evaluating logistic regression...")
    lr_model, roc_auc, test_predictions = train_and_evaluate_logreg(
        train_X, test_X, train_y, test_y
    )

    latent_features_info = {
        'train_X': train_X,
        'train_y': train_y,
        'test_X': test_X,
        'test_y': test_y,
    }
    
    return model, lr_model, roc_auc, test_predictions, latent_features_info

def run_semisupervised_experiment(train_loader, test_loader):
    """Run semi-supervised autoencoder experiment with pre-trained model"""
    print("\n" + "=" * 50)
    print("SEMI-SUPERVISED AUTOENCODER EXPERIMENT")
    print("=" * 50)
    
    # Load pre-trained model from hyperparameter search
    model, best_config = load_best_model_from_hyperparam_search('semisupervised_ae')
    
    # Extract features
    print("Extracting latent features...")
    train_X, train_y, _ = extract_latent_features(model, train_loader)
    test_X, test_y, _ = extract_latent_features(model, test_loader)
    
    # Train and evaluate logistic regression
    print("Training and evaluating logistic regression...")
    lr_model, roc_auc, test_predictions = train_and_evaluate_logreg(
        train_X, test_X, train_y, test_y
    )

    latent_features_info = {
        'train_X': train_X,
        'train_y': train_y,
        'test_X': test_X,
        'test_y': test_y,
    }
    
    return model, lr_model, roc_auc, test_predictions, latent_features_info

def run_semisupervised_rvae_experiment(train_loader, test_loader):
    """Run semi-supervised RVAE experiment with pre-trained model"""
    print("\n" + "=" * 50)
    print("SEMI-SUPERVISED RVAE EXPERIMENT")
    print("=" * 50)
    
    # Load pre-trained model from hyperparameter search
    model, best_config = load_best_model_from_hyperparam_search('semisupervised_rvae')
    
    # Extract features
    print("Extracting latent features...")
    train_X, train_y, _ = extract_latent_features(model, train_loader)
    test_X, test_y, _ = extract_latent_features(model, test_loader)
    
    # Train and evaluate logistic regression
    print("Training and evaluating logistic regression...")
    lr_model, roc_auc, test_predictions = train_and_evaluate_logreg(
        train_X, test_X, train_y, test_y
    )

    latent_features_info = {
        'train_X': train_X,
        'train_y': train_y,
        'test_X': test_X,
        'test_y': test_y,
    }
    
    return model, lr_model, roc_auc, test_predictions, latent_features_info

def run_unsupervised_ae_plus_clinical_experiment(train_loader, test_loader, 
                                               train_clinical_data, test_clinical_data):
    """Run unsupervised autoencoder + clinical features experiment with pre-trained model"""
    print("\n" + "=" * 50)
    print("UNSUPERVISED AUTOENCODER + CLINICAL FEATURES EXPERIMENT")
    print("=" * 50)
    
    # Load pre-trained model from hyperparameter search
    model, best_config = load_best_model_from_hyperparam_search('unsupervised_ae')
    
    # Extract EEG latent features
    print("Extracting latent features...")
    train_eeg_features, train_y, _ = extract_latent_features(model, train_loader)
    test_eeg_features, test_y, _ = extract_latent_features(model, test_loader)
    
    # Combine with clinical features
    print("Combining EEG latent features with clinical features...")
    train_combined = np.hstack((train_eeg_features, train_clinical_data))
    test_combined = np.hstack((test_eeg_features, test_clinical_data))
    
    # Train and evaluate logistic regression
    print("Training and evaluating logistic regression...")
    lr_model, roc_auc, test_predictions = train_and_evaluate_logreg(
        train_combined, test_combined, train_y, test_y
    )
    
    return model, lr_model, roc_auc, test_predictions

def run_semisupervised_plus_clinical_experiment(train_loader, test_loader, 
                                              train_clinical_data, test_clinical_data):
    """Run semi-supervised autoencoder + clinical features experiment with pre-trained model"""
    print("\n" + "=" * 50)
    print("SEMI-SUPERVISED AUTOENCODER + CLINICAL FEATURES EXPERIMENT")
    print("=" * 50)
    
    # Load pre-trained model from hyperparameter search
    model, best_config = load_best_model_from_hyperparam_search('semisupervised_ae')
    
    # Extract EEG latent features
    print("Extracting latent features...")
    train_eeg_features, train_y, _ = extract_latent_features(model, train_loader)
    test_eeg_features, test_y, _ = extract_latent_features(model, test_loader)
    
    # Combine with clinical features
    print("Combining EEG latent features with clinical features...")
    train_combined = np.hstack((train_eeg_features, train_clinical_data))
    test_combined = np.hstack((test_eeg_features, test_clinical_data))
    
    # Train and evaluate logistic regression
    print("Training and evaluating logistic regression...")
    lr_model, roc_auc, test_predictions = train_and_evaluate_logreg(
        train_combined, test_combined, train_y, test_y
    )
    
    return model, lr_model, roc_auc, test_predictions

def run_semisupervised_rvae_plus_clinical_experiment(train_loader, test_loader, 
                                                   train_clinical_data, test_clinical_data):
    """Run semi-supervised RVAE + clinical features experiment with pre-trained model"""
    print("\n" + "=" * 50)
    print("SEMI-SUPERVISED RVAE + CLINICAL FEATURES EXPERIMENT")
    print("=" * 50)
    
    # Load pre-trained model from hyperparameter search
    model, best_config = load_best_model_from_hyperparam_search('semisupervised_rvae')
    
    # Extract EEG latent features
    print("Extracting latent features...")
    train_eeg_features, train_y, _ = extract_latent_features(model, train_loader)
    test_eeg_features, test_y, _ = extract_latent_features(model, test_loader)
    
    # Combine with clinical features
    print("Combining EEG latent features with clinical features...")
    train_combined = np.hstack((train_eeg_features, train_clinical_data))
    test_combined = np.hstack((test_eeg_features, test_clinical_data))
    
    # Train and evaluate logistic regression
    print("Training and evaluating logistic regression...")
    lr_model, roc_auc, test_predictions = train_and_evaluate_logreg(
        train_combined, test_combined, train_y, test_y
    )
    
    return model, lr_model, roc_auc, test_predictions

def run_all_experiments(data, train_eeg_dataloader, test_eeg_dataloader):
    """Run all experimental pipelines using pre-trained models and return results"""
    
    # Initialize big list to store: train_X, test_X, train_y, test_y for each experiment
    latent_features = {}

    # Extract data components
    train_clinical_data = data['train']['clinical_data']
    test_clinical_data = data['test']['clinical_data']
    train_y = data['train']['response']
    test_y = data['test']['response']
    
    results = {}
    
    print("\n" + "=" * 60)
    print("RUNNING ALL EXPERIMENTS WITH PRE-TRAINED MODELS")
    print("=" * 60)
    
    ######## 1. Clinical features only (no pre-trained model needed) ########
    results['clinical_only'] = run_clinical_only_experiment(
        train_clinical_data, test_clinical_data, train_y, test_y
    )
    plot_confusion_matrix(results['clinical_only'][-1], test_y, "Clinical Only", "clinical_cm")
    
    ######### 2. Unsupervised autoencoder EEG features ########
    results['unsupervised_ae'] = run_unsupervised_ae_experiment(
        train_eeg_dataloader, test_eeg_dataloader
    )
    plot_confusion_matrix(results['unsupervised_ae'][-2], test_y, "Unsupervised Vanilla Autoencoder", "unsup_ae_cm")
    latent_features['unsupervised_ae'] = results['unsupervised_ae'][-1]

    ######### 3. Semi-supervised autoencoder EEG features ########
    results['semisupervised_ae'] = run_semisupervised_experiment(
        train_eeg_dataloader, test_eeg_dataloader
    )
    plot_confusion_matrix(results['semisupervised_ae'][-2], test_y, "Semi-Supervised Autoencoder", "semisup_ae_cm")
    latent_features['semisupervised_ae'] = results['semisupervised_ae'][-1]

    ######### 4. Semi-supervised RVAE EEG features ########
    results['semisupervised_rvae'] = run_semisupervised_rvae_experiment(
            train_eeg_dataloader, test_eeg_dataloader
    )
    plot_confusion_matrix(results['semisupervised_rvae'][-2], test_y, "Semi-Supervised RVAE", "semisup_rvae_cm")
    latent_features['semisupervised_rvae'] = results['semisupervised_rvae'][-1]

    ######### 5. Unsupervised autoencoder EEG + clinical features ########
    results['unsupervised_ae_clinical'] = run_unsupervised_ae_plus_clinical_experiment(
        train_eeg_dataloader, test_eeg_dataloader, 
        train_clinical_data, test_clinical_data
    )
    plot_confusion_matrix(results['unsupervised_ae_clinical'][-1], test_y,
                          "Unsupervised Autoencoder + Clinical Features", "unsup_ae_clinical_cm")
    
    ######### 6. Semi-supervised autoencoder EEG + clinical features ########
    results['semisupervised_ae_clinical'] = run_semisupervised_plus_clinical_experiment(
        train_eeg_dataloader, test_eeg_dataloader, 
        train_clinical_data, test_clinical_data
    )
    plot_confusion_matrix(results['semisupervised_ae_clinical'][-1], test_y,
                          "Semi-Supervised Autoencoder + Clinical Features", "semisup_ae_clinical_cm")
    
    # 7. Semi-supervised RVAE EEG + clinical features
    results['semisupervised_rvae_clinical'] = run_semisupervised_rvae_plus_clinical_experiment(
        train_eeg_dataloader, test_eeg_dataloader, 
        train_clinical_data, test_clinical_data
    )
    plot_confusion_matrix(results['semisupervised_rvae_clinical'][-1], test_y,
                          "Semi-Supervised RVAE + Clinical Features", "semisup_rvae_clinical_cm")
    
    # pickle latent features dict 
    save_path = DATA_DIR / "latent_features_dict.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump(latent_features, f)
        print(f"\nLatent features saved to {relative_path_str(save_path)}\n")

    return results

def print_experiment_summary(results):
    """Print a summary of all experiment results"""
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Experiment':<40} {'ROC-AUC':<10}")
    print("-" * 80)
    
    for experiment_name, result in results.items():
        # Extract ROC-AUC from different result formats
        if isinstance(result, tuple) and len(result) >= 2:
            roc_auc = result[-1]  # Last element is typically ROC-AUC
        else:
            roc_auc = "N/A"
        
        experiment_display = experiment_name.replace('_', ' ').title()
        print(f"{experiment_display:<40} {roc_auc:<10.3f}" if isinstance(roc_auc, (int, float)) 
              else f"{experiment_display:<40} {roc_auc}")
    
    print("=" * 80)