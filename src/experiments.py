from src.models.baseline_autoencoder import BaselineAutoencoder
from src.models.semisupervised_autoencoder import SemiSupervisedAutoencoder
from src.training import *
from src.utils import *

def run_baseline_experiment(train_loader, test_loader):
    """Run complete baseline autoencoder experiment"""
    print("\n" + "=" * 50)
    print("BASELINE AUTOENCODER EXPERIMENT")
    print("=" * 50)
    
    # Initialize model
    model = BaselineAutoencoder()
    
    # Train autoencoder
    print("Training baseline autoencoder...")
    model = train_baseline_autoencoder(model, train_loader, n_epochs=N_EPOCHS)
    
    # Extract features
    print("Extracting latent features...")
    train_X, train_y = extract_latent_features(model, train_loader)
    test_X, test_y = extract_latent_features(model, test_loader)
    
    # Train and evaluate logistic regression
    print("Training and evaluating logistic regression...")
    lr_model, predictions = train_and_evaluate_logistic_regression(
        train_X, train_y, test_X, test_y
    )
    
    return model, lr_model, predictions

def run_semisupervised_experiment(train_loader, test_loader):
    """Run complete semi-supervised autoencoder experiment"""
    print("\n" + "=" * 50)
    print("SEMI-SUPERVISED AUTOENCODER EXPERIMENT")
    print("=" * 50)
    
    # Initialize model
    model = SemiSupervisedAutoencoder()
    
    # Train autoencoder
    print("Training semi-supervised autoencoder...")
    model = train_semisupervised_autoencoder(
        model, train_loader, n_epochs=N_EPOCHS, 
        reconstruction_weight=RECONSTRUCTION_WEIGHT, prediction_weight=PREDICTION_WEIGHT
    )
    
    # Extract features
    print("Extracting latent features...")
    train_X, train_y = extract_latent_features(model, train_loader)
    test_X, test_y = extract_latent_features(model, test_loader)
    
    # Train and evaluate logistic regression
    print("Training and evaluating logistic regression...")
    lr_model, predictions = train_and_evaluate_logistic_regression(
        train_X, train_y, test_X, test_y
    )
    
    return model, lr_model, predictions