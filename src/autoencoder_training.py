import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import numpy as np
from src.configs import *

def train_unsupervised_autoencoder(model, train_loader, n_epochs=N_EPOCHS, lr=LEARNING_RATE):
    """Train baseline unsupervised autoencoder with reconstruction loss only"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for batch in train_loader:
            eeg_data = batch['eeg']
            
            optimizer.zero_grad()
            reconstruction, latent = model(eeg_data)
            loss = criterion(reconstruction, eeg_data)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{n_epochs}], Recon Loss: {avg_loss:.6f}')
    
    return model

def train_semisupervised_autoencoder(model, train_loader, 
                                   n_epochs=N_EPOCHS, 
                                   lr=LEARNING_RATE,
                                   reconstruction_weight=RECONSTRUCTION_WEIGHT, 
                                   prediction_weight=PREDICTION_WEIGHT):
    """Train semi-supervised autoencoder with both reconstruction and prediction losses"""
    reconstruction_criterion = nn.MSELoss()
    prediction_criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(n_epochs):
        total_recon_loss = 0
        total_pred_loss = 0
        total_loss = 0
        
        for batch in train_loader:
            eeg_data = batch['eeg']
            response = batch['response'].unsqueeze(1)  # Shape: (batch_size, 1)
            
            optimizer.zero_grad()
            reconstruction, prediction, latent = model(eeg_data)
            
            # Two losses
            recon_loss = reconstruction_criterion(reconstruction, eeg_data)
            pred_loss = prediction_criterion(prediction, response)
            
            # Combined loss
            total_batch_loss = (reconstruction_weight * recon_loss + 
                              prediction_weight * pred_loss)
            
            total_batch_loss.backward()
            optimizer.step()
            
            total_recon_loss += recon_loss.item()
            total_pred_loss += pred_loss.item()
            total_loss += total_batch_loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_recon = total_recon_loss / len(train_loader)
            avg_pred = total_pred_loss / len(train_loader)
            avg_total = total_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{n_epochs}], Recon Loss: {avg_recon:.6f}, '
                  f'Pred Loss: {avg_pred:.6f}, Total Loss: {avg_total:.6f}')
    
    return model

def extract_latent_features(model, data_loader):
    """Extract latent features from trained autoencoder"""
    model.eval()
    latent_features = []
    responses = []
    
    with torch.no_grad():
        for batch in data_loader:
            eeg_data = batch['eeg']
            response = batch['response']
            
            if hasattr(model, 'predictor'):  # Semi-supervised model
                _, _, latent = model(eeg_data)
            else:  # Baseline model
                _, latent = model(eeg_data)
            
            latent_features.append(latent.cpu().numpy())
            responses.append(response.cpu().numpy())
    
    return np.vstack(latent_features), np.concatenate(responses)

def train_and_evaluate_logreg(train_features, test_features, train_labels, 
                                        test_labels):
    """Train logistic regression on latent features and evaluate"""
    # Train logistic regression
    lr_model = LogisticRegression(random_state=SEED, max_iter=1000)
    lr_model.fit(train_features, train_labels)
    
    # Predict on test set
    test_predictions = lr_model.predict(test_features)
    test_probabilities = lr_model.predict_proba(test_features)[:, 1]
    test_accuracy = accuracy_score(test_labels, test_predictions)
    
    print(f"\nLogistic Regression Results:")
    print(f"Test Accuracy: {test_accuracy:.3f}")
    print(f"ROC AUC Score: {roc_auc_score(test_labels, test_probabilities):.3f}")
    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, test_predictions))
    print("\nClassification Report:")
    print(classification_report(test_labels, test_predictions))
    
    return lr_model, test_predictions