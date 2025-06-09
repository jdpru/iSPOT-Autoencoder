import torch
import torch.nn as nn
import torch.optim as optim
from src.configs import N_EPOCHS, LEARNING_RATE, RECONSTRUCTION_WEIGHT, PREDICTION_WEIGHT, DEVICE

def train_semisupervised_autoencoder(model, train_loader, 
                                   n_epochs=N_EPOCHS, 
                                   lr=LEARNING_RATE,
                                   reconstruction_weight=RECONSTRUCTION_WEIGHT, 
                                   prediction_weight=PREDICTION_WEIGHT):
    """Train semi-supervised autoencoder with both reconstruction and prediction losses"""
    model = model.to(DEVICE)
    reconstruction_criterion = nn.MSELoss()
    prediction_criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(n_epochs):
        total_recon_loss = 0
        total_pred_loss = 0
        total_loss = 0
        
        for batch in train_loader:
            eeg_data = batch['eeg'].to(DEVICE)
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