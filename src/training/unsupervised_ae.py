import torch
import torch.nn as nn
import torch.optim as optim
from src.configs import N_EPOCHS, LEARNING_RATE, L1_WEIGHT, DEVICE
from src.utils import l1_penalty

def train_unsupervised_autoencoder(model, train_loader, n_epochs=N_EPOCHS, lr=LEARNING_RATE, l1_weight=L1_WEIGHT):
    """Train baseline unsupervised autoencoder with L1 regularization on encoder"""
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        total_recon_loss = 0
        total_l1_loss = 0

        for batch in train_loader:
            eeg_data = batch['eeg'].to(DEVICE)
            
            optimizer.zero_grad()
            reconstruction, latent = model(eeg_data)
            
            # Reconstruction loss
            recon_loss = criterion(reconstruction, eeg_data)
            
            # L1 penalty on encoder
            l1_loss = l1_penalty(model, l1_weight, encoder_only=True)
            
            # Total loss
            total_batch_loss = recon_loss + l1_loss

            total_batch_loss.backward()
            optimizer.step()
            
            total_loss += total_batch_loss.item()
            total_recon_loss += recon_loss.item()
            total_l1_loss += l1_loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            avg_recon = total_recon_loss / len(train_loader)
            avg_l1 = total_l1_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{n_epochs}], Total: {avg_loss:.6f}, '
                  f'Recon: {avg_recon:.6f}, L1: {avg_l1:.6f}')
    
    return model