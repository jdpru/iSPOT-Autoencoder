import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from src.configs import N_EPOCHS, LEARNING_RATE, RECONSTRUCTION_WEIGHT, PREDICTION_WEIGHT, DEVICE

def kl_divergence(mu, log_sigma):
    """Calculate KL divergence for variational autoencoder"""
    sigma2 = torch.exp(2 * log_sigma)
    kld = -0.5 * torch.sum(1 + 2*log_sigma - mu.pow(2) - sigma2, dim=1)
    return torch.mean(kld)

def train_semisupervised_rvae(model, train_loader, 
                             n_epochs=N_EPOCHS, 
                             lr=LEARNING_RATE,
                             reconstruction_weight=RECONSTRUCTION_WEIGHT, 
                             prediction_weight=PREDICTION_WEIGHT):
    """Train semi-supervised RVAE with reconstruction, KL divergence, and prediction losses"""
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        
        for batch in train_loader:
            eeg_data = batch['eeg'].to(DEVICE)
            response = batch['response'].float().unsqueeze(1).to(DEVICE)  # Shape: (batch_size, 1)
            
            optimizer.zero_grad()
            reconstruction, prediction, mu, log_sigma = model(eeg_data)
            
            # Flatten input for reconstruction loss
            eeg_flat = eeg_data.view(eeg_data.size(0), -1)  # [batch, input_dim]
            
            # Calculate losses
            recon_loss = F.mse_loss(reconstruction, eeg_flat)
            kl_loss = kl_divergence(mu, log_sigma)
            pred_loss = F.binary_cross_entropy(prediction, response)
            
            # Combined loss
            loss = reconstruction_weight * recon_loss + kl_loss + prediction_weight * pred_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Train Loss: {avg:.4f}")
    
    return model