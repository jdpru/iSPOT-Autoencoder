import torch
import torch.nn as nn
import torch.optim as optim
from src.configs import N_EPOCHS, LEARNING_RATE, AUTOENCODER_INPUT_DIM, LATENT_DIM, DROPOUT_RATE, DEVICE

def train_unsupervised_autoencoder(model, train_loader, n_epochs=N_EPOCHS, lr=LEARNING_RATE):
    """Train baseline unsupervised autoencoder with reconstruction loss only"""
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        for batch in train_loader:
            eeg_data = batch['eeg'].to(DEVICE)
            
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