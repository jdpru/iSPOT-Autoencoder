import torch
import torch.nn as nn
from src.configs import hidden_size, LATENT_DIM, N_ELECTRODES

# TO-DO: Make sure this matches the other model configs

class SemiSupervisedRVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_gru = nn.GRU(input_size=N_ELECTRODES,
                                  hidden_size=hidden_size,
                                  batch_first=True)
        self.fc_mu      = nn.Linear(hidden_size, LATENT_DIM)
        self.fc_logsig = nn.Linear(hidden_size, LATENT_DIM)
        self.fc_dec_init = nn.Linear(LATENT_DIM, hidden_size)
        self.decoder_gru = nn.GRU(input_size=N_ELECTRODES,
                                  hidden_size=hidden_size,
                                  batch_first=True)
        self.fc_recon = nn.Linear(hidden_size, N_ELECTRODES)
        self.classifier = nn.Sequential(
            nn.Linear(LATENT_DIM, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_sigma):
        sigma = torch.exp(log_sigma)
        eps   = torch.randn_like(sigma)
        return mu + sigma * eps

    def forward(self, x):
        _, h = self.encoder_gru(x)
        h = h.squeeze(0)
        mu = self.fc_mu(h)
        log_sigma = self.fc_logsig(h)
        z = self.reparameterize(mu, log_sigma)
        h_dec_init = torch.tanh(self.fc_dec_init(z)).unsqueeze(0)
        dec_input = torch.zeros_like(x)
        dec_out, _ = self.decoder_gru(dec_input, h_dec_init)
        recon = self.fc_recon(dec_out)
        pred  = self.classifier(z)
        return recon, pred, mu, log_sigma