import torch
import torch.nn as nn
from src.configs import GRU_HIDDEN_SIZE, LATENT_DIM, AUTOENCODER_INPUT_DIM, DEVICE

class SemiSupervisedRVAE(nn.Module):
    def __init__(self, input_dim=AUTOENCODER_INPUT_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.name = 'SemiSupervisedRVAE'
        self.encoder_gru = nn.GRU(input_size=input_dim,
                                  hidden_size=GRU_HIDDEN_SIZE,
                                  batch_first=True)
        self.fc_mu      = nn.Linear(GRU_HIDDEN_SIZE, latent_dim)
        self.fc_logsig = nn.Linear(GRU_HIDDEN_SIZE, latent_dim)
        self.fc_dec_init = nn.Linear(latent_dim, GRU_HIDDEN_SIZE)
        self.decoder_gru = nn.GRU(input_size=input_dim,
                                  hidden_size=GRU_HIDDEN_SIZE,
                                  batch_first=True)
        self.fc_recon = nn.Linear(GRU_HIDDEN_SIZE, input_dim)
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def reparameterize(self, mu, log_sigma):
        sigma = torch.exp(log_sigma)
        eps   = torch.randn_like(sigma)
        return mu + sigma * eps

    def forward(self, x):
        # x: [batch, time_steps, n_channels]
        _, h = self.encoder_gru(x)
        h = h.squeeze(0)
        mu = self.fc_mu(h)
        log_sigma = self.fc_logsig(h)
        z = self.reparameterize(mu, log_sigma)
        # Decode sequence
        h_dec_init = torch.tanh(self.fc_dec_init(z)).unsqueeze(0)
        dec_input = torch.zeros_like(x)
        dec_out, _ = self.decoder_gru(dec_input, h_dec_init)
        # project each timestep
        recon_seq = self.fc_recon_step(dec_out)  # [batch, time_steps, n_channels]

        batch = recon_seq.size(0)
        recon = recon_seq.view(batch, -1)       # [batch, input_dim]
        pred = self.classifier(z)               # [batch,1]
        return recon, pred, mu, log_sigma