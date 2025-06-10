import torch
import torch.nn as nn
from src.configs import GRU_HIDDEN_SIZE, LATENT_DIM, AUTOENCODER_INPUT_DIM, N_ELECTRODES

class SemiSupervisedRVAE(nn.Module):
    def __init__(self, input_dim=N_ELECTRODES, latent_dim=LATENT_DIM):
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
        # x: [batch_size, 3016] - flattened EEG data
        batch_size = x.size(0)
        
        # Reshape for GRU: [batch_size, seq_len, features]
        # You need to decide how to reshape 3016 -> (seq_len, features)
        # Option 1: 26 electrodes × 116 timepoints -> (116, 26)
        x_seq = x.view(batch_size, 116, 26)  # [batch_size, 116, 26]
        
        # Encoder
        _, h = self.encoder_gru(x_seq)  # h: [1, batch_size, hidden_size]
        h = h.squeeze(0)  # [batch_size, hidden_size]

        # Latent space
        mu = self.fc_mu(h)          # [batch_size, latent_dim]
        log_sigma = self.fc_logsig(h)  # [batch_size, latent_dim]
        z = self.reparameterize(mu, log_sigma)  # [batch_size, latent_dim]
        
        # Decoder initialization
        h_dec_init = torch.tanh(self.fc_dec_init(z))  # [batch_size, hidden_size]
        h_dec_init = h_dec_init.unsqueeze(0)  # [1, batch_size, hidden_size]
        
        # Decoder - create zero input sequence
        dec_input = torch.zeros_like(x_seq)  # [batch_size, 116, 26]
        dec_out, _ = self.decoder_gru(dec_input, h_dec_init)  # [batch_size, 116, hidden_size]
        
        # Reconstruction - apply to each timestep
        recon_seq = self.fc_recon(dec_out)  # [batch_size, 116, 26]
        
        # Flatten reconstruction to match input shape
        recon = recon_seq.view(batch_size, -1)  # [batch_size, 3016]
        
        # Prediction
        pred = self.predictor(z)  # [batch_size, 1]
    
        return recon, pred, mu, log_sigma

    # def forward(self, x):
    #     # x: [batch, time_steps, n_channels]
    #     _, h = self.encoder_gru(x)
    #     h = h.squeeze(0)
    #     mu = self.fc_mu(h)
    #     log_sigma = self.fc_logsig(h)
    #     z = self.reparameterize(mu, log_sigma)
    #     # Decode sequence
    #     h_dec_init = torch.tanh(self.fc_dec_init(z)).unsqueeze(0)
    #     dec_input = torch.zeros_like(x)
    #     dec_out, _ = self.decoder_gru(dec_input, h_dec_init)
    #     # project each timestep
    #     recon_seq = self.fc_recon(dec_out)  # [batch, time_steps, n_channels]

    #     batch = recon_seq.size(0)
    #     recon = recon_seq.view(batch, -1)       # [batch, input_dim]
    #     pred = self.predictor(z)               # [batch,1]
    #     return recon, pred, mu, log_sigma