# For unsupervised vanilla AE
unsup_ae_search_space = {
    'latent_dim': [64, 128],
    'dropout_rate': [0.2, 0.4],
    'lr': [0.001, 0.0001],
}
# Initially 2 × 2 × 2 = 8 combinations

# For semi-supervised vanilla AE
semi_ae_search_space = {
    'latent_dim': [64, 128],
    'dropout_rate': [0.2, 0.4],       
    'lr': [0.001],                     # Fix initially
    'recon_weight': [1.0],             # Fix initially
    'pred_weight': [0.1, 0.5]
}
# Initially 2 × 2 × 1 × 1 × 2 = 8 combinations


# For unsupervised RVAE
unsup_rvae_search_space = {
    'latent_dim': [64, 128],
    'hidden_size': [64, 128],
    'latent_dim': [64, 128],
    'recon_weight': [0.1, 0.5, 1.0],
    'dropout_rate': [0.2, 0.4],
    'lr': [0.001, 0.0001],
}
# Initially 2 × 2 × 2 × 3 × 2 = 48 combinations

# For semi-supervised RVAE
semi_rvae_search_space = {
    'latent_dim': [64, 128],
    'recon_weight': [0.1, 0.5, 1.0],
    'pred_weight': [0.1, 0.5, 1.0],
    'dropout_rate': [0.2, 0.4],
    'lr': [0.001, 0.0001],
}
# Initially 2 × 2 × 2 × 3 × 3 × 2 = 72 combinations