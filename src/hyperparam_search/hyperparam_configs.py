# For unsupervised vanilla AE
unsup_ae_search_space = {
    'latent_dim': [32, 64],
    'dropout_rate': [0.2, 0.4],
    'lr': [0.001, 0.0001],
}
# Initially 2 × 2 × 2 = 8 combinations

# For semi-supervised vanilla AE
semi_ae_search_space = {
    'latent_dim': [32, 64],
    'dropout_rate': [0.2, 0.4],       
    'lr': [0.001],                     # Fix initially
    'recon_weight': [1.0],             # Fix initially
    'pred_weight': [0.5, 1.5]
}
# Initially 2 × 2 × 1 × 1 × 2 = 8 combinations

# For semi-supervised RVAE
semi_rvae_search_space = {
    'latent_dim': [32, 64],
    'recon_weight': [1.0],
    'pred_weight': [0.5, 1.5],
    'dropout_rate': [0.2, 0.4],
    'lr': [0.001, 0.0001],
}