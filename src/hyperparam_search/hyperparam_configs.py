# For unsupervised AE
unsup_ae_search_space = {
    'latent_dim': [64, 128],
    'dropout_rate': [0.2, 0.4],
    'lr': [0.001, 0.0001],
}
# Initially 2 × 2 × 2 = 8 combinations

# For semi-supervised AE
semi_ae_search_space = {
    'latent_dim': [64, 128],
    'dropout_rate': [0.2, 0.4],       
    'lr': [0.001],                     # Fix initially
    'recon_weight': [1.0],             # Fix initially
    'pred_weight': [0.1, 0.5]
}
# Initially 2 × 2 × 1 × 1 × 2 = 8 combinations