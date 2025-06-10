# For unsupervised vanilla AE
unsup_ae_search_space = {
    'latent_dim': [16, 32, 64, 128],
    'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
    'lr': [0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003],
    'batch_size': [16, 32, 64]
}
# Total: 360 combinations

# For semi-supervised vanilla AE
semi_ae_search_space = {
    'latent_dim': [16, 32, 64, 128],
    'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
    'lr': [0.003, 0.001, 0.0003, 0.0001],
    'recon_weight': [0.5, 1.0, 2.0],
    'pred_weight': [0.1, 0.5, 1.0, 1.5, 2.0],
    'batch_size': [32, 64]
}
# Total: 1,200 combinations

# For semi-supervised RVAE
semi_rvae_search_space = {
    'latent_dim': [16, 32, 64, 128],
    'recon_weight': [0.5, 1.0, 2.0, 3.0],
    'pred_weight': [0.1, 0.5, 1.0, 1.5, 2.0],
    'lr': [0.003, 0.001, 0.0003, 0.0001],
    'beta': [0.1, 0.5, 1.0, 2.0],
    'batch_size': [32, 64]
}
# Total: 1,280 combinations