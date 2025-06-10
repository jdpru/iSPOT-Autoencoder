
# Expanded hyperparameter search spaces for GPU utilization

# For unsupervised vanilla AE - 144 combinations
unsup_ae_search_space = {
    'latent_dim': [16, 32, 64, 128],           # 4 options - explore smaller/larger representations
    'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5], # 5 options - wider dropout range
    'lr': [0.01, 0.003, 0.001, 0.0003, 0.0001, 0.00003], # 6 options - log scale exploration
    'batch_size': [16, 32, 64]                 # 3 options - GPU can handle larger batches
}
# Total: 4 × 5 × 6 × 3 = 360 combinations

# For semi-supervised vanilla AE - 240 combinations  
semi_ae_search_space = {
    'latent_dim': [16, 32, 64, 128],           # 4 options
    'dropout_rate': [0.1, 0.2, 0.3, 0.4, 0.5], # 5 options
    'lr': [0.003, 0.001, 0.0003, 0.0001],     # 4 options - focus on promising range
    'recon_weight': [0.5, 1.0, 2.0],          # 3 options - vary reconstruction importance
    'pred_weight': [0.1, 0.5, 1.0, 1.5, 2.0], # 5 options - vary prediction importance
    'batch_size': [32, 64]                     # 2 options
}
# Total: 4 × 5 × 4 × 3 × 5 × 2 = 1,200 combinations

# For semi-supervised RVAE - 480 combinations
semi_rvae_search_space = {
    'latent_dim': [16, 32, 64, 128],           # 4 options
    'recon_weight': [0.5, 1.0, 2.0, 3.0],     # 4 options - RVAE may need different recon weights
    'pred_weight': [0.1, 0.5, 1.0, 1.5, 2.0], # 5 options
    'lr': [0.003, 0.001, 0.0003, 0.0001],     # 4 options
    'beta': [0.1, 0.5, 1.0, 2.0],             # 4 options - KL divergence weight (new!)
    'batch_size': [32, 64]                     # 2 options
}
# Total: 4 × 4 × 5 × 4 × 4 × 2 = 1,280 combinations