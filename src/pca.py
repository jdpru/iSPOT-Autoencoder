import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.configs import *

# Load the latent features
with open(DATA_DIR / "latent_features_dict.pkl", "rb") as f:
    latent_dict = pickle.load(f)

# Extract semi-supervised autoencoder features
model_data = latent_dict['semisupervised_ae']
train_X = model_data['train_X']  # (594, 64)
train_y = model_data['train_y']  # (594,)
test_X = model_data['test_X']    # (100, 64)
test_y = model_data['test_y']    # (100,)

print(f"Train features shape: {train_X.shape}")
print(f"Test features shape: {test_X.shape}")

# Combine train and test for PCA fitting (common practice)
all_X = np.vstack([train_X, test_X])
all_y = np.hstack([train_y, test_y])

print(f"Combined features shape: {all_X.shape}")

# Optional: Standardize the features before PCA
scaler = StandardScaler()
all_X_scaled = scaler.fit_transform(all_X)

# Fit PCA with 2 components
pca = PCA(n_components=2)
all_X_pca = pca.fit_transform(all_X_scaled)

print(f"PCA transformed shape: {all_X_pca.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.3f}")

# Split back into train and test
train_X_pca = all_X_pca[:len(train_X)]  # First 594 samples
test_X_pca = all_X_pca[len(train_X):]   # Last 100 samples

# Create visualization
plt.figure(figsize=(12, 5))

# Plot 1: Train data
plt.subplot(1, 2, 1)
colors = ['red', 'blue']
labels = ['Non-Responder', 'Responder']
for i, (color, label) in enumerate(zip(colors, labels)):
    mask = train_y == i
    plt.scatter(train_X_pca[mask, 0], train_X_pca[mask, 1], 
               c=color, label=label, alpha=0.7, s=30)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Train Data - PCA of Semi-Supervised AE Latent Features')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Test data
plt.subplot(1, 2, 2)
for i, (color, label) in enumerate(zip(colors, labels)):
    mask = test_y == i
    plt.scatter(test_X_pca[mask, 0], test_X_pca[mask, 1], 
               c=color, label=label, alpha=0.7, s=30)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Test Data - PCA of Semi-Supervised AE Latent Features')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '2_pca_semisup_ae_latent_features.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n" + "="*50)
print("PCA ANALYSIS SUMMARY")
print("="*50)
print(f"Original feature dimension: {train_X.shape[1]}")
print(f"Reduced dimension: {all_X_pca.shape[1]}")
print(f"PC1 explains {pca.explained_variance_ratio_[0]:.1%} of variance")
print(f"PC2 explains {pca.explained_variance_ratio_[1]:.1%} of variance")
print(f"Total variance captured: {pca.explained_variance_ratio_.sum():.1%}")

# Optional: Save PCA results
pca_results = {
    'train_X_pca': train_X_pca,
    'test_X_pca': test_X_pca,
    'train_y': train_y,
    'test_y': test_y,
    'pca_model': pca,
    'scaler': scaler,
    'explained_variance_ratio': pca.explained_variance_ratio_
}

with open(DATA_DIR / 'semisupervised_ae_pca_results.pkl', 'wb') as f:
    pickle.dump(pca_results, f)
    
print(f"\nPCA results saved to: {DATA_DIR / 'semisupervised_ae_pca_results.pkl'}")

