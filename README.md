# iSPOT-Autoencoder

**Semi-supervised learning for antidepressant treatment response prediction using EEG data**

**Authors:** JD Pruett & Janelle Cheung

## Overview

This project explores whether we can predict which patients will respond to antidepressant medications using brain activity patterns recorded via EEG. We use autoencoder architectures to learn meaningful latent representations from EEG signals, then feed these learned features into logistic regression models for treatment response prediction.

## Dataset

This project uses data from the **iSPOT-D (International Study to Predict Optimized Treatment in Depression)** clinical trial, led by Dr. Leanne Williams at Stanford University. iSPOT-D is a landmark study that collected comprehensive neurophysiological and clinical data to advance personalized medicine for depression treatment.

- **694 patients** total (594 training, 100 test)
- **3 antidepressant treatments**: Escitalopram, Sertraline, Venlafaxine XR 
- **Binary outcome**: Treatment response (responder/non-responder) 
- **EEG features**: 26 scalp electrodes × 116 timepoints = 3,016 features per patient
  - Each electrode records 116 seconds at 500 Hz sampling rate
  - Broadband power calculated per second

*We gratefully acknowledge Dr. Leanne Williams and the iSPOT-D research team for making this valuable dataset available for advancing computational psychiatry research.* 

## Approach

### 1. Baseline: Unsupervised Autoencoder
- Standard encoder-decoder architecture
- Learns latent representations using only reconstruction loss
- Latent features → Logistic regression → Response prediction

### 2. Main Model: Semi-supervised Autoencoder
- **Dual-head architecture**: Decoder + Predictor heads
- **Dual loss function**: 
  - Reconstruction loss (MSE) from decoder
  - Binary cross-entropy loss from predictor head
- Both losses backpropagate to inform the same latent space
- Latent features → Logistic regression → Response prediction

### 3. Advanced Model: Semi-supervised Recurrent VAE (RVAE)
- Incorporates temporal structure with GRU layers
- Variational approach with KL divergence regularization
- Same dual-supervision strategy as above

## Key Features

- **Hyperparameter search** for all model architectures
- **Cross-validation** with proper train/validation splits
- **Multiple baselines**: Clinical features only, EEG only, combined features
- **Evaluation metrics**: ROC-AUC, accuracy, sensitivity, specificity
- **Visualization**: Confusion matrices and model comparisons

## Project Structure

```
src/
├── configs/           # Configuration files and hyperparameters
├── data/             # Data loading and preprocessing utilities  
├── models/           # Autoencoder architectures
├── training/         # Training loops for different models
├── hyperparam_search/ # Grid search implementations
├── experiments.py    # Main experimental pipeline
└── main.py          # Entry point
```

## Usage 

1. **Run hyperparameter search** (optional, pre-trained models available):
   ```bash
   python src/hyperparam_search/run.py
   ```

2. **Run all experiments**:
   ```bash
   python src/main.py
   ```

This will evaluate all model variants and generate performance comparisons.

## Dependencies

- PyTorch
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn 🌊

See `environment.yml` for full environment setup.
