import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from src.config import *
from src.utils import *
from src.experiments import *
from torch.utils.data import DataLoader

def main():
    # Load data
    train_dataset, test_dataset = load_train_test_datasets()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Run experiments
    baseline_results = run_baseline_experiment(train_loader, test_loader)
    semisup_results = run_semisupervised_experiment(train_loader, test_loader)

if __name__ == '__main__':
    setup()
    main()
