import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from src.config import *
from src.utils import *

def main():
    train_data, test_data = load_train_test_data()

if __name__ == '__main__':
    setup()
    main()
