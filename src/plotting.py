import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from src.configs import *
from src.utils import *
from sklearn.metrics import confusion_matrix, accuracy_score

def plot_confusion_matrix(y_pred, y_true, title, filename):
    """
    Plot a confusion matrix for binary classification. 
    """
    
    # Calculate confusion matrix and metrics
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    
    # Calculate additional metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)  # True Positive Rate
    specificity = tn / (tn + fp)  # True Negative Rate
    
    # Create the plot
    plt.figure(figsize=(6, 5))
    
    # Plot heatmap with clear labels
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Non-Responder (0)', 'Responder (1)'],
                yticklabels=['Non-Responder (0)', 'Responder (1)'],
                square=True, linewidths=0.5)
    
    # Add labels and title with metrics
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title(f'{title}\nAccuracy: {accuracy:.3f} | Sensitivity: {sensitivity:.3f} | Specificity: {specificity:.3f}', 
              fontsize=12, pad=20)
    
    # Add count labels in each cell for clarity
    plt.text(0.5, 0.2, f'True Negatives\n{tn}', ha='center', va='center', fontweight='bold')
    plt.text(1.5, 0.2, f'False Positives\n{fp}', ha='center', va='center', fontweight='bold')  
    plt.text(0.5, 1.2, f'False Negatives\n{fn}', ha='center', va='center', fontweight='bold')
    plt.text(1.5, 1.2, f'True Positives\n{tp}', ha='center', va='center', fontweight='bold')
    
    save_path = FIGURES_DIR / f"{filename}.png"
    save_figure(save_path)

def save_figure(filepath, fig=None, dpi=300, bbox_inches='tight'):
    '''
    Helper function to save a matplotlib figure.
    '''
    if fig is None:
        plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
        plt.close()
    else:
        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
        plt.close(fig)
    
    print("\n\t", relative_path_str(filepath), "\n")