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
    Plot a well-labeled confusion matrix for binary classification.
    
    Args:
        y_pred: Predicted labels (0/1)
        y_true: True labels (0/1) 
        title: Title for the plot
        save_name: Optional filename to save (without extension)
    """
    from sklearn.metrics import confusion_matrix, accuracy_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    
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