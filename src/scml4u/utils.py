#!/usr/bin/env python
# -*- coding: utf-8 -*-

# MRG packages
import os
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt


def compute_auroc_sklearn(y_true, y_score, binary=True):
    # https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc?hl=fr
    if binary:
        return roc_auc_score(y_true, y_score)
    else:
        return roc_auc_score(y_true, y_score, multi_class='ovr')


def roc_curve_sklearn(y_true, y_score, binary=True):
    if binary:
        return roc_curve(y_true, y_score)
    else:
        return roc_curve(y_true, y_score, multi_class='ovr')


def plot_roc_curve_sklearn(y_true, y_score, binary=True, show=False, to_save=True, output_path='roc_curve.png'):
    
    if binary:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
    else:
        
        # Assuming y_true contains class labels and y_score contains probabilities
        n_classes = y_score.shape[1]
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true == i, y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        plt.figure()
        colors = ['darkorange', 'blue', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], lw=2,
                    label=f'ROC curve of class {i} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve - Multiclass')
        plt.legend(loc="lower right")
    
    if show:
        plt.show()
    if to_save:
        plt.savefig(output_path)
    plt.close()

    return


def plot_loss_from_mtx(train_loss_path, val_loss_path, output_path, to_save=True, show=False, figsize=(10, 6)):
        
    import scipy.io
    import matplotlib.pyplot as plt

    train_loss_matrix = scipy.io.mmread(train_loss_path)
    val_loss_matrix = scipy.io.mmread(val_loss_path)

    # Convert to 1D arrays if they're 2D
    if train_loss_matrix.ndim > 1:
        train_loss_matrix = train_loss_matrix.flatten()
    if val_loss_matrix.ndim > 1:
        val_loss_matrix = val_loss_matrix.flatten()

    plt.figure(figsize=figsize)
    plt.plot(train_loss_matrix, label="Train Loss", linewidth=2)
    plt.plot(val_loss_matrix, label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    if to_save:
        plt.savefig(os.path.join(output_path, "loss_curve.png"), dpi=300, bbox_inches='tight')
    if show:
        plt.show(block=False)
        plt.pause(0.001)  # Brief pause to allow GUI to update
            
    plt.close()  # Close the figure to free memory

    return
