#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script explains how to use xxxxxxxxxxxxxx.
"""

# Python packages
import os
import sys
import numpy as np
import tqdm
# Setup paths first
_this_dir = os.path.dirname(os.path.abspath(__file__))
_main_dir = os.path.abspath(os.path.join(_this_dir, '..'))
_mrg_dir = os.path.abspath(os.path.join(_main_dir, '..'))
_data_dir = os.path.join(_main_dir, '..', 'data')
_input_dir = os.path.join(_main_dir, '..', 'data', 'input_scml4u')
_output_dir = os.path.join(_main_dir, '..', 'data', 'output_scml4u')

sys.path.append(os.path.join(_main_dir, 'src'))

# MRG packages - import after path setup
import scml4u.dataloader as dataloader
import scml4u.inference_classifier
import scml4u.utils

np.set_printoptions(threshold=sys.maxsize)

dataset_path='mnist'
resize_value=28



print('\n' + "-------------------------------------------------------" + '\n')

dataset = dataloader.DatasetLoader(dataset_path=dataset_path, resize_value=resize_value)

y_true = []
y_pred = []
y_score = []


for idx in tqdm.tqdm(range(len(dataset))):
    label = dataset.get_image_label(idx)
    result = scml4u.inference_classifier.run_inference_classifier(
        image_path= dataset.get_image_path(idx),
        model_type='mlp',
        weights_path=os.path.join( 'results', 'big-mnist4/model_epoch8.pth'),
        resize_value=resize_value,
        num_classes=10,
        channels=3,
        hidden_layers=3,
        display_results=False
    )
    y_true.append(label)
    y_pred.append(result['predicted_class'])
    y_score.append(result['probabilities'])


y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_score = np.array(y_score)  # shape: (n_samples, n_classes)

print('accuracy: ', np.mean(y_true == y_pred))

print('y_true: ', y_true)
print('y_pred: ', y_pred)
print('y_score: ', y_score)

"""
# Plot confusion matrix
scml4u.utils.plot_matrice_de_confusion(
    y_true, y_pred,
    class_names=[str(i) for i in range(10)],
    output_path=os.path.join(output_path, "confusion_matrix.png"),
    to_save=True,
    show=True,
    normalize=True
)
"""

# Compute AUROC (multiclass)
auroc = scml4u.utils.compute_auroc_sklearn(y_true, y_score, binary=False)
print(f"Multiclass AUROC: {auroc:.4f}")

# Plot ROC curve
scml4u.utils.plot_roc_curve_sklearn(
    y_true, y_score,
    binary=False,
    show=True,
    to_save=True,
    output_path=os.path.join("results", "roc_curve.png")
)

# PCA

print("End.")
