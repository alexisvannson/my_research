#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script explains how to use xxxxxxxxxxxxxx.
"""

# Python packages
import os
import sys
_this_dir = os.path.dirname(os.path.abspath(__file__))
_main_dir = os.path.abspath(os.path.join(_this_dir, '..'))
_mrg_dir = os.path.abspath(os.path.join(_main_dir, '..'))
_data_dir = os.path.join(_main_dir, '..', 'data')
_input_dir = os.path.join(_main_dir, '..', 'data', 'input_scml4u')
_output_dir = os.path.join(_main_dir, '..', 'data', 'output_scml4u')
#sys.path.append(os.path.join(_main_dir, '..'))  # to see /blabla4u
#sys.path.append(os.path.join(_main_dir))  # to see /blabla4u/src

# ????
sys.path.append(os.path.join(_main_dir, 'src'))


# Python packages
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

# MRG packages
import scml4u


print('\n' + "-------------------------------------------------------" + '\n')

dataset_path = os.path.join(_input_dir, 'mnist')
output_path  = os.path.join(_output_dir, 'mnist')

prediction = scml4u.inference_classifier.run_inference_classifier(
    image_path="/Users/philippevannson/Desktop/ongoing_stuff/scml4u/mnist/8/test_00061.png",
    model_type='mlp',
    weights_path='results/model_epoch9.pth',
    resize_value=28,
    num_classes=10,
    channels=3,
    hidden_layers=5,
)


print(prediction)


print("End.")
