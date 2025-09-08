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
output_path  = os.path.join(_output_dir, 'results', 'mnist')

_ = scml4u.train_classifier.run_train_MLP(epochs=10, 
                                      resize_value=28, 
                                      hidden_layers=5,
                                      dataset_path='mnist', 
                                      output_path='results', 
                                      show=False, 
                                      to_save=True)
"""
scml4u.train_classifier.run_train_GNN(epochs=100,
                                      resize_value=28,
                                      n_blocks=10,
                                      dataset_path='mnist', 
                                      output_path='examples/results', 
                                      show=False, 
                                      to_save=True)
"""

print("End.")
