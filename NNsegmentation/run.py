#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 18:10:01 2019

@author: zqwu
"""

# Sample scripts for model training

import tensorflow as tf
import numpy as np
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import pickle
from .data import generate_patches, generate_ordered_patches, predict_whole_map
from .models import Segment
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import cv2

# Data path
TRAIN_DATA_PATH = {
  'annotation': '/mnt/comp_micro/Projects/CellVAE/Data/NNSegment/Annotations_8Sites.pkl',
  'RFBG': '/mnt/comp_micro/Projects/CellVAE/Data/NNSegment/Annotations_BGRF_4Sites.pkl'
}

sites = ['D%d-Site_%d' % (i, j) for j in range(9) for i in range(3, 6)]
TEST_DATA_PATH = {
  site: '/mnt/comp_micro/Projects/CellVAE/Combined/%s.npy' % site for site in sites
}

# Training patches from human annotations
train_patches = pickle.load(open(TRAIN_DATA_PATH['annotation'], 'rb'))

# Supplementary training patches from RF predictions (background only)
train_patches2 = pickle.load(open(TRAIN_DATA_PATH['RFBG'], 'rb'))
combined = train_patches + train_patches2
np.random.shuffle(combined) 

# Random patches used for monitoring
test_patches = [train_patches[i] for i in np.random.choice(np.arange(len(train_patches)), (50,), replace=False)]

# Define model
model_path = './temp_save/'
if not os.path.exists(model_path):
    os.mkdir(model_path)
model = Segment(input_shape=(256, 256, 2), # Phase + Retardance
                unet_feat=32,
                fc_layers=[64, 32],
                n_classes=3,
                model_path=model_path)

# In the first phase of training, only use human annotations
for st in range(5):
    model.fit(train_patches,
              label_input='annotation',
              n_epochs=200,
              valid_patches=test_patches,
              valid_label_input='annotation')
    model.save(model.model_path + '/stage%d.h5' % st)

# In the second phase of training, adding in RF background patches to refine edges
for st in range(5):
    model.fit(combined,
              label_input='annotation',
              n_epochs=50,
              valid_patches=test_patches,
              valid_label_input='annotation')
    model.save(model.model_path + '/stage%d.h5' % st)
  
# Generate predictions for all data
for site in TEST_DATA_PATH:
  print(site)
  predict_whole_map(TEST_DATA_PATH[site], 
                    model, 
                    n_classes=3, 
                    batch_size=8, 
                    n_supp=5)
