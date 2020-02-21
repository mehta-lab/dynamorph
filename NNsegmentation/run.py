#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 18:10:01 2019

@author: zqwu
"""
import tensorflow as tf
import numpy as np
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import pickle
from .data import generate_patches, generate_ordered_patches, predict_whole_map
from .models import Segment
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import cv2

#input_params = {
#    'label_input': 'annotation',
#    'x_size': 256,
#    'y_size': 256,
#    'time_slices': 1}
#data_path = '/mnt/comp_micro/Projects/CellVAE/Combined'
#annotated_sites = [f[:9] for f in os.listdir(data_path) if 'Annotations' in f]
#
#annotated_patches = []
#for site in annotated_sites:
#  input_file = os.path.join(data_path, '%s.npy' % site)
#  label_file = os.path.join(data_path, '%s_Annotations.npy' % site)
#  annotated_patches.extend(generate_ordered_patches(input_file, label_file, **input_params))
#with open('/mnt/comp_micro/Projects/CellVAE/Data/NNSegment/Annotations_8Sites.pkl', 'wb') as f:
#  pickle.dump(annotated_patches, f)


#input_params = {
#    'label_input': 'prob',
#    'x_size': 256,
#    'y_size': 256,
#    'label_value_threshold': 0.5,
#    'rotate': True,
#    'mirror': True,
#    'seed': 123}
#data_path = '/mnt/comp_micro/Projects/CellVAE/Combined'
#input_file = os.path.join(data_path, 'D5-Site_0.npy')
#label_file = os.path.join(data_path, 'D5-Site_0_Probabilities.npy')
#dense_annotated_patches = generate_patches(input_file, label_file, n_patches=1000, **input_params)
#with open('/mnt/comp_micro/Projects/CellVAE/Data/NNSegment/Annotations_Dense_1Site_Ver2.pkl', 'wb') as f:
#  pickle.dump(dense_annotated_patches, f)




train_patches = pickle.load(open('/mnt/comp_micro/Projects/CellVAE/Data/NNSegment/Annotations_8Sites.pkl', 'rb'))
train_patches2 = pickle.load(open('/mnt/comp_micro/Projects/CellVAE/Data/NNSegment/Annotations_BGRF_4Sites.pkl', 'rb'))
#train_patches_dense = pickle.load(open('/mnt/comp_micro/Projects/CellVAE/Data/NNSegment/Annotations_Dense_1Site.pkl', 'rb'))
test_patches = [train_patches[i] for i in np.random.choice(np.arange(len(train_patches)), (50,), replace=False)]
combined = train_patches + train_patches2
np.random.shuffle(combined) 


model_path = './temp_save/'
if not os.path.exists(model_path):
  os.mkdir(model_path)
  
model = Segment(input_shape=(256, 256, 2), 
                unet_feat=32,
                fc_layers=[64, 32],
                n_classes=3,
                model_path=model_path)

for st in range(5):
  model.fit(train_patches,
            label_input='annotation',
            n_epochs=200,
            valid_patches=test_patches,
            valid_label_input='annotation')
  model.save(model.model_path + '/stage%d.h5' % st)


for st in range(5):
  model.fit(combined,
            label_input='annotation',
            n_epochs=50,
            valid_patches=test_patches,
            valid_label_input='annotation')
  model.save(model.model_path + '/stage%d.h5' % st)
  


sites = ['D%d-Site_%d' % (i, j) for j in range(9) for i in range(3, 6)]
for site in sites:
  print(site)
  predict_whole_map('/mnt/comp_micro/Projects/CellVAE/Combined/%s.npy' % site, 
                    model, 
                    n_classes=3, 
                    batch_size=8, 
                    n_supp=5)
