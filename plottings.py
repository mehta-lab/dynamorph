#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 15:20:21 2019

@author: michaelwu
"""
import numpy as np
import cv2
import os
from NNsegmentation.models import Segment
from NNsegmentation.data import predict_whole_map

RAW_DATA_PATH = '/mnt/comp_micro/Projects/CellVAE/Combined'
color_mg = np.array([240, 94, 56], dtype='uint8')
color_nonmg = np.array([66, 101, 251], dtype='uint8')
color_bg = np.array([150, 150, 150], dtype='uint8')
color_fg = (color_mg * 0.7 + color_nonmg * 0.3).astype('uint8')

raw_input = np.load(RAW_DATA_PATH + '/D5-Site_0.npy')
raw_input = raw_input[0, :, :, 0:1]
cv2.imwrite('/home/michaelwu/fig2_raw.png', raw_input)

annotations = np.load(RAW_DATA_PATH + '/D5-Site_0_Annotations.npy')
annotations = annotations[0]
mat = np.zeros((annotations.shape[0], annotations.shape[1], 3), dtype='uint8')
mat[:, :] = np.expand_dims((raw_input / 256).astype('uint8'), 2)
alpha = 0.3
mat[np.where(annotations == 1)[:2]] = alpha * mat[np.where(annotations == 1)[:2]] + (1 - alpha) * color_bg.reshape((1, 3))
mat[np.where(annotations == 2)[:2]] = alpha * mat[np.where(annotations == 2)[:2]] + (1 - alpha) * color_mg.reshape((1, 3))
mat[np.where(annotations == 3)[:2]] = alpha * mat[np.where(annotations == 3)[:2]] + (1 - alpha) * color_nonmg.reshape((1, 3))
cv2.imwrite('/home/michaelwu/fig2_annotations.png', mat)

RF_predictions = np.load(RAW_DATA_PATH + '/D5-Site_0_RFProbabilities.npy')
RF_predictions = RF_predictions[0]
RF_bg = RF_predictions[:, :, 0:1]
RF_fg = RF_predictions[:, :, 1:2] + RF_predictions[:, :, 2:3]
mat = np.zeros((annotations.shape[0], annotations.shape[1], 3), dtype='uint8')
mat[:, :] = np.expand_dims((raw_input / 256).astype('uint8'), 2)
positions = np.where((RF_bg > 0.5))[:2]
alpha = 0.3
mat[positions] = alpha * mat[positions] + (1 - alpha) * color_bg.reshape((1, 3))
cv2.imwrite('/home/michaelwu/fig2_rf_predictions.png', mat)

mat = np.zeros((annotations.shape[0], annotations.shape[1], 3), dtype='uint8')
mat[:, :] = np.expand_dims((raw_input / 256).astype('uint8'), 2)
mg_positions = np.where(RF_predictions[:, :, 1] > 0.5)[:2]
nonmg_positions = np.where(RF_predictions[:, :, 2] > 0.5)[:2]
alpha = 0.7
mat = mat * (1 - (alpha * RF_predictions[:, :, 0:1])) + np.ones_like(mat) * color_bg.reshape((1, 1, 3)) * (alpha * RF_predictions[:, :, 0:1])
mat[mg_positions] = (mat * (1 - (alpha * RF_predictions[:, :, 1:2])) + np.ones_like(mat) * color_mg.reshape((1, 1, 3)) * (alpha * RF_predictions[:, :, 1:2]))[mg_positions]
mat[nonmg_positions] = (mat * (1 - (alpha * RF_predictions[:, :, 2:3])) + np.ones_like(mat) * color_nonmg.reshape((1, 1, 3)) * (alpha * RF_predictions[:, :, 2:3]))[nonmg_positions]
cv2.imwrite('/home/michaelwu/off_fig2_rf_predictions_annotation_only.png', mat)

model = Segment(input_shape=(256, 256, 2), 
                unet_feat=32,
                fc_layers=[64, 32],
                n_classes=3,
                model_path='./NNsegmentation/temp_save')
model.load(model.model_path + '/final.h5')
NN_predictions = predict_whole_map(np.load(RAW_DATA_PATH + '/D5-Site_0.npy')[20:21], model, n_supp=20)[0]
mat = np.zeros((annotations.shape[0], annotations.shape[1], 3), dtype='uint8')
mat[:, :] = np.expand_dims((raw_input / 256).astype('uint8'), 2)
mg_positions = np.where(NN_predictions[:, :, 1] > 0.5)[:2]
nonmg_positions = np.where(NN_predictions[:, :, 2] > 0.5)[:2]
alpha = 0.7
mat = mat * (1 - (alpha * NN_predictions[:, :, 0:1])) + np.ones_like(mat) * color_bg.reshape((1, 1, 3)) * (alpha * NN_predictions[:, :, 0:1])
mat[mg_positions] = (mat * (1 - (alpha * NN_predictions[:, :, 1:2])) + np.ones_like(mat) * color_mg.reshape((1, 1, 3)) * (alpha * NN_predictions[:, :, 1:2]))[mg_positions]
mat[nonmg_positions] = (mat * (1 - (alpha * NN_predictions[:, :, 2:3])) + np.ones_like(mat) * color_nonmg.reshape((1, 1, 3)) * (alpha * NN_predictions[:, :, 2:3]))[nonmg_positions]
cv2.imwrite('/home/michaelwu/off_fig2_nn_predictions.png', mat)

model.load(model.model_path + '/stage0_0.h5')
NN_predictions = predict_whole_map(np.load(RAW_DATA_PATH + '/D5-Site_0.npy')[20:21], model, n_supp=20)[0]
mat = np.zeros((annotations.shape[0], annotations.shape[1], 3), dtype='uint8')
mat[:, :] = np.expand_dims((raw_input / 256).astype('uint8'), 2)
mg_positions = np.where(NN_predictions[:, :, 1] > 0.5)[:2]
nonmg_positions = np.where(NN_predictions[:, :, 2] > 0.5)[:2]
alpha = 0.7
mat = mat * (1 - (alpha * NN_predictions[:, :, 0:1])) + np.ones_like(mat) * color_bg.reshape((1, 1, 3)) * (alpha * NN_predictions[:, :, 0:1])
mat[mg_positions] = (mat * (1 - (alpha * NN_predictions[:, :, 1:2])) + np.ones_like(mat) * color_mg.reshape((1, 1, 3)) * (alpha * NN_predictions[:, :, 1:2]))[mg_positions]
mat[nonmg_positions] = (mat * (1 - (alpha * NN_predictions[:, :, 2:3])) + np.ones_like(mat) * color_nonmg.reshape((1, 1, 3)) * (alpha * NN_predictions[:, :, 2:3]))[nonmg_positions]
cv2.imwrite('/home/michaelwu/off_fig2_nn_predictions_annotation_only.png', mat)