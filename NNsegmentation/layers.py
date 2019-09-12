#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 17:51:20 2019

@author: zqwu
"""

import tensorflow as tf
import numpy as np
import keras
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Dense, Layer, Input
from sklearn.metrics import roc_auc_score, f1_score

class Reshape(Layer):
  def __init__(self,
               target_shape,
               **kwargs):
    self.target_shape = target_shape
    super(Reshape, self).__init__(**kwargs)

  def build(self, input_shape):
    super(Reshape, self).build(input_shape)

  def call(self, x):
    output = K.reshape(x, self.target_shape)
    return output

  def compute_output_shape(self, input_shape):
    #Input shape: (batch_size * z_size, x_size, y_size, channels)
    #Output shape: (batch_size, x_size, y_size, channels * z_size)
    return tuple([input_shape[0],
                  input_shape[2],
                  input_shape[3],
                  input_shape[4]])

class MergeOnZ(Layer):
  def __init__(self,
               z_size=5,
               unet_feat=32,
               **kwargs):
    self.z_size = z_size
    self.unet_feat = unet_feat
    self.output_dim = self.z_size * self.unet_feat
    super(MergeOnZ, self).__init__(**kwargs)

  def build(self, input_shape):
    super(MergeOnZ, self).build(input_shape)

  def call(self, x):
    x_shape = K.shape(x)
    x = K.reshape(x, [x_shape[0]//self.z_size, # Batch size
                      self.z_size, # z
                      x_shape[1], # x
                      x_shape[2], # y
                      x_shape[3]]) # channels

    output = K.reshape(K.permute_dimensions(x, (0, 2, 3, 1, 4)),
                       [x_shape[0]//self.z_size, # Batch size
                        x_shape[1], # x
                        x_shape[2], # y
                        self.output_dim]) # z * channels
    return output

  def compute_output_shape(self, input_shape):
    #Input shape: (batch_size * z_size, x_size, y_size, channels)
    #Output shape: (batch_size, x_size, y_size, channels * z_size)
    return tuple([input_shape[0], # avoiding None
                  input_shape[1],
                  input_shape[2],
                  self.output_dim])
  
class weighted_binary_cross_entropy(object):
  
  def __init__(self, n_classes=2):
    self.n_classes = n_classes
    self.__name__ = "weighted_binary_cross_entropy"
    
  def __call__(self, y_true, y_pred):
    w = y_true[:, :, :, -1]
    y_true = y_true[:, :, :, :-1]
    loss = K.categorical_crossentropy(y_true, y_pred, from_logits=True) * w
    return loss


class ValidMetrics(keras.callbacks.Callback):
  def __init__(self, valid_data=None, test_data=None):
    self.valid_data = valid_data
    self.test_data = test_data

  def on_epoch_end(self, epoch, logs={}):
    return
    if self.valid_data is not None:
      y_pred = self.model.predict(self.valid_data[0])[:, :, :, 1]
      y_true = self.valid_data[1][:, :, :, 1] > 0.5
      roc = roc_auc_score(y_true.flatten(), y_pred.flatten())
      f1 = f1_score(y_true.flatten(), y_pred.flatten()>0.5)
      print('\r valid-roc-auc: %f  valid-f1: %f\n' % (roc, f1))
    if self.test_data is not None:
      y_pred = self.model.predict(self.test_data[0])[:, :, :, 1]
      y_true = self.test_data[1][:, :, :, 1] > 0.5
      roc = roc_auc_score(y_true.flatten(), y_pred.flatten())
      f1 = f1_score(y_true.flatten(), y_pred.flatten()>0.5)
      print('\r test-roc-auc: %f  test-f1: %f\n' % (roc, f1))
    return
 
