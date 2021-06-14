#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 17:51:20 2019

@author: zqwu
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
# from keras import backend as K
# from keras.models import Model, load_model
# from keras.layers import Dense, Layer, Input
from sklearn.metrics import roc_auc_score, f1_score


class SplitSlice(keras.layers.Layer):
    """ Customized layer for tensor reshape
    
    Used for 2.5D segmentation
    """
    def __init__(self,
                 n_channels,
                 x_size,
                 y_size,
                 **kwargs):
        self.n_channels = n_channels
        self.x_size = x_size
        self.y_size = y_size
        super(SplitSlice, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SplitSlice, self).build(input_shape)

    def call(self, x):
        # Input shape: (batch_size, n_channel, n_slice, x_size, y_size)
        # Output shape: (batch_size * n_slice, n_channel, x_size, y_size)
        _x = keras.backend.permute_dimensions(x, (0, 2, 1, 3, 4))
        target_shape = (-1, self.n_channels, self.x_size, self.y_size)
        output = keras.backend.reshape(_x, target_shape)
        return output

    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0], # batch
                      input_shape[1], # c
                      input_shape[-2], # x
                      input_shape[-1]]) # y


class MergeSlices(keras.layers.Layer):
    """ Customized layer for tensor reshape
    """
    def __init__(self,
                 n_slice=5,
                 n_channel=32,
                 **kwargs):
        self.n_slice = n_slice
        self.n_channel = n_channel
        self.output_dim = self.n_slice * self.n_channel
        super(MergeSlices, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MergeSlices, self).build(input_shape)

    def call(self, x):
        # Input shape: (batch_size * n_slice, n_channel, x_size, y_size)
        # Output shape: (batch_size, n_slice * n_channel, x_size, y_size)
        x_shape = keras.backend.shape(x)
        _x = keras.backend.reshape(x, [x_shape[0]//self.n_slice, # Batch size
                          self.n_slice, # n_slice
                          self.n_channel, # n_channel
                          x_shape[2], # x
                          x_shape[3]]) # y

        output = keras.backend.reshape(_x, [x_shape[0]//self.n_slice, # Batch size
                                self.output_dim, # n_slice * n_channel
                                x_shape[2], # x
                                x_shape[3]]) # y
        return output

    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0], # avoiding None
                      self.output_dim,
                      input_shape[2],
                      input_shape[3]])


class weighted_binary_cross_entropy(object):
    """ Customized loss function
    """
    def __init__(self, n_classes=2):
        self.n_classes = n_classes
        self.__name__ = "weighted_binary_cross_entropy"
      
    def __call__(self, y_true, y_pred):
        """
        Args:
            y_true (tensor): in shape (batch_size, x_size, y_size, n_classes + 1)
                first `n_classes` slices of the last dimension are labels
                last slice of the last dimension is weight
            y_pred (tensor): in shape (batch_size, x_size, y_size, n_classes)
                model predictions
        
        """

        w = y_true[:, -1]
        y_true = y_true[:, :-1]
        
        # Switch to channel last form
        y_true = keras.backend.permute_dimensions(y_true, (0, 2, 3, 1))
        y_pred = keras.backend.permute_dimensions(y_pred, (0, 2, 3, 1))
        
        loss = keras.backend.categorical_crossentropy(y_true, y_pred, from_logits=True) * w
        return loss


class ValidMetrics(keras.callbacks.Callback):
    """ Customized callback function for validation data evaluation

    Calculate ROC-AUC and F1 on validation data and test data (if applicable)
    after each epoch
    
    """

    def __init__(self, valid_data=None, test_data=None):
        self.valid_data = valid_data
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        if self.valid_data is not None:
            y_pred = self.model.predict(self.valid_data[0])[:, 0]
            y_true = self.valid_data[1][:, 0] > 0.5
            roc = roc_auc_score(y_true.flatten(), y_pred.flatten())
            f1 = f1_score(y_true.flatten(), y_pred.flatten()>0.5)
            print('\r valid-roc-auc: %f  valid-f1: %f\n' % (roc, f1))
        if self.test_data is not None:
            y_pred = self.model.predict(self.test_data[0])[:, 0]
            y_true = self.test_data[1][:, 0] > 0.5
            roc = roc_auc_score(y_true.flatten(), y_pred.flatten())
            f1 = f1_score(y_true.flatten(), y_pred.flatten()>0.5)
            print('\r test-roc-auc: %f  test-f1: %f\n' % (roc, f1))
        return
   

