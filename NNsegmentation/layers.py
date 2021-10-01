#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 17:51:20 2019

@author: zqwu
"""

# import tensorflow as tf
# import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
import segmentation_models
# from keras import backend as K
# from keras.models import Model, load_model
# from keras.layers import Dense, Layer, Input
from sklearn.metrics import roc_auc_score, f1_score, log_loss
# import tensorflow.nn.sparse_softmax_cross_entropy_with_logits as logits
# import tensorflow.nn.softmax_cross_entropy_with_logits as logits


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

            # l_loss = log_loss(y_true.flatten(), y_pred.flatten() > 0.1)
            # print(f'\r valid-logloss {l_loss}')

            # y_pred = self.model.predict(self.valid_data[0])
            # y_true = self.valid_data[1]
            y_pred_c0 = y_pred[:, 0]
            y_true_c0 = y_true[:, 0]
            roc = roc_auc_score(y_true_c0.flatten(), y_pred_c0.flatten())
            f1 = f1_score(y_true_c0.flatten() > 0.5, y_pred_c0.flatten() > 0.5)
            print('\r valid-roc-auc-c0: %f  valid-f1-c0: %f\n' % (roc, f1))

        if self.test_data is not None:
            y_pred = self.model.predict(self.test_data[0])[:, 0]
            y_true = self.test_data[1][:, 0] > 0.5
            roc = roc_auc_score(y_true.flatten(), y_pred.flatten())
            f1 = f1_score(y_true.flatten(), y_pred.flatten() > 0.5)
            print('\r test-roc-auc: %f  test-f1: %f\n' % (roc, f1))
        return


class metricsHistory(keras.callbacks.Callback):
    '''
    contains callbacks for training parameters
    '''

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('epoch'))
        self.losses.append(logs.get('loss'))
        self.losses.append(logs.get('dice'))
        self.losses.append(logs.get('val_loss'))
        self.losses.append(logs.get('val_dice'))


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

        # w = y_true[:, -1:]
        # y_true = y_true[:, :-1]

        # Switch to channel last form
        y_true = keras.backend.permute_dimensions(y_true, (0, 2, 3, 1))
        y_pred = keras.backend.permute_dimensions(y_pred, (0, 2, 3, 1))

        if self.n_classes > 2:
            print('using categorical xentropy for this >2 class model')
            # loss = keras.backend.categorical_crossentropy(y_true, y_pred,
            #                                               from_logits=True,
            #                                               axis=-1)
            loss = segmentation_models.losses.categorical_focal_loss(y_true, y_pred)
        else:
            print('using binary xentropy for this 2 class model')
            loss = keras.backend.binary_crossentropy(y_true, y_pred, from_logits=True)
        return loss


class FBeta:
    def __init__(self, beta=1, num_cls=2):
        self.__name__ = "fbeta"
        self.beta = beta
        self.classes = num_cls

    def __call__(self, y_true, y_pred):
        # y_true = y_true[:, :-1]
        # Switch to channel last form
        # y_true = keras.backend.permute_dimensions(y_true, (0, 2, 3, 1))
        # y_pred = keras.backend.permute_dimensions(y_pred, (0, 2, 3, 1))
        if self.classes == 2:
            return _F_beta(y_true, y_pred, beta=self.beta)
        else:
            fb_cl1 = _F_beta(y_true[:, 0:1], y_pred[:, 1:3])
            fb_cl2 = _F_beta(y_true[:, 1:2], y_pred[:, 0:3:2])
            fb_cl3 = _F_beta(y_true[:, 2:3], y_pred[:, 0:2])
            return (fb_cl1 + fb_cl2 + fb_cl3) / 3


class Precision:
    def __init__(self, num_cls=2):
        self.__name__ = "Precision"
        self.classes = num_cls

    def __call__(self, y_true, y_pred):
        # last channel is an injected "weights"
        # y_true = y_true[:, :-1]
        # Switch to channel last form
        # y_true = keras.backend.permute_dimensions(y_true, (0, 2, 3, 1))
        # y_pred = keras.backend.permute_dimensions(y_pred, (0, 2, 3, 1))
        if self.classes == 2:
            return precision(y_true, y_pred)
        else:
            # compute precision for each class and average
            # assume (batch, channel, y, x) order
            precision_cl1 = precision(y_true[:, 0:1], y_pred[:, 1:3])
            precision_cl2 = precision(y_true[:, 1:2], y_pred[:, 0:3:2])
            precision_cl3 = precision(y_true[:, 2:3], y_pred[:, 0:2])
            return (precision_cl1+precision_cl2+precision_cl3)/3


class Recall:
    def __init__(self, num_cls=2):
        self.__name__ = "Recall"
        self.classes = num_cls

    def __call__(self, y_true, y_pred):
        # y_true = y_true[:, :-1]
        # Switch to channel last form
        # y_true = keras.backend.permute_dimensions(y_true, (0, 2, 3, 1))
        # y_pred = keras.backend.permute_dimensions(y_pred, (0, 2, 3, 1))
        if self.classes == 2:
            return recall(y_true, y_pred)
        else:
            recall_cl1 = recall(y_true[:, 0:1], y_pred[:, 1:3])
            recall_cl2 = recall(y_true[:, 1:2], y_pred[:, 0:3:2])
            recall_cl3 = recall(y_true[:, 2:3], y_pred[:, 0:2])
            return (recall_cl1 + recall_cl2 + recall_cl3) / 3


def precision(y_true, y_pred, axis=None, smooth=1):
    batch_precision_coefs = _precision(y_true, y_pred, axis=axis, smooth=smooth)
    # precision_coefs = K.mean(batch_precision_coefs, axis=0)
    # return precision_coefs[1]
    return batch_precision_coefs


def recall(y_true, y_pred, axis=None, smooth=1):
    batch_recall_coefs = _recall(y_true, y_pred, axis=axis, smooth=smooth)
    # recall_coefs = K.mean(batch_recall_coefs, axis=0)
    # return recall_coefs[1]
    return batch_recall_coefs


# def F1(y_true, y_pred, axis=[2,3], smooth=1):
#     batch_F1_coefs = _F1(y_true, y_pred, axis=axis, smooth=smooth)
#     # F1_coefs = K.mean(batch_F1_coefs, axis=0)
#     # return F1_coefs[1]
#     return batch_F1_coefs


def _precision(y_true, y_pred, axis=None, smooth=1):
    # y_true_int = K.round(y_true)
    # y_pred_int = K.round(y_pred)
    # true_positive = K.sum(K.clip(y_true_int * y_pred_int, 0, 1), axis=axis)
    # false_positive = K.sum(y_pred_int, axis=axis) - true_positive

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=axis)
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=axis)
    return true_positives / (predicted_positives + K.epsilon())
    # return (true_positive+smooth)/(true_positive+false_positive+smooth)


def _recall(y_true, y_pred, axis=None, smooth=1):
    # y_true_int = K.round(y_true)
    # y_pred_int = K.round(y_pred)
    # true_positive = K.sum(K.clip(y_true_int * y_pred_int, 0, 1), axis=axis)
    # false_negative = K.sum(K.clip(y_true_int, 0, 1), axis=axis) - true_positive

    true_positive = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=axis)
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=axis)
    # return (true_positive)/(true_positive+false_negative+smooth)
    return true_positive / (possible_positives + K.epsilon())


# def _F1(y_true, y_pred, axis=None, smooth=1):
#     # y_true_int = K.round(y_true)
#     # y_pred_int = K.round(y_pred)
#     # true_positive = K.sum(y_true_int * y_pred_int, axis=axis)
#     # false_positive = K.sum(y_pred_int, axis=axis) - true_positive
#     # false_negative = K.sum(y_true_int, axis=axis) - true_positive
#     # return ( 2 * true_positive+smooth) / (2*true_positive + false_negative + false_positive +smooth)
#     precision = _precision(y_true, y_pred)
#     recall = _recall(y_true, y_pred)
#     return 2 * ((precision*recall) / (precision+recall + K.epsilon()))


def _F_beta(y_true, y_pred, axis=None, smooth=1, beta=1):
    """
    F beta is the generalized form of the F-score.
    beta = 1 evenly weighs precision and recall
    beta = 2 emphasizes recall
    beta = 0.5 emphasizes precision
    :param y_true:
    :param y_pred:
    :param axis:
    :param smooth:
    :param beta:
    :return:
    """
    # y_true_int = K.round(y_true)
    # y_pred_int = K.round(y_pred)
    # true_positive = K.sum(y_true_int * y_pred_int, axis=axis)
    # false_positive = K.sum(y_pred_int, axis=axis) - true_positive
    # false_negative = K.sum(y_true_int, axis=axis) - true_positive
    # prec = (true_positive+smooth)/(true_positive+false_positive+smooth)
    # recall = (true_positive+smooth)/(true_positive+false_negative+smooth)
    # return ( (1+beta**2) * prec * recall + smooth) / ( (beta**2)*prec + recall + smooth)
    precision = _precision(y_true, y_pred, axis=axis)
    recall = _recall(y_true, y_pred, axis=axis)
    return 2 * ((precision*recall) / (precision+recall + K.epsilon()))
