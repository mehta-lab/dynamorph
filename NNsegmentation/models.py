#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 13:22:55 2019

@author: zqwu
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
keras.backend.set_image_data_format('channels_first')
# keras.backend.set_image_data_format('channels_last')

import tempfile
import os
import scipy
from scipy.special import logsumexp
from copy import deepcopy
# from keras import backend as K
# from keras.models import Model, load_model
# from keras.layers import Dense, Layer, Input, BatchNormalization, Conv2D, Lambda
import segmentation_models
from .layers import weighted_binary_cross_entropy, ValidMetrics, \
    SplitSlice, MergeSlices, precision, recall, FBeta, Precision, Recall
from .data import load_input, preprocess


def _softmax(arr, axis=-1):
    """ Helper function for performing softmax operation """
    softmax_arr = np.exp(arr - logsumexp(arr, axis=axis, keepdims=True))
    return softmax_arr


class Segment(object):
    """ Semantic segmentation model based on U-Net """

    def __init__(self,
                 input_shape=(2, 256, 256),
                 n_classes=3,
                 freeze_encoder=False,
                 model_path=None,
                 **kwargs):
        """ Define model

        Args:
            input_shape (tuple of int, optional): shape of input features 
                (without batch dimension), should be in the order of 
                (c, x, y) or (c, z, x, y)
            n_classes (int, optional): number of prediction classes
            freeze_encoder (bool, optional): if to freeze backbone weights
            model_path (str or None, optional): path to save model weights
                if not given, a temp folder will be used

        """

        self.input_shape = input_shape
        self.n_channels = self.input_shape[0]
        self.x_size, self.y_size = self.input_shape[-2:]

        # if predicting and want fully convolutional, set image shape to 1024, 1024
        if self.x_size is None and self.y_size is None:
            self.input_shape = (2, 1024, 1024)
        if self.x_size is None:
            self.x_size = 1024
        if self.y_size is None:
            self.y_size = 1024
        
        self.n_classes = n_classes

        self.freeze_encoder = freeze_encoder
        if model_path is None:
            self.model_path = tempfile.mkdtemp()
        else:
            self.model_path = model_path

        # self.metrics = []
        self.build_model()

    def build_model(self):
        """ Define model structure and compile """

        self.input = keras.layers.Input(shape=self.input_shape, dtype='float32')
        self.pre_conv = keras.layers.Conv2D(3, (1, 1), activation=None, name='pre_conv')(self.input)

        self.unet = segmentation_models.Unet(
            backbone_name='resnet34',
            # backbone_name='vgg19',
            input_shape=(3, self.x_size, self.y_size),
            classes=self.n_classes,
            # activation='relu',
            # activation='sigmoid',
            # activation='linear',
            activation='softmax',
            encoder_weights='imagenet',
            encoder_features='default',
            decoder_block_type='upsampling',
            decoder_filters=(256, 128, 64, 32, 16),
            # decoder_filters=(256, 128, 64, 32, 32, 16, 16),
            decoder_use_batchnorm=True)

        # Segmentation models losses can be combined together by '+' and scaled by integer or float factor
        # set class weights for dice_loss (car: 1.; pedestrian: 2.; background: 0.5;)
        dice_loss = segmentation_models.losses.DiceLoss(class_weights=np.array([1., 1., 1.]))
        focal_loss = segmentation_models.losses.BinaryFocalLoss() if self.n_classes == 1 else \
            segmentation_models.losses.CategoricalFocalLoss(alpha=0.25, gamma=5.0)
        total_loss = dice_loss + (1 * focal_loss)

        self.loss_func = weighted_binary_cross_entropy(n_classes=self.n_classes)
        # self.loss_func = focal_loss

        # actually total_loss can be imported directly from library, above example just show you how to manipulate with losses
        # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss

        self.metrics = [FBeta(num_cls=self.n_classes),
                        Precision(num_cls=self.n_classes),
                        Recall(num_cls=self.n_classes),
                        segmentation_models.metrics.IOUScore(threshold=0.5),
                        segmentation_models.metrics.FScore(threshold=0.5)]

    def compile_model(self, lr=0.001, opt=None):

        if not opt:
            opt = keras.optimizers.Adam(
                learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
                name='Adam'
            )

        output = self.unet(self.pre_conv)

        self.model = keras.models.Model(self.input, output)
        self.model.compile(optimizer=opt,
                           loss=self.loss_func,
                           metrics=self.metrics,
                           # Horovod: Specify `experimental_run_tf_function=False` to ensure TensorFlow
                           # uses hvd.DistributedOptimizer() to compute gradients.
                           experimental_run_tf_function=False
                           )

    def fit(self, 
            patches,
            label_input='prob',
            batch_size=8, 
            n_epochs=10,
            valid_patches=None,
            valid_label_input='prob',
            class_weights=None,
            cbs = None,
            **kwargs):
        """ Fit model
        
        Args:
            patches (list): list of input-label pairs
                see docs of `generate_patches`
            label_input (str or None, optional): 'prob' or 'annotation' or None
                label input type, probabilities or discrete annotation
            batch_size (int, optional): default=8, batch size
            n_epochs (int, optional): default=10, number of epochs
            valid_patches (list or None, optional): if given, input-label pairs
                of validation data
            valid_label_input (str, optional): 'prob' or 'annotation'
                label input type of `valid_patches` (if applicable)
            class_weights (None of list, optional): if given, specify training 
                weights for different classes
            **kwargs: Other keyword arguments for keras model `fit` function

        """

        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        # `X` and `y` should originally be 5 dimensional: (batch, c, z, x, y),
        # in default model z=1 will be neglected
        X, y = preprocess(patches,
                          n_classes=self.n_classes,
                          label_input=label_input,
                          class_weights=class_weights)
        X = X.reshape(self.batch_input_shape)
        y = y.reshape(self.batch_label_shape)
        assert X.shape[0] == y.shape[0]

        if not cbs:
            call_backs = [keras.callbacks.TerminateOnNaN(),
                          keras.callbacks.ReduceLROnPlateau(patience=5, min_lr=1e-6),
                          ]
        else:
            call_backs = cbs

        validation_data = None
        if valid_patches is not None:
            valid_X, valid_y = preprocess(valid_patches, 
                                          n_classes=self.n_classes, 
                                          label_input=valid_label_input)
            valid_X = valid_X.reshape(self.batch_input_shape)
            valid_y = valid_y.reshape(self.batch_label_shape)
            assert valid_X.shape[0] == valid_y.shape[0]
            valid_score_callback = ValidMetrics()
            valid_score_callback.valid_data = (valid_X, valid_y)
            validation_data = (valid_X, valid_y)
            call_backs.append(valid_score_callback)

        self.model.fit(x=X, 
                       y=y,
                       batch_size=batch_size,
                       epochs=n_epochs,
                       verbose=1,
                       callbacks=call_backs,
                       validation_data=validation_data,
                       **kwargs)

    def predict(self, patches, label_input='prob'):
        """ Generate prediction for given data

        Args:
            patches (list): list of input-label pairs (label could be None)
                see docs of `generate_patches`
            label_input (str or None, optional): 'prob' or 'annotation' or None
                label input type, probabilities or discrete annotation

        """

        if patches.__class__ is list:
            X, _ = preprocess(patches, label_input=label_input)
            X = X.reshape(self.batch_input_shape)
            y_pred = self.model.predict(X)
        elif patches.__class__ is np.ndarray:
            X = patches.reshape(self.batch_input_shape)
            y_pred = self.model.predict(X)
        else:
            raise ValueError("Input format not supported")
        y_pred = _softmax(y_pred, 1)
        assert y_pred.shape[1:] == (self.n_classes, self.x_size, self.y_size)
        y_pred = np.expand_dims(y_pred, 2) # Manually add z dimension
        return y_pred


    @property
    def batch_input_shape(self):
        return tuple([-1,] + list(self.input_shape))


    @property
    def batch_label_shape(self):
        return tuple([-1, self.n_classes+1, self.x_size, self.y_size])


    def save(self, path):
        """ Save model weights to `path` """
        self.model.save_weights(path)


    def load(self, path):
        """ Load model weights from `path` """
        self.model.load_weights(path)



# class SegmentWithMultipleSlice(Segment):
#     """ Semantic segmentation model with inputs having multiple time/z slices """
#
#     def __init__(self,
#                  unet_feat=32,
#                  **kwargs):
#         """ Define model
#
#         Args:
#             unet_feat (int, optional): output dimension of unet (used as
#                 hidden units)
#             **kwargs: keyword arguments for `Segment`
#                 note that `input_shape` should have 4 dimensions
#
#         """
#
#         self.unet_feat = unet_feat
#         super(SegmentWithMultipleSlice, self).__init__(**kwargs)
#         self.n_slices = self.input_shape[1] # Input shape (c, z, x, y)
#
#
#     def build_model(self):
#         """ Define model structure and compile """
#
#         # input shape: batch_size, n_channel, n_slice, x_size, y_size
#         self.input = keras.layers.Input(shape=self.input_shape, dtype='float32')
#
#         # Combine time slice dimension and batch dimension
#         inp = SplitSlice(self.n_channels, self.x_size, self.y_size)(self.input)
#         self.pre_conv = keras.layers.Conv2D(3, (1, 1), activation=None, name='pre_conv')(inp)
#
#         self.unet = segmentation_models.Unet(
#             backbone_name='resnet34',
#             input_shape=(3, self.x_size, self.y_size),
#             classes=self.unet_feat,
#             activation='linear',
#             encoder_weights='imagenet',
#             encoder_features='default',
#             decoder_block_type='upsampling',
#             decoder_filters=(256, 128, 64, 32, 16),
#             decoder_use_batchnorm=True)
#
#         output = self.unet(self.pre_conv)
#
#         # Split time slice dimension and merge to channel dimension
#         output = MergeSlices(self.n_slices, self.unet_feat)(output)
#         output = keras.layers.Conv2D(self.unet_feat, (1, 1), activation='relu', name='post_conv')(output)
#         output = keras.layers.Conv2D(self.n_classes, (1, 1), activation=None, name='pred_head')(output)
#
#         self.model = keras.models.Model(self.input, output)
#         self.model.compile(optimizer='Adam',
#                            loss=self.loss_func,
#                            metrics=[])
