#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 13:22:55 2019

@author: zqwu
"""

import segmentation_models
import tensorflow as tf
import numpy as np
import keras
import tempfile
import os
import scipy
from scipy.special import logsumexp
from copy import deepcopy
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Dense, Layer, Input, BatchNormalization, Conv2D, Lambda
from .layers import weighted_binary_cross_entropy, ValidMetrics, Reshape, MergeOnZ
from .data import load_input, preprocess


def _softmax(arr, axis=-1):
    """ Helper function for performing softmax operation """
    softmax_arr = np.exp(arr - logsumexp(arr, axis=axis, keepdims=True))
    return softmax_arr


class Segment(object):
    """ Semantic segmentation model based on U-Net """

    def __init__(self,
                 input_shape=(256, 256, 20),
                 n_classes=2,
                 freeze_encoder=False,
                 model_path=None,
                 **kwargs):
        """ Define model

        Args:
            input_shape (tuple of int, optional): shape of input features 
                (without batch dimension and time slice dimension if 
                applicable), should be in the order of X, Y, C
            n_classes (int, optional): number of prediction classes
            freeze_encoder (bool, optional): if to freeze backbone weights
            model_path (str or None, optional): path to save model weights
                if not given, a temp folder will be used

        """

        self.input_shape = input_shape
        self.n_classes = n_classes

        self.freeze_encoder = freeze_encoder
        if model_path is None:
            self.model_path = tempfile.mkdtemp()
        else:
            self.model_path = model_path
        self.call_backs = [keras.callbacks.TerminateOnNaN(),
                           keras.callbacks.ReduceLROnPlateau(patience=5, min_lr=1e-7),
                           keras.callbacks.ModelCheckpoint(self.model_path + '/weights.{epoch:02d}-{val_loss:.2f}.hdf5')]
        self.valid_score_callback = ValidMetrics()
        self.loss_func = weighted_binary_cross_entropy(n_classes=self.n_classes)
        self.build_model()
    

    def build_model(self):
        """ Define model structure and compile """

        self.input = Input(shape=self.input_shape, dtype='float32')
        self.pre_conv = Dense(3, activation=None, name='pre_conv')(self.input)

        # Define U-Net backbone with self-defined inputs
        # Requires segmentation_models==0.2.1
        backbone = segmentation_models.backbones.get_backbone(
            'resnet34',
            input_shape=list(self.input_shape[:2]) + [3],
            weights='imagenet',
            include_top=False)
        
        if self.freeze_encoder:
            for layer in backbone.layers:
                if not isinstance(layer, BatchNormalization):
                    layer.trainable=False

        skip_connection_layers = segmentation_models.backbones.get_feature_layers('resnet34', n=4)
        self.unet = segmentation_models.unet.builder.build_unet(
            backbone,
            self.n_classes,
            skip_connection_layers,
            decoder_filters=(256, 128, 64, 32, 16),
            block_type='upsampling',
            activation='linear',
            n_upsample_blocks=5,
            upsample_rates=(2, 2, 2, 2, 2),
            use_batchnorm=True)
        output = self.unet(self.pre_conv)
        
        self.model = Model(self.input, output)
        self.model.compile(optimizer='Adam', 
                           loss=self.loss_func,
                           metrics=[])


    def fit(self, 
            patches,
            label_input='prob',
            batch_size=8, 
            n_epochs=10,
            valid_patches=None,
            valid_label_input='prob',
            class_weights=None,
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
        X, y = preprocess(patches, 
                          n_classes=self.n_classes, 
                          label_input=label_input, 
                          class_weights=class_weights)
        validation_data = None
        if valid_patches is not None:
            validation_data = preprocess(valid_patches, n_classes=self.n_classes, label_input=valid_label_input)
            self.valid_score_callback.valid_data = validation_data
          
        self.model.fit(x=X, 
                       y=y,
                       batch_size=batch_size,
                       epochs=n_epochs,
                       verbose=1,
                       callbacks=self.call_backs + [self.valid_score_callback],
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
            y_pred = self.model.predict(X)
        elif patches.__class__ is np.ndarray:
            y_pred = self.model.predict(patches)
        else:
            raise ValueError("Input format not supported")
        y_pred = _softmax(y_pred, -1)
        return y_pred


    def save(self, path):
        """ Save model weights to `path` """
        self.model.save_weights(path)
    

    def load(self, path):
        """ Load model weights from `path` """
        self.model.load_weights(path)


class Segment_with_time(Segment):
    """ Semantic segmentation model with inputs having multiple time slices """

    def __init__(self,
                 n_time_slice=5,
                 unet_feat=32,
                 **kwargs):
        """ Define model

        Args:
            n_time_slice (int, optional): number of time slices in inputs
            unet_feat (int, optional): output dimension of unet (used as 
                hidden units)
            **kwargs: keyword arguments for `Segment`

        """
        self.n_time_slice = n_time_slice
        self.unet_feat = unet_feat
        super(Segment_with_time, self).__init__(**kwargs)


    def build_model(self):
        """ Define model structure and compile """

        self.input = Input(shape=tuple([self.n_time_slice] + list(self.input_shape)), dtype='float32')

        # Combine time slice dimension and batch dimension
        inp = Reshape([-1] + list(self.input_shape))(self.input)
        pre_conv = Dense(3, activation=None, name='pre_conv')(inp)

        backbone = segmentation_models.backbones.get_backbone(
          'resnet34',
          input_shape=list(self.input_shape[:2]) + [3],
          weights='imagenet',
          include_top=False)

        if self.freeze_encoder:
          for layer in backbone.layers:
              if not isinstance(layer, BatchNormalization):
                  layer.trainable=False

        skip_connection_layers = segmentation_models.backbones.get_feature_layers('resnet34', n=4)
        self.unet = segmentation_models.unet.builder.build_unet(
          backbone,
          self.unet_feat,
          skip_connection_layers,
          decoder_filters=(256, 128, 64, 32, 16),
          block_type='upsampling',
          activation='linear',
          n_upsample_blocks=5,
          upsample_rates=(2, 2, 2, 2, 2),
          use_batchnorm=True)

        output = self.unet(pre_conv)

        # Split time slice dimension and merge to channel dimension
        output = MergeOnZ(self.n_time_slice, self.unet_feat)(output)
        output = Dense(self.unet_feat, activation='relu')(output)
        output = Dense(self.n_classes, activation=None)(output)
        self.model = Model(self.input, output)
        self.model.compile(optimizer='Adam',
                         loss=self.loss_func,
                         metrics=[])