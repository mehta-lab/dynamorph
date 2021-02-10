#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 17:38:59 2019

@author: zqwu
"""

import h5py
import numpy as np
import os
import cv2

CHANNEL_MAX = 65535.


def load_input(file_name):
    if os.path.splitext(file_name)[1] == '.h5':
        dat = h5py.File(file_name, 'r+')
        dat = np.stack([dat[key] for key in sorted(dat.keys())], 0)
    elif os.path.splitext(file_name)[1] == '.npy':
        dat = np.load(file_name)
    assert len(dat.shape) == 5, "Please format inputs as 5-dimensional (t, c, z, x, y) arrays"
    return dat


def load_label(file_name):
    if os.path.splitext(file_name)[1] == '.h5':
        dat = h5py.File(file_name, 'r+')
        key = list(dat.keys())[0]
        dat = dat[key]
    elif os.path.splitext(file_name)[1] == '.npy':
        dat = np.load(file_name)
    return dat


def cv2_fn_wrapper(cv2_fn, mat, *args, **kwargs):
    """" A wrapper for cv2 functions
    
    Data in channel first format are adjusted to channel last format for 
    cv2 functions
    """
    
    mat_shape = mat.shape
    x_size = mat_shape[-2]
    y_size = mat_shape[-1]
    _mat = mat.reshape((-1, x_size, y_size)).transpose((1, 2, 0))
    _output = cv2_fn(_mat, *args, **kwargs)
    _x_size = _output.shape[0]
    _y_size = _output.shape[1]
    output_shape = tuple(list(mat_shape[:-2]) + [_x_size, _y_size])
    output = _output.transpose((2, 0, 1)).reshape(output_shape)
    return output


def rotate_image(mat, angle, image_center=None):
    """ Rotate image `mat` by `angle`

    Args:
        mat (np.array): target image, size C * Z * X * Y
        angle (float): rotation angle
        image_center (tuple or None, optional): center of rotation
            if not specified, center of `mat` is used

    Returns:
        np.array: rotated image

    """

    n_channel, n_z, height, width = mat.shape
    if image_center is None:
        image_center = (width/2, height/2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    rotated_mat = cv2_fn_wrapper(cv2.warpAffine, mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat


def generate_patches(input_file, 
                     label_file, 
                     use_channels=[],
                     label_input='prob',
                     n_patches=1000,
                     x_size=256,
                     y_size=256,
                     rotate=False,
                     mirror=False,
                     seed=None,
                     **kwargs):
    """ Curate dataset for segmentation

    Given `input_file` and `label_file`, `n_patches` patches will be cropped
    randomly from the full image, optional image augmentation can be added.

    Args:
        input_file (str): input file path
        label_file (str): label file path
        use_channels (list, optional): list of channel indices used for model
            prediction, by default all channels will be used
        label_input (str, optional): 'prob' or 'annotation'
            label input type, probabilities or discrete annotation
        n_patches (int, optional): number of generated patches required
        x_size (int, optional): size of patch
        y_size (int, optional): size of patch
        rotate (bool, optional): if randomly rotate patch to augment data
        mirror (bool, optional): if randomly mirror patch to augment data
        seed (int, optional): random seed for data augmentation

    Returns:
        list: list of input-label pairs for segmentation model training
    """

    input_f = load_input(input_file) #TCZXY
    label_f = load_label(label_file) #TCZXY
    if len(use_channels) == 0:
        use_channels = list(range(input_f.shape[1]))
    input_f = input_f[:, np.array(use_channels)]
    
    n_frame, n_channel, n_z, x_full_size, y_full_size = input_f.shape
    
    x_margin = int(x_size/np.sqrt(2))
    y_margin = int(y_size/np.sqrt(2))
    
    data = []
    if not seed is None:
        np.random.seed(seed)
    while len(data) < n_patches:
        # Randomly pick time slice
        t_point = np.random.randint(n_frame)
        
        # Randomly pick image center
        x_center = np.random.randint(x_size/np.sqrt(2), x_full_size-x_size/np.sqrt(2))
        y_center = np.random.randint(y_size/np.sqrt(2), y_full_size-y_size/np.sqrt(2))

        if rotate:
            angle = np.random.rand() * 360
            patch_input_slice = input_f[t_point, ...,
                                        (x_center - x_margin):(x_center + x_margin),
                                        (y_center - y_margin):(y_center + y_margin)]
            patch_label_slice = label_f[t_point, ...,
                                        (x_center - x_margin):(x_center + x_margin),
                                        (y_center - y_margin):(y_center + y_margin)]
            
            # Rotate image by `angle`
            patch_input_slice = rotate_image(np.array(patch_input_slice).astype(float), angle)
            patch_label_slice = rotate_image(np.array(patch_label_slice).astype(float), angle)

            # Crop patch from rotated image
            _patch_x_size, _patch_y_size = patch_input_slice.shape[-2:]
            center = (_patch_x_size//2, _patch_y_size//2)
            patch_X = patch_input_slice[...,
                                        (center[0] - x_size//2):(center[0] + x_size//2),
                                        (center[1] - y_size//2):(center[1] + y_size//2)]
            patch_y = patch_label_slice[...,
                                        (center[0] - x_size//2):(center[0] + x_size//2),
                                        (center[1] - y_size//2):(center[1] + y_size//2)]
        else:
            x_margin = x_size//2
            y_margin = y_size//2
            patch_X = np.array(input_f[t_point, ...,
                                       (x_center - x_margin):(x_center + x_margin),
                                       (y_center - y_margin):(y_center + y_margin)]).astype(float)
            patch_y = np.array(label_f[t_point, ...,
                                       (x_center - x_margin):(x_center + x_margin),
                                       (y_center - y_margin):(y_center + y_margin)]).astype(float)
        if mirror:
            if np.random.rand() > 0.5:
                patch_X = cv2_fn_wrapper(cv2.flip, patch_X, 1)
                patch_y = cv2_fn_wrapper(cv2.flip, patch_y, 1)
        
        if label_input == 'prob':
            data.append([patch_X, patch_y])
        elif label_input == 'annotation':
            # Exclude this pair if there is no annotation in the patch
            if len(np.unique(patch_y)) == 1:
                continue
            data.append([patch_X, patch_y.astype(int)])
    return data


def generate_ordered_patches(input_file, 
                             label_file, 
                             use_channels=[],
                             label_input='prob',
                             x_size=256,
                             y_size=256,
                             time_slices=1,
                             **kwargs):
    """ Curate dataset for segmentation

    Given `input_file` and `label_file`, patches will be cropped as tiles of
    the full mat, multiple slices in the time axis could be combined if needed.

    Args:
        input_file (str): input file path
        label_file (str): label file path
        use_channels (list, optional): list of channel indices used for model
            prediction, by default all channels will be used
        label_input (str, optional): 'prob' or 'annotation'
            label input type, probabilities or discrete annotation
        x_size (int, optional): size of patch
        y_size (int, optional): size of patch
        time_slices (int, optional): deprecated
            if larger than 1, same position along multiple time slices will be 
            cropped and stacked

    Returns:
        list: list of input-label pairs for segmentation model training
    """

    input_f = load_input(input_file) #TCZXY
    label_f = load_label(label_file) #TCZXY
    if len(use_channels) == 0:
        use_channels = list(range(input_f.shape[1]))
    input_f = input_f[:, np.array(use_channels)]
    
    n_frame, n_channel, n_z, x_full_size, y_full_size = input_f.shape

    n_slice_x = x_full_size // x_size
    n_slice_y = y_full_size // y_size  
    data = []
    for t_point in range(n_frame - (time_slices - 1)):
        if len(np.unique(label_f[t_point])) == 1:
            continue
        print(t_point)
        for i in range(n_slice_x):
            for j in range(n_slice_y):
                if time_slices == 1:
                    patch_X = np.array(input_f[t_point, ...,
                                               (i*x_size):((i+1)*x_size), 
                                               (j*y_size):((j+1)*y_size)]).astype(float)
                else:
                    patch_X = np.array(input_f[t_point:(t_point+time_slices), ...,
                                               (i*x_size):((i+1)*x_size), 
                                               (j*y_size):((j+1)*y_size)]).astype(float)
                
                patch_y = np.array(label_f[t_point, ...,
                                           (i*x_size):((i+1)*x_size), 
                                           (j*y_size):((j+1)*y_size)])
                if label_input == 'prob':
                    patch_y = patch_y.astype(float)
                elif label_input == 'annotation':
                    patch_y = patch_y.astype(int)
                    if len(np.unique(patch_y)) == 1:
                        continue
                data.append([patch_X, patch_y])
    return data


def preprocess(patches,
               n_classes=3, 
               label_input='prob', 
               class_weights=None):
    """ Preprocess patches for model training/prediction
    
    Args:
        patches (list): input-label pairs generated by `generate_patches`, etc.
        n_classes (int, optional): default=3 (BG, Non-MG, MG)
            number of prediction classes
        label_input (str or None, optional): 'prob' or 'annotation' or None
            label input type, probabilities or discrete annotation
        class_weights (None of list, optional): if given, specify training 
            weights for different classes

    Returns:
        np.array: input features
        np.array or None: labels, `None` if data is only used for prediction

    """
    Xs = []
    ys = []
    ws = []
    if class_weights is None:
        class_weights = np.ones((n_classes,))
    
    n_channel, n_z, x_size, y_size = patches[0][0].shape
    for pair in patches:
        assert pair[0].shape == (n_channel, n_z, x_size, y_size)
        Xs.append(pair[0])
        if label_input:
            assert pair[1].shape[2:] == (x_size, y_size)
            assert pair[1].shape[1] == 1, "Only support 2D segmentation, z dimension should be 1"
        if label_input == 'prob':
            assert pair[1].shape[0] == n_classes # channel dimension equals num classes
            ys.append(pair[1])
            # TODO: add class weights
            ws.append(np.ones((1, 1, x_size, y_size)))
        elif label_input == 'annotation':
            # Transform numeric labels (1, 2, etc.) to one-hot
            # 0 are regarded as no labels available at the pixel
            y = np.zeros((n_classes, 1, x_size, y_size))
            w = np.zeros((1, 1, x_size, y_size))
            for c in range(n_classes):
                x_pos, y_pos = np.where(pair[1] == (c+1))[-2:]
                positions = tuple([np.ones_like(x_pos) * c,
                                   np.zeros_like(x_pos),
                                   x_pos,
                                   y_pos])
                y[positions] = 1
                w[..., x_pos, y_pos] = class_weights[c]
            ys.append(y)
            ws.append(w)
        elif label_input is None:
            pass
        else:
            raise ValueError("Label type not recognized")

    Xs = np.stack(Xs, 0).astype(float) # Batch, c, z, x, y
    Xs = Xs / CHANNEL_MAX # Scale to [0, 1]
    if label_input is not None:
        ys = np.stack(ys, 0) # Batch, n_classes, 1, x, y
        ws = np.stack(ws, 0) # Batch, 1, 1, x, y
        return Xs, np.concatenate([ys, ws], 1)
    else:
        return Xs, None


def plot_prediction_prob(d1, path):
    """ Helper function for visualizing prediction results

    Plot semantic segmentation results as image and save to `path`

    Args:
        d1 (np.array): semantic segmentation result
        path (str): path for saving image
    """
    assert d1.shape[0] == 3
    x_size, y_size = d1.shape[-2:]
    
    mat = np.zeros((x_size, y_size, 4))
    mat[:, :, :3] += d1[1, 0].reshape((x_size, y_size, 1)) * np.array([200, 130, 0]).reshape((1, 1, 3))
    mat[:, :, -1] += d1[1, 0] * 255
    mat[:, :, :3] += d1[2, 0].reshape((x_size, y_size, 1)) * np.array([75, 25, 230]).reshape((1, 1, 3))
    mat[:, :, -1] += d1[2, 0] * 255
    
    cv2.imwrite(path, mat)
    return


def predict_whole_map(file_path, 
                      model, 
                      use_channels=[],
                      out_file_path=None, 
                      batch_size=8, 
                      n_supp=5, 
                      time_slices=1,
                      **kwargs):
    """ Wrapper method for semantic segmentation on given input

    Inputs will be cropped into tiling patches and processed by `model` to 
    generate semantic segmentation results. Multiple runs (with simple ensemble 
    averaging) can be employed to alleviate edge effects.

    Args:
        file_path (str or np.array): path to target raw data (.npy)
        model (keras model): pre-saved model for semantic segmentation
        use_channels (list, optional): list of channel indices used for model
            prediction, by default all channels will be used
        out_file_path (str or None, optional): path of output
            if not specified, results will be saved under the same folder as 
            `file_path` with suffix "_NNProbabilities.npy"
        batch_size (int, optional): default=8, batch size
        n_supp (int, optional): default=5, number of extra prediction rounds
            each round of supplementary prediction will be initiated with 
            different offset
        time_slices (int, optional): deprecated
            set to the same value as patch generator

    Returns:
        None or np.array: if `out_file_path` is not specified, return 
            segmentation results (also saved using default path)
    
    """
    if file_path.__class__ is str:
        inputs = load_input(file_path)
    else:
        inputs = file_path
        
    if len(use_channels) == 0:
        use_channels = list(range(inputs.shape[1]))
    inputs = inputs[:, np.array(use_channels)]
    
    x_size = model.x_size
    y_size = model.y_size
    n_classes = model.n_classes
    
    n_frame, n_channel, n_z, x_full_size, y_full_size = inputs.shape
    assert x_full_size % x_size == 0
    assert y_full_size % y_size == 0
    assert n_channel == model.n_channels
    rows = x_full_size // x_size
    columns = y_full_size // y_size

    total_outputs = []
    for t in range(n_frame - (time_slices - 1)):
        print("Predicting %d" % t)
        inp = inputs[t:(t+time_slices)]

        batch_inputs = []
        outputs = []
        # Predict patch by patch in a tiling pattern
        for r in range(rows):
            for c in range(columns):
                patch_inp = inp[..., r*x_size:(r+1)*x_size, c*y_size:(c+1)*y_size]
                if time_slices == 1:
                    patch_inp = patch_inp[0]
                batch_inputs.append((patch_inp, None))
                if len(batch_inputs) == batch_size:
                    batch_outputs = model.predict(batch_inputs, label_input=None)
                    outputs.extend([_output for _output in batch_outputs])
                    batch_inputs = []
        if len(batch_inputs) > 0:
            batch_outputs = model.predict(batch_inputs, label_input=None)
            outputs.extend([_output for _output in batch_outputs])
            batch_inputs = []
      
        # Connect and assemble segmentation results
        ct = 0
        concatenated_output = -np.ones((n_classes, 1, x_full_size, y_full_size))
        for r in range(rows):
            for c in range(columns):
                concatenated_output[...,
                                    r*x_size:(r+1)*x_size, 
                                    c*y_size:(c+1)*y_size] = outputs[ct]
                ct += 1
        
        # Supplementary runs
        for i_supp in range(n_supp):
            # Initiate with random offsets
            x_offset = np.random.randint(1, x_size)
            y_offset = np.random.randint(1, y_size)
            batch_inputs = []
            outputs = []
            for r in range(rows - 1):
                for c in range(columns - 1):
                    patch_inp = inp[..., 
                                    (x_offset + r*x_size):(x_offset + (r+1)*x_size), 
                                    (y_offset + c*y_size):(y_offset + (c+1)*y_size)]
                    if time_slices == 1:
                        patch_inp = patch_inp[0]
                    batch_inputs.append((patch_inp, None))
                    if len(batch_inputs) == batch_size:
                        batch_outputs = model.predict(batch_inputs, label_input=None)
                        outputs.extend([_output for _output in batch_outputs])
                        batch_inputs = []
            if len(batch_inputs) > 0:
                batch_outputs = model.predict(batch_inputs, label_input=None)
                outputs.extend([_output for _output in batch_outputs])
                batch_inputs = []

            supp_output = np.copy(concatenated_output)
            ct = 0
            for r in range(rows - 1):
                for c in range(columns - 1):
                    supp_output[...,
                                (x_offset + r*x_size):(x_offset + (r+1)*x_size), 
                                (y_offset + c*y_size):(y_offset + (c+1)*y_size)] = outputs[ct]
                    ct += 1
            concatenated_output = (concatenated_output * (i_supp + 1) + supp_output)/(i_supp + 2)
        total_outputs.append(concatenated_output)
    total_outputs = np.stack(total_outputs, 0)
    
    if file_path.__class__ is str:
        if out_file_path is None:
            out_file_path = os.path.splitext(file_path)[0] + '_NNProbabilities'
        np.save(out_file_path, total_outputs)
        
        # Save image for sanity check
        cv2.imwrite(os.path.splitext(file_path)[0] + '.png', inputs[0, 0, 0])
        plot_prediction_prob(total_outputs[0], os.path.splitext(file_path)[0] + '_NNpred.png')
    else:
        return total_outputs
