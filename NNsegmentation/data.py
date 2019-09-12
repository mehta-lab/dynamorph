#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 17:38:59 2019

@author: zqwu
"""

import h5py
import numpy as np
import os
import cv2

CHANNEL_MAX = np.array([65535., 65535.])

def load_input(file_name):
  if os.path.splitext(file_name)[1] == '.h5':
    dat = h5py.File(file_name, 'r+')
    dat = np.stack([dat[key] for key in sorted(dat.keys())], 0)
  elif os.path.splitext(file_name)[1] == '.npy':
    dat = np.load(file_name)
  return dat

def load_label(file_name):
  if os.path.splitext(file_name)[1] == '.h5':
    dat = h5py.File(file_name, 'r+')
    key = list(dat.keys())[0]
    dat = dat[key]
  elif os.path.splitext(file_name)[1] == '.npy':
    dat = np.load(file_name)
  return dat


def rotate_image(mat, angle, image_center=None):
  # angle in degrees

  height, width = mat.shape[:2]
  if image_center is None:
    image_center = (width/2, height/2)

  rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)


  abs_cos = abs(rotation_mat[0,0])
  abs_sin = abs(rotation_mat[0,1])

  bound_w = int(height * abs_sin + width * abs_cos)
  bound_h = int(height * abs_cos + width * abs_sin)

  rotation_mat[0, 2] += bound_w/2 - image_center[0]
  rotation_mat[1, 2] += bound_h/2 - image_center[1]

  rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
  return rotated_mat

def generate_patches(input_file, 
                     label_file, 
                     label_input='prob',
                     n_patches=1000,
                     x_size=256,
                     y_size=256,
                     label_value_threshold=0.5,
                     rotate=False,
                     mirror=False,
                     seed=None,
                     **kwargs):
  input_f = load_input(input_file) #TXYZC
  label_f = load_label(label_file) #TXYZC
  
  x_margin = int(x_size/np.sqrt(2))
  y_margin = int(y_size/np.sqrt(2))
  
  data = []
  if not seed is None:
    np.random.seed(seed)
  while len(data) < n_patches:
    # Randomly pick time slice
    t_point = np.random.randint(label_f.shape[0])
    
    # Randomly pick image center
    x_center = np.random.randint(x_size/np.sqrt(2), input_f[0].shape[0]-x_size/np.sqrt(2))
    y_center = np.random.randint(y_size/np.sqrt(2), input_f[0].shape[1]-y_size/np.sqrt(2))

    if rotate:
      angle = np.random.rand() * 360
      patch_input_slice = input_f[t_point,
                                  (x_center - x_margin):(x_center + x_margin),
                                  (y_center - y_margin):(y_center + y_margin)]
      patch_label_slice = label_f[t_point,
                                  (x_center - x_margin):(x_center + x_margin),
                                  (y_center - y_margin):(y_center + y_margin)]
      
      patch_input_slice = rotate_image(np.array(patch_input_slice).astype(float), angle)
      patch_label_slice = rotate_image(np.array(patch_label_slice).astype(float), angle)
      center = (patch_input_slice.shape[0]//2, patch_input_slice.shape[1]//2)
      patch_X = patch_input_slice[(center[0] - x_size//2):(center[0] + x_size//2),
                                  (center[1] - y_size//2):(center[1] + y_size//2)]
      patch_y = patch_label_slice[(center[0] - x_size//2):(center[0] + x_size//2),
                                  (center[1] - y_size//2):(center[1] + y_size//2)]
    else:
      x_margin = x_size//2
      y_margin = y_size//2
      patch_X = np.array(input_f[t_point,
                                 (x_center - x_margin):(x_center + x_margin),
                                 (y_center - y_margin):(y_center + y_margin)]).astype(float)
      patch_y = np.array(label_f[t_point,
                                 (x_center - x_margin):(x_center + x_margin),
                                 (y_center - y_margin):(y_center + y_margin)]).astype(float)
    if mirror:
      if np.random.rand() > 0.5:
        patch_X = cv2.flip(patch_X, 1)
        patch_y = cv2.flip(patch_y, 1)
    
    patch_X = patch_X / CHANNEL_MAX.reshape((1, 1, -1))
    if label_input == 'prob':
      data.append([patch_X, patch_y.reshape((x_size, y_size, -1))])
    elif label_input == 'annotation':
      if len(np.unique(patch_y)) == 1:
        continue
      data.append([patch_X, patch_y.astype(int).reshape((x_size, y_size, -1))])
  return data

def generate_ordered_patches(input_file, 
                             label_file, 
                             label_input='prob',
                             x_size=256,
                             y_size=256,
                             label_value_threshold=0.5,
                             time_slices=1,
                             **kwargs):
  input_f = load_input(input_file)
  label_f = load_label(label_file)

  n_slice_x = label_f.shape[1] // x_size
  n_slice_y = label_f.shape[2] // y_size  
  data = []
  for t_point in range(len(input_f) - (time_slices - 1)):
    if len(np.unique(label_f[t_point])) == 1:
      continue
    print(t_point)
    for i in range(n_slice_x):
      for j in range(n_slice_y):
        if time_slices == 1:
          patch_X = np.array(input_f[t_point, 
                                     (i*x_size):((i+1)*x_size), 
                                     (j*y_size):((j+1)*y_size)]).astype(float)
        else:
          patch_X = np.array(input_f[t_point:(t_point+time_slices), 
                                     (i*x_size):((i+1)*x_size), 
                                     (j*y_size):((j+1)*y_size)]).astype(float)
        patch_X = patch_X / CHANNEL_MAX.reshape((1, 1, -1))
        if label_input == 'prob':
          patch_y = np.array(label_f[t_point, 
                                     (i*x_size):((i+1)*x_size), 
                                     (j*y_size):((j+1)*y_size)]).astype(float)
          
        elif label_input == 'annotation':
          patch_y = np.array(label_f[t_point, 
                                     (i*x_size):((i+1)*x_size), 
                                     (j*y_size):((j+1)*y_size)]).astype(int)
          if len(np.unique(patch_y)) == 1:
            continue
        data.append([patch_X, patch_y])
  return data

def preprocess(patches, n_classes=3, label_input='prob', class_weights=None):
  """
  patches: list of (X, y)
    X: np.array, input_shape
    y: np.array, input_shape[:2]
  label_input: str, 'prob' or 'annotation'
  """
  Xs = []
  ys = []
  ws = []
  if class_weights is None:
    class_weights = np.ones((n_classes,))
  for pair in patches:
    Xs.append(pair[0])
    if label_input == 'prob':
      ys.append(pair[1])
      # TODO: add class weights
      ws.append(np.ones((pair[1].shape[0], pair[1].shape[1], 1)))
    elif label_input == 'annotation':
      y = np.zeros((pair[1].shape[0], pair[1].shape[1], n_classes))
      w = np.zeros((pair[1].shape[0], pair[1].shape[1], 1))
      for c in range(n_classes):
        positions = list(np.where(pair[1] == (c+1))[:2])
        positions.append(np.ones_like(positions[0]) * c)
        y[tuple(positions)] = 1
        w[tuple(positions[:2])] = class_weights[c]
      ys.append(y)
      ws.append(w)
    elif label_input is None:
      pass
    else:
      raise ValueError("Label type not recognized")

  Xs = np.stack(Xs, 0)
  Xs = Xs.astype(float)
  if label_input is not None:
    ys = np.stack(ys, 0)
    ws = np.stack(ws, 0)
    return Xs, np.concatenate([ys, ws], -1)
  else:
    return Xs, None



def plot_prediction_prob(d1, path):
  assert len(d1.shape) == 3
  assert d1.shape[2] == 3

  mat = np.zeros((d1.shape[0], d1.shape[1], 4))
  mat[:, :, :3] += d1[:, :, 1:2] * np.array([200, 130, 0]).reshape((1, 1, 3))
  mat[:, :, -1] += d1[:, :, 1] * 255
  mat[:, :, :3] += d1[:, :, 2:3] * np.array([75, 25, 230]).reshape((1, 1, 3))
  mat[:, :, -1] += d1[:, :, 2] * 255
  
  cv2.imwrite(path, mat)
  return


def predict_whole_map(file_path, 
                      model, 
                      out_file_path=None, 
                      n_classes=3, 
                      batch_size=8, 
                      n_supp=5, 
                      time_slices=1):
  inputs = load_input(file_path)
  x_size = model.input_shape[0]
  y_size = model.input_shape[1]

  total_outputs = []
  for t in range(inputs.shape[0] - (time_slices - 1)):
    print("Predicting %d" % t)
    inp = inputs[t:(t+time_slices)]
    # Matching dimensions
    assert inp.shape[1] % x_size == 0
    assert inp.shape[2] % y_size == 0
    assert inp.shape[3] == model.input_shape[2]
    rows = inp.shape[1] // x_size
    columns = inp.shape[2] // y_size

    batch_inputs = []
    outputs = []
    for r in range(rows):
      for c in range(columns):
        patch_inp = inp[:, r*x_size:(r+1)*x_size, c*y_size:(c+1)*y_size]
        if time_slices == 1:
          patch_inp = patch_inp[0]
        patch_inp = patch_inp  / CHANNEL_MAX.reshape((1, 1, -1))
        batch_inputs.append((patch_inp, None))
        if len(batch_inputs) == batch_size:
          batch_outputs = model.predict(batch_inputs, label_input=None)
          outputs.extend(batch_outputs)
          batch_inputs = []
    if len(batch_inputs) > 0:
      batch_outputs = model.predict(batch_inputs, label_input=None)
      outputs.extend(batch_outputs)
      batch_inputs = []
    
    ct = 0
    concatenated_output = -np.ones((inp.shape[1], inp.shape[2], n_classes))
    for r in range(rows):
      for c in range(columns):
        concatenated_output[r*x_size:(r+1)*x_size, c*y_size:(c+1)*y_size] = outputs[ct]
        ct += 1
    
    for i_supp in range(n_supp):
      x_offset = np.random.randint(1, x_size)
      y_offset = np.random.randint(1, y_size)
      batch_inputs = []
      outputs = []
      for r in range(rows - 1):
        for c in range(columns - 1):
          patch_inp = inp[:, (x_offset + r*x_size):(x_offset + (r+1)*x_size), 
                          (y_offset + c*y_size):(y_offset + (c+1)*y_size)]
          if time_slices == 1:
            patch_inp = patch_inp[0]
          patch_inp = patch_inp  / CHANNEL_MAX.reshape((1, 1, -1))
          batch_inputs.append((patch_inp, None))
          if len(batch_inputs) == batch_size:
            batch_outputs = model.predict(batch_inputs, label_input=None)
            outputs.extend(batch_outputs)
            batch_inputs = []
      if len(batch_inputs) > 0:
        batch_outputs = model.predict(batch_inputs, label_input=None)
        outputs.extend(batch_outputs)
        batch_inputs = []

      supp_output = np.copy(concatenated_output)
      ct = 0
      for r in range(rows - 1):
        for c in range(columns - 1):
          supp_output[(x_offset + r*x_size):(x_offset + (r+1)*x_size), 
                      (y_offset + c*y_size):(y_offset + (c+1)*y_size)] = outputs[ct]
          ct += 1
      concatenated_output = (concatenated_output * (i_supp + 1) + supp_output)/(i_supp + 2)
    total_outputs.append(concatenated_output)
  total_outputs = np.stack(total_outputs, 0)
  
  if out_file_path is None:
    out_file_path = os.path.splitext(file_path)[0] + '_NNProbabilities'
  np.save(out_file_path, total_outputs)
  
  cv2.imwrite(os.path.splitext(file_path)[0] + '.png', inputs[0, :, :, 0])
  plot_prediction_prob(total_outputs[0], os.path.splitext(file_path)[0] + '_NNpred.png')
  
  
#def read_XY(patches):
#  X = []
#  y = []
#  for pair in patches:
#    z_size = pair[0].shape[0]
#    center = z_size // 2
#    
#    X.append(pair[0][center])
#    if pair[1] is not None:
#      y.append(pair[1])
#    else:
#      y.append(None)
# 
#  X = np.stack(X, 0) # values should range from -3 to 3
#  X = np.clip(X, -3.0, 3.0)/6. + 0.5
#  y = np.stack(y, 0)
#  return X, y
#
#def display_samples(patches):
#  import matplotlib.pyplot as plt
#  X, y = read_XY(patches)
#  random_inds = np.random.choice(len(X), (8,), replace=False)
#  for i in range(8):
#    plt.subplot(4, 4, i*2 + 1)
#    plt.axis('off')
#    plt.imshow(X[random_inds[i], :, :])
#    
#    plt.subplot(4, 4, i*2 + 2)
#    plt.axis('off')
#    plt.imshow(y[random_inds[i], :, :])
#    plt.axis('off')
#  plt.savefig('./sample.png', dpi=600)

