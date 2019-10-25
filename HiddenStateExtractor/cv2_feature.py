#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 09:53:46 2019

@author: michaelwu
"""
import cv2
import h5py
import numpy as np
import scipy
import pickle
import random
import os
import matplotlib.pyplot as plt
from .naive_imagenet import preprocess, read_file_path, DATA_ROOT, CHANNEL_MAX
import multiprocessing as mp

# Feature extractor
def extract_features(x, vector_size=32):
  x = x.astype('uint8')
  try:
    dscs = []
    alg = cv2.KAZE_create()
    for x_slice in x:
      # finding image keypoints
      kps = alg.detect(x_slice)
      kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
      # computing descriptors vector
      kps, dsc = alg.compute(x_slice, kps)
      dsc = dsc.flatten()
      # Padding
      needed_size = (vector_size * 64)
      if dsc.size < needed_size:
        dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
      dscs.append(dsc)
    dscs = np.stack(dscs, 0)
  except Exception as e:
    print('Error: ' + str(e))
    return None
  return dscs

def worker(f_n):
  x = preprocess(f_n, cs=[0, 1], channel_max=CHANNEL_MAX)
  y = extract_features(x, vector_size=32)
  return y

def extract_density(f_n):
  with h5py.File(f_n, 'r') as f:
    dat = f['masked_mat']
    phase = np.array(dat[:, :, 0]) / 65535.
    mask = np.array(dat[:, :, 2])
    
    bg_density = np.median(phase[np.where(mask == 0)])
    peak_density = (phase * mask).max() - bg_density
    total_density = ((phase - bg_density) * mask).sum()
  return peak_density, total_density
  
if __name__ == '__main__':
  fs = read_file_path(DATA_ROOT + '/Data/StaticPatches')
  output = {}
  
  pool = mp.Pool(8)
  for f in fs:
    output[f] = pool.apply_async(worker, args=(f,))
  pool.close()
  pool.join()
  
  for f in fs:
    output[f] = output[f].get()  
  with open('/mnt/comp_micro/Projects/CellVAE/Data/EncodedKAZE.pkl', 'wb') as f:
    pickle.dump(output, f)