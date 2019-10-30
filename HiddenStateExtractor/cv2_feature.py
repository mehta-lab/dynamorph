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

def get_size(dat):
  mask = np.array(dat[:, :, 2])
  contours, _ = cv2.findContours(mask.astype('uint8'), 1, 2)
  areas = [cv2.contourArea(cnt) for cnt in contours]
  return mask.sum(), np.max(areas)

def get_density(dat):
  """ Return: 2 * (top-1, 95% quantile, average of top-200, sum) """
  phase = np.array(dat[:, :, 0]) / 65535.
  retardance = np.array(dat[:, :, 1]) / 65535.
  mask = np.array(dat[:, :, 2])
  
  bg_phase = np.median(phase[np.where(mask == 0)])
  bg_retardance = np.median(retardance[np.where(mask == 0)])

  peak_phase = ((phase - bg_phase) * mask).max()
  sum_phase = ((phase - bg_phase) * mask).sum()
  phase_vals = (phase - bg_phase)[np.where(mask)]
  quantile_phase = np.quantile(phase_vals, 0.95)
  top200_phase = np.mean(sorted(phase_vals)[-200:])

  peak_retardance = ((retardance - bg_retardance) * mask).max()
  sum_retardance = ((retardance - bg_retardance) * mask).sum()
  retardance_vals = (retardance - bg_retardance)[np.where(mask)]
  quantile_retardance = np.quantile(retardance_vals, 0.95)
  top200_retardance = np.mean(sorted(retardance_vals)[-200:])

  return (peak_phase, quantile_phase, top200_phase, sum_phase), \
         (peak_retardance, quantile_retardance, top200_retardance, sum_retardance)

def get_aspect_ratio(dat):
  """Return: w, h, angle(adjusted) """
  contours, _ = cv2.findContours(dat[:, :, 2].astype('uint8'), 1, 2)
  areas = [cv2.contourArea(cnt) for cnt in contours]
  rect = cv2.minAreaRect(contours[np.argmax(areas)])
  w, h = rect[1]
  if w < h:
    ang = rect[2] + 180
  else:
    ang = rect[2] + 90
  return w, h, ang

def get_aspect_ratio_no_rotation(dat):
  """ Return: w, h """
  contours, _ = cv2.findContours(dat[:, :, 2].astype('uint8'), 1, 2)
  areas = [cv2.contourArea(cnt) for cnt in contours]
  rect = cv2.boundingRect(contours[np.argmax(areas)])
  return rect[2], rect[3]


if __name__ == '__main__':
  fs = read_file_path(DATA_ROOT + '/Data/StaticPatches')
  sites = ['D%d-Site_%d' % (i, j) for j in range(9) for i in range(3, 6)]
  
  sizes = {}
  densities = {}
  aprs = {}
  aprs_nr = {}
  for site in sites:
    dat = pickle.load(open('./%s_all_patches.pkl' % site, 'rb'))
    for f in dat:
      d = dat[f]["masked_mat"]
      sizes[f] = get_size(d)
      densities[f] = get_density(d)
      aprs[f] = get_aspect_ratio(d)
      aprs_nr[f] = get_aspect_ratio_no_rotation(d)