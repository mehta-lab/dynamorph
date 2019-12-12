#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:43:41 2019

@author: zqwu
"""

import cv2
import numpy as np
import h5py
import os
from sklearn.cluster import DBSCAN
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.signal import convolve2d
import pickle

size1 = 11
filter1 = np.zeros((size1, size1), dtype=int)
for i in range(size1):
  for j in range(size1):
    if np.sqrt((i-size1//2)**2 + (j-size1//2)**2) <= size1//2:
      filter1[i, j] = 1

size2 = 21
filter2 = np.zeros((size2, size2), dtype=int)
for i in range(size2):
  for j in range(size2):
    if np.sqrt((i-size2//2)**2 + (j-size2//2)**2) < size2//2:
      filter2[i, j] = 1

def select_window(mat, window, padding=0.):
  # Select the submatrix from mat according to window, negative boundaries allowed (padded with -1)
  if window[0][0] < 0:
    output_mat = np.concatenate([padding * np.ones_like(mat[window[0][0]:]), mat[:window[0][1]]], 0)
  elif window[0][1] > 2048:
    output_mat = np.concatenate([mat[window[0][0]:], padding * np.ones_like(mat[:(window[0][1] - 2048)])], 0)
  else:
    output_mat = mat[window[0][0]:window[0][1]]

  if window[1][0] < 0:
    output_mat = np.concatenate([padding * np.ones_like(output_mat[:, window[1][0]:]), output_mat[:, :window[1][1]]], 1)
  elif window[1][1] > 2048:
    output_mat = np.concatenate([output_mat[:, window[1][0]:], padding * np.ones_like(output_mat[:, :(window[1][1] - 2048)])], 1)
  else:
    output_mat = output_mat[:, window[1][0]:window[1][1]]
  return output_mat

def within_range(r, pos):
  if pos[0] >= r[0][1] or pos[0] < r[0][0]:
    return False
  if pos[1] >= r[1][1] or pos[1] < r[1][0]:
    return False
  return True

def remove_close_points(masking_points, target_points):
  dist = np.abs(np.array(masking_points).reshape((-1, 1, 2)) - np.array(target_points).reshape((1, -1, 2))).sum(2).min(1)
  return [p for i, p in masking_points if dist[i] > 5]

def generate_mask(positions, positions_labels, cell_id, window, window_segmentation):
  x_size = window[0][1] - window[0][0]
  y_size = window[1][1] - window[1][0]
  remove_mask = np.zeros((x_size, y_size), dtype=int)
  target_mask = np.zeros((x_size, y_size), dtype=int)

  for i, p in enumerate(positions):
    if not within_range(window, p):
      continue
    if positions_labels[i] != cell_id and positions_labels[i] >= 0:
      remove_mask[p[0] - window[0][0], p[1] - window[1][0]] = 1
    if positions_labels[i] == cell_id:
      target_mask[p[0] - window[0][0], p[1] - window[1][0]] = 1

  remove_mask = np.sign(convolve2d(remove_mask, filter1, mode='same'))
  target_mask2 = np.sign(convolve2d(target_mask, filter2, mode='same'))
  remove_mask = ((remove_mask - target_mask2) > 0) * 1

  remove_mask[np.where(window_segmentation[:, :, 0] == -1)] = 1
  return remove_mask.reshape((x_size, y_size, 1)), \
      target_mask.reshape((x_size, y_size, 1)), \
      target_mask2.reshape((x_size, y_size, 1))

def instance_clustering(cell_segmentation, 
                        ct_thr=(500, 12000), 
                        instance_map=True, 
                        map_path=None, 
                        fg_thr=0.3,
                        DBSCAN_thr=(10, 250)):
  all_cells = cell_segmentation[:, :, 0] < fg_thr
  positions = np.array(list(zip(*np.where(all_cells))))
  if len(positions) < 1000:
    return ([], [], []), np.zeros((0, 2), dtype=int), np.zeros((0,), dtype=int)

  clustering = DBSCAN(eps=DBSCAN_thr[0], min_samples=DBSCAN_thr[1]).fit(positions)
  positions_labels = clustering.labels_

  cell_ids, point_cts = np.unique(positions_labels, return_counts=True)
  
  mg_cell_positions = []
  non_mg_cell_positions = []
  other_cells = []
  
  for cell_id, ct in zip(cell_ids, point_cts):
    if cell_id < 0:
      continue
    if ct <= ct_thr[0] or ct >= ct_thr[1]:
      continue
    points = positions[np.where(positions_labels == cell_id)[0]]

    mean_pos = np.mean(points, 0).astype(int)
    window = [(mean_pos[0]-128, mean_pos[0]+128), (mean_pos[1]-128, mean_pos[1]+128)]
    outliers = [p for p in points if not within_range(window, p)]
    if len(outliers) > len(points) * 0.05:
      continue

    cell_segmentation_labels = cell_segmentation[points[:, 0], points[:, 1]]
    mg_ratio = (np.argmax(cell_segmentation_labels, 1) == 1).sum()/len(points)
    non_mg_ratio = (np.argmax(cell_segmentation_labels, 1) == 2).sum()/len(points)
    if mg_ratio > 0.9:
      mg_cell_positions.append((cell_id, mean_pos))
    elif non_mg_ratio > 0.9:
      non_mg_cell_positions.append((cell_id, mean_pos))
    else:
      other_cells.append((cell_id, mean_pos))

  if instance_map and map_path is not None:
    segmented = np.zeros(cell_segmentation.shape[:2]) - 1
    for cell_id, mean_pos in mg_cell_positions:
      points = positions[np.where(positions_labels == cell_id)[0]]
      for p in points:
        segmented[p[0], p[1]] = cell_id%10

    for cell_id, mean_pos in non_mg_cell_positions:
      points = positions[np.where(positions_labels == cell_id)[0]]
      for p in points:
        segmented[p[0], p[1]] = cell_id%10

    plt.clf()
    cmap = matplotlib.cm.get_cmap('tab10')
    cmap.set_under(color='k')
    plt.imshow(segmented, cmap=cmap, vmin=-0.001, vmax=10.001)
    

    font_mg = {'color': 'white', 'size': 4}
    font_non_mg = {'color': 'red', 'size': 4}
    for cell_id, mean_pos in mg_cell_positions:
      plt.text(mean_pos[1], mean_pos[0], str(cell_id), fontdict=font_mg)
    for cell_id, mean_pos in non_mg_cell_positions:
      plt.text(mean_pos[1], mean_pos[0], str(cell_id), fontdict=font_non_mg)
    
    plt.axis('off')
    plt.savefig(map_path, dpi=300)
  return (mg_cell_positions, non_mg_cell_positions, other_cells), positions, positions_labels

def get_cell_rect_angle(tm):
  contours, _ = cv2.findContours(tm.astype('uint8'), 1, 2)
  areas = [cv2.contourArea(cnt) for cnt in contours]
  rect = cv2.minAreaRect(contours[np.argmax(areas)])
  w, h = rect[1]
  ang = rect[2]
  if w < h:
    ang = ang - 90
  return ang

if __name__ == '__main__':
  
  path = '/mnt/comp_micro/Projects/CellVAE'
  sites = ['D%d-Site_%d' % (i, j) for j in range(9) for i in range(3, 6)]
  CHANNEL_MAX = [65535., 65535.]
  align_long_axis = True
  if align_long_axis:
    # window size 256 * sqrt(2)
    window_size = 364
  else:
    window_size = 256
  
  for site in sites:
    site_data = {}
    print("On site %s" % site)
    if os.path.exists('../%s_all_patches_rotated.pkl' % site):
      continue
    image_stack = np.load(os.path.join(path, 'Combined', '%s.npy' % site))
    segmentation_stack = np.load(os.path.join(path, 'Combined', '%s_NNProbabilities.npy' % site))

    if not os.path.exists(path + '/Data/StaticPatches/%s' % site):
      os.mkdir(path + '/Data/StaticPatches/%s' % site)

    if not os.path.exists(path + '/Data/StaticPatches/%s/cell_positions.pkl' % site) or \
       not os.path.exists(path + '/Data/StaticPatches/%s/cell_pixel_assignments.pkl' % site):
      cell_positions = {}
      cell_pixel_assignments = {}
      for t_point in range(image_stack.shape[0]):
        print("\tClustering time %d" % t_point)
        cell_segmentation = segmentation_stack[t_point, :, :]
        instance_map_path = path + '/Data/StaticPatches/%s/segmentation_%d.png' % (site, t_point)
        res = instance_clustering(cell_segmentation, instance_map=True, map_path=instance_map_path)

        cell_positions[t_point] = res[0] # MG, Non-MG, Chimeric Cells
        cell_pixel_assignments[t_point] = res[1:]


      with open(path + '/Data/StaticPatches/%s/cell_positions.pkl' % site, 'wb') as f:
        pickle.dump(cell_positions, f)
      with open(path + '/Data/StaticPatches/%s/cell_pixel_assignments.pkl' % site, 'wb') as f:
        pickle.dump(cell_pixel_assignments, f)
    else:
      cell_positions = pickle.load(open(path + '/Data/StaticPatches/%s/cell_positions.pkl' % site, 'rb'))
      cell_pixel_assignments = pickle.load(open(path + '/Data/StaticPatches/%s/cell_pixel_assignments.pkl' % site, 'rb'))

    ### Generate time-independent static patches ###
    for t_point in range(image_stack.shape[0]):
      print("\tWriting time %d" % t_point)
      raw_image = image_stack[t_point, :, :]
      cell_segmentation = segmentation_stack[t_point, :, :]
      
      positions, positions_labels = cell_pixel_assignments[t_point]      
      mg_cells, non_mg_cells, other_cells = cell_positions[t_point]
      background_pool = raw_image[np.where(cell_segmentation[:, :, 0] > 0.9)]
      background_pool = np.median(background_pool, 0)
      background_filling = np.ones((window_size, window_size, 1)) * background_pool.reshape((1, 1, -1))


      for cell_id, cell_position in mg_cells:
        window = [(cell_position[0]-window_size//2, cell_position[0]+window_size//2),
                  (cell_position[1]-window_size//2, cell_position[1]+window_size//2)]
        window_segmentation = select_window(cell_segmentation, window, padding=-1)
        
        remove_mask, tm, tm2 = generate_mask(positions, 
                                             positions_labels, 
                                             cell_id, 
                                             window, 
                                             window_segmentation)

        output_mat = select_window(raw_image, window, padding=0)
        masked_output_mat = output_mat * (1 - remove_mask) + background_filling * remove_mask

        if align_long_axis:
          ang = get_cell_rect_angle(tm)

          M = cv2.getRotationMatrix2D((window_size/2, window_size/2), ang, 1)
          tm_ = cv2.warpAffine(tm.astype('uint8'), M, (window_size, window_size)).reshape((window_size, window_size, 1))
          tm2_ = cv2.warpAffine(tm2.astype('uint8'), M, (window_size, window_size)).reshape((window_size, window_size, 1))
          output_mat_ = cv2.warpAffine(output_mat.astype('uint16'), M, (window_size, window_size))
          masked_output_mat_ = cv2.warpAffine(masked_output_mat.astype('uint16'), M, (window_size, window_size))

          # HARDCODED for size to be 256 * 256
          tm = tm_[(window_size//2 - 128):(window_size//2 + 128),
                   (window_size//2 - 128):(window_size//2 + 128)]
          tm2 = tm2_[(window_size//2 - 128):(window_size//2 + 128),
                     (window_size//2 - 128):(window_size//2 + 128)]
          output_mat = output_mat_[(window_size//2 - 128):(window_size//2 + 128),
                                   (window_size//2 - 128):(window_size//2 + 128)]
          masked_output_mat = masked_output_mat_[(window_size//2 - 128):(window_size//2 + 128),
                                                 (window_size//2 - 128):(window_size//2 + 128)]

        # Just to prevent backward compatibility issue cast to int64 and float 64 respectively
        output_mat = np.concatenate([output_mat, tm, tm2], 2).astype('int64')
        masked_output_mat = np.concatenate([masked_output_mat, tm, tm2], 2).astype('float64')
        #cv2.imwrite(path + '/Data/StaticPatches/%s/%d_%d.png' % (site, t_point, cell_id), output_mat[:, :, 0]/CHANNEL_MAX[0] * 255.)
        #cv2.imwrite(path + '/Data/StaticPatches/%s/%d_%d_masked.png' % (site, t_point, cell_id), masked_output_mat[:, :, 0]/CHANNEL_MAX[0] * 255.)
        try:
          with h5py.File(path + '/Data/StaticPatches/%s/%d_%d_rotated.h5' % (site, t_point, cell_id), 'w') as f:
            f.create_dataset("mat", data=output_mat)
            f.create_dataset("masked_mat", data=masked_output_mat)
        except Exception as e:
          print(e)
          print("ERROR ON PATCH %s/Data/StaticPatches/%s/%d_%d_rotated.h5" % (path, site, t_point, cell_id))
        site_data[path + '/Data/StaticPatches/%s/%d_%d.h5' % (site, t_point, cell_id)] = {"mat": output_mat, "masked_mat": masked_output_mat}
    with open('../%s_all_patches_rotated.pkl' % site, 'wb') as f:
      pickle.dump(site_data, f)
