#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 15:20:21 2019

@author: michaelwu
"""
import numpy as np
import cv2
import os
import pickle
import torch as t
import torch
import h5py
import pandas as pd
from NNsegmentation.models import Segment
from NNsegmentation.data import predict_whole_map
from SingleCellPatch.instance_clustering import instance_clustering, within_range
from SingleCellPatch.generate_trajectories import frame_matching
import matplotlib
from matplotlib import cm
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import seaborn as sns
import imageio
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr
from HiddenStateExtractor.vq_vae import VQ_VAE, CHANNEL_MAX, CHANNEL_VAR, CHANNEL_RANGE, prepare_dataset, rescale
from HiddenStateExtractor.naive_imagenet import read_file_path, DATA_ROOT
from HiddenStateExtractor.morphology_clustering import select_clean_trajecteories, Kmean_on_short_trajs
from HiddenStateExtractor.movement_clustering import save_traj
import statsmodels.api as sm
import scipy

color_mg = np.array([240, 94, 56], dtype='uint8')
color_nonmg = np.array([66, 101, 251], dtype='uint8')
color_bg = np.array([150, 150, 150], dtype='uint8')
color_fg = (color_mg * 0.7 + color_nonmg * 0.3).astype('uint8')
sites = ['D%d-Site_%d' % (i, j) for j in range(9) for i in range(3, 6)]

# Contrast Setting
phase_a = 2.
phase_b = -50000.
retardance_a = 3.
retardance_b = 0.

def enhance_contrast(mat, a=1.5, b=-10000):
  mat2 = cv2.addWeighted(mat, a, mat, 0, b)
  return mat2

def plot_patches(names, out_paths, masked=True):
  sites = set(n.split('/')[-2] for n in names)
  for site in sites:
    image_inds = [i for i, n in enumerate(names) if n.split('/')[-2] == site]
    site_dat = pickle.load(open('../data_temp/%s_all_patches.pkl' % site, 'rb'))
    for i in image_inds:
      if masked:
        mat = site_dat[names[i]]["masked_mat"][:, :, 0]
      else:
        mat = site_dat[names[i]]["mat"][:, :, 0]
      mat2 = np.clip(enhance_contrast(mat, phase_a, phase_b), 0, 65535).astype('uint16')
      cv2.imwrite(out_paths[i], mat2.astype('uint16'))

def save_movie(names, path, masked=True):
  sites = set(n.split('/')[-2] for n in names)
  assert len(sites) == 1
  site_dat = pickle.load(open('../data_temp/%s_all_patches.pkl' % list(sites)[0], 'rb'))
  stacks = []
  for n in names:
    if masked:
      mat = site_dat[n]["masked_mat"][:, :, 0]
    else:
      mat = site_dat[n]["mat"][:, :, 0]
    mat2 = np.clip(enhance_contrast(mat, phase_a, phase_b), 0, 65535).astype('uint16')
    stacks.append(mat2)
  imageio.mimsave(path, np.stack(stacks, 0))

############################################################################################################

# Fig 2 A1
# Raw input (phase channel)
RAW_DATA_PATH = '/mnt/comp_micro/Projects/CellVAE/Combined'
raw_input_stack = np.load(RAW_DATA_PATH + '/D5-Site_0.npy')
raw_input = raw_input_stack[0, :, :, 0:1]
cv2.imwrite('/home/michaelwu/fig2_raw.png', raw_input)

##########

# Supp Video 1
raw_movie = [cv2.resize(slic[:, :, 0], (512, 512)) for slic in raw_input_stack]
imageio.mimsave('/home/michaelwu/supp_video1_sample_movie.gif', np.stack(raw_movie, 0))

##########

# Fig 2 A2
# Human annotations of (background, mg, non-mg)
annotations = np.load(RAW_DATA_PATH + '/D5-Site_0_Annotations.npy')
annotations = annotations[0]
mat = np.zeros((raw_input.shape[0], raw_input.shape[1], 3), dtype='uint8')
mat[:, :] = (raw_input / 256).astype('uint8')
alpha = 0.7
mat[np.where(annotations == 3)[:2]] = (1 - alpha) * mat[np.where(annotations == 3)[:2]] + alpha * color_nonmg.reshape((1, 3))
mat[np.where(annotations == 2)[:2]] = (1 - alpha) * mat[np.where(annotations == 2)[:2]] + alpha * color_mg.reshape((1, 3))
mat[np.where(annotations == 1)[:2]] = (1 - alpha) * mat[np.where(annotations == 1)[:2]] + alpha * color_bg.reshape((1, 3))
cv2.imwrite('/home/michaelwu/fig2_annotations.png', mat)

##########

# Fig 2 A3
# U-Net prediction
NN_predictions_stack = np.load(RAW_DATA_PATH + '/D5-Site_0_NNProbabilities.npy')
NN_predictions = NN_predictions_stack[0]
mat = np.zeros((raw_input.shape[0], raw_input.shape[1], 3), dtype='uint8')
mat[:, :] = (raw_input / 256).astype('uint8')
alpha = 0.7
mat = mat * (1 - (alpha * NN_predictions[:, :, 0:1])) + np.ones_like(mat) * color_bg.reshape((1, 1, 3)) * (alpha * NN_predictions[:, :, 0:1])
nonmg_positions = np.where(NN_predictions[:, :, 2] > 0.5)[:2]
mat[nonmg_positions] = (mat * (1 - (alpha * NN_predictions[:, :, 2:3])) + np.ones_like(mat) * color_nonmg.reshape((1, 1, 3)) * (alpha * NN_predictions[:, :, 2:3]))[nonmg_positions]
mg_positions = np.where(NN_predictions[:, :, 1] > 0.5)[:2]
mat[mg_positions] = (mat * (1 - (alpha * NN_predictions[:, :, 1:2])) + np.ones_like(mat) * color_mg.reshape((1, 1, 3)) * (alpha * NN_predictions[:, :, 1:2]))[mg_positions]
cv2.imwrite('/home/michaelwu/fig2_nn_predictions.png', mat)

##########

# Supp Fig 1 RF
slice_num = 11
raw_input_off = raw_input_stack[slice_num, :, :, 0:1]
RF_predictions_stack = np.load(RAW_DATA_PATH + '/D5-Site_0_RFProbabilities.npy')
RF_predictions_off = RF_predictions_stack[slice_num]
cv2.imwrite('/home/michaelwu/supp_fig1_raw.png', raw_input_off)

mat = np.zeros((raw_input_off.shape[0], raw_input_off.shape[1], 3), dtype='uint8')
mat[:, :] = (raw_input_off / 256).astype('uint8')
alpha = 0.7
mg_positions = np.where(RF_predictions_off[:, :, 1] > 0.5)[:2]
nonmg_positions = np.where(RF_predictions_off[:, :, 2] > 0.5)[:2]
mat = mat * (1 - (alpha * RF_predictions_off[:, :, 0:1])) + np.ones_like(mat) * color_bg.reshape((1, 1, 3)) * (alpha * RF_predictions_off[:, :, 0:1])
mat[mg_positions] = (mat * (1 - (alpha * RF_predictions_off[:, :, 1:2])) + np.ones_like(mat) * color_mg.reshape((1, 1, 3)) * (alpha * RF_predictions_off[:, :, 1:2]))[mg_positions]
mat[nonmg_positions] = (mat * (1 - (alpha * RF_predictions_off[:, :, 2:3])) + np.ones_like(mat) * color_nonmg.reshape((1, 1, 3)) * (alpha * RF_predictions_off[:, :, 2:3]))[nonmg_positions]
cv2.imwrite('/home/michaelwu/supp_fig1_rf_predictions_annotation_only.png', mat)

##########

# Supp Fig 1 NN-only
model = Segment(input_shape=(256, 256, 2), 
                unet_feat=32,
                fc_layers=[64, 32],
                n_classes=3,
                model_path='./NNsegmentation/temp_save')
model.load(model.model_path + '/stage0_0.h5')
NN_predictions_off2 = predict_whole_map(raw_input_stack[slice_num:(slice_num + 1)], model, n_supp=20)[0]
mat = np.zeros((raw_input_off.shape[0], raw_input_off.shape[1], 3), dtype='uint8')
mat[:, :] = (raw_input_off / 256).astype('uint8')
alpha = 0.7
mg_positions = np.where(NN_predictions_off2[:, :, 1] > 0.5)[:2]
nonmg_positions = np.where(NN_predictions_off2[:, :, 2] > 0.5)[:2]
mat = mat * (1 - (alpha * NN_predictions_off2[:, :, 0:1])) + np.ones_like(mat) * color_bg.reshape((1, 1, 3)) * (alpha * NN_predictions_off2[:, :, 0:1])
mat[mg_positions] = (mat * (1 - (alpha * NN_predictions_off2[:, :, 1:2])) + np.ones_like(mat) * color_mg.reshape((1, 1, 3)) * (alpha * NN_predictions_off2[:, :, 1:2]))[mg_positions]
mat[nonmg_positions] = (mat * (1 - (alpha * NN_predictions_off2[:, :, 2:3])) + np.ones_like(mat) * color_nonmg.reshape((1, 1, 3)) * (alpha * NN_predictions_off2[:, :, 2:3]))[nonmg_positions]
cv2.imwrite('/home/michaelwu/supp_fig1_nn_predictions_annotation_only.png', mat)

##########

# Supp Fig 1 NN-combined
model.load(model.model_path + '/final.h5')
NN_predictions_off = predict_whole_map(raw_input_stack[slice_num:(slice_num + 1)], model, n_supp=20)[0]
mat = np.zeros((raw_input_off.shape[0], raw_input_off.shape[1], 3), dtype='uint8')
mat[:, :] = (raw_input_off / 256).astype('uint8')
alpha = 0.7
mg_positions = np.where(NN_predictions_off[:, :, 1] > 0.5)[:2]
nonmg_positions = np.where(NN_predictions_off[:, :, 2] > 0.5)[:2]
mat = mat * (1 - (alpha * NN_predictions_off[:, :, 0:1])) + np.ones_like(mat) * color_bg.reshape((1, 1, 3)) * (alpha * NN_predictions_off[:, :, 0:1])
mat[mg_positions] = (mat * (1 - (alpha * NN_predictions_off[:, :, 1:2])) + np.ones_like(mat) * color_mg.reshape((1, 1, 3)) * (alpha * NN_predictions_off[:, :, 1:2]))[mg_positions]
mat[nonmg_positions] = (mat * (1 - (alpha * NN_predictions_off[:, :, 2:3])) + np.ones_like(mat) * color_nonmg.reshape((1, 1, 3)) * (alpha * NN_predictions_off[:, :, 2:3]))[nonmg_positions]
cv2.imwrite('/home/michaelwu/supp_fig1_nn_predictions.png', mat)

##########

# Fig 2 B1
# Instance separation
cells, positions, positions_labels = instance_clustering(NN_predictions, fg_thr=0.2)
mg_cell_positions, non_mg_cell_positions, other_cells = cells
mat = np.zeros((raw_input.shape[0], raw_input.shape[1], 3), dtype='uint8')
mat[:, :] = (raw_input / 256).astype('uint8')
cmap = matplotlib.cm.get_cmap('tab10')
alpha = 0.7
for cell_id, mean_pos in mg_cell_positions:
  points = positions[np.where(positions_labels == cell_id)[0]]
  for p in points:
    mat[p[0], p[1]] = (1 - alpha) * mat[p[0], p[1]] + alpha * np.array(cmap.colors[cell_id%10]) * 255
for cell_id, mean_pos in non_mg_cell_positions:
  points = positions[np.where(positions_labels == cell_id)[0]]
  for p in points:
    mat[p[0], p[1]] = (1 - alpha) * mat[p[0], p[1]] + alpha * np.array(cmap.colors[cell_id%10]) * 255
for cell_id, mean_pos in other_cells:
  points = positions[np.where(positions_labels == cell_id)[0]]
  for p in points:
    mat[p[0], p[1]] = (1 - alpha) * mat[p[0], p[1]] + alpha * np.array(cmap.colors[cell_id%10]) * 255
cv2.imwrite('/home/michaelwu/fig2_nn_predictions_instance.png', mat)
cv2.imwrite('/home/michaelwu/fig2_nn_predictions_instance_small.png', mat[:940, :940])

##########

# Fig 2 B2 - left
# Generate bounding boxes
mat = np.zeros((raw_input.shape[0], raw_input.shape[1], 3), dtype='uint8')
mat[:, :] = (raw_input / 256).astype('uint8')
def add_box(mat, box_center, color):
  length = mat.shape[0]
  box_range = [(max(box_center[0] - 64., 0), min(box_center[0] + 64., length)),
               (max(box_center[1] - 64., 0), min(box_center[1] + 64., length))] # assuming square
  # Left edge
  x = box_range[0][0]
  x_ = (int(max(x - 3., 0)), int(min(x + 3., length)))
  mat[x_[0]:x_[1], int(box_range[1][0]):int(box_range[1][1])] = color.reshape((1, 1, 3))
  # Right edge
  x = box_range[0][1]
  x_ = (int(max(x - 3., 0)), int(min(x + 3., length)))
  mat[x_[0]:x_[1], int(box_range[1][0]):int(box_range[1][1])] = color.reshape((1, 1, 3))
  # Top edge
  y = box_range[1][0]
  y_ = (int(max(y - 3., 0)), int(min(y + 3., length)))
  mat[int(box_range[0][0]):int(box_range[0][1]), y_[0]:y_[1]] = color.reshape((1, 1, 3))
  # Bottom edge
  y = box_range[1][1]
  y_ = (int(max(y - 3., 0)), int(min(y + 3., length)))
  mat[int(box_range[0][0]):int(box_range[0][1]), y_[0]:y_[1]] = color.reshape((1, 1, 3))
  return mat
for cell_id, mean_pos in non_mg_cell_positions:
  mat = add_box(mat, mean_pos, color_nonmg)
for cell_id, mean_pos in mg_cell_positions:
  mat = add_box(mat, mean_pos, color_mg)
cv2.imwrite('/home/michaelwu/fig2_nn_predictions_boxed_small.png', mat[:940, :940])

##########

# Fig 2 B2 - right
# Generate boxed samples
np.random.seed(123)
mg_inds = np.random.choice(np.arange(len(mg_cell_positions)), (30,), replace=False)
non_mg_inds = np.random.choice(np.arange(len(non_mg_cell_positions)), (5,), replace=False)
mat = np.zeros((raw_input.shape[0], raw_input.shape[1], 3), dtype='uint8')
mat[:, :] = (raw_input / 256).astype('uint8')
for i in mg_inds:
  mean_pos = mg_cell_positions[i][1]
  if within_range(((128, 940-128), (128, 940-128)), mean_pos):
    patch = mat[(mean_pos[0] - 128):(mean_pos[0] + 128),
                (mean_pos[1] - 128):(mean_pos[1] + 128)]
    cv2.imwrite('/home/michaelwu/fig2_nn_predictions_boxed_mg_%d.png' % mg_cell_positions[i][0], patch)
for i in non_mg_inds:
  mean_pos = non_mg_cell_positions[i][1]
  if within_range(((128, 940-128), (128, 940-128)), mean_pos):
    patch = mat[(mean_pos[0] - 128):(mean_pos[0] + 128),
                (mean_pos[1] - 128):(mean_pos[1] + 128)]
    cv2.imwrite('/home/michaelwu/fig2_nn_predictions_boxed_non_mg_%d.png' % non_mg_cell_positions[i][0], patch)

##########

# Fig 2 C1
# Frame Matching
frame0 = raw_input_stack[0, :, :, 0:1]
frame1 = raw_input_stack[1, :, :, 0:1]
pred0 = NN_predictions_stack[0]
pred1 = NN_predictions_stack[1]
res0 = instance_clustering(pred0, fg_thr=0.2)
res1 = instance_clustering(pred1, fg_thr=0.2)

cell_positions = {0: res0[0], 1: res1[0]}
cell_pixel_assignments = {0: res0[1:], 1: res1[1:]}
mg_positions_dict = {k: dict(cell_positions[k][0]) for k in cell_positions}
non_mg_positions_dict = {k: dict(cell_positions[k][1]) for k in cell_positions}
t_points = [0, 1]
intensities_dict = {}
for t_point in t_points:
  intensities_d = dict(zip(*np.unique(cell_pixel_assignments[t_point][1], return_counts=True)))
  intensities_d = {p[0]: intensities_d[p[0]] for p in cell_positions[t_point][0] + cell_positions[t_point][1]}
  intensities_dict[t_point] = intensities_d

mg_matchings = {}
non_mg_matchings = {}
for t_point in t_points[:-1]:
  ids1 = sorted(mg_positions_dict[t_point].keys())
  ids2 = sorted(mg_positions_dict[t_point+1].keys())      
  f1 = [mg_positions_dict[t_point][i] for i in ids1]
  f2 = [mg_positions_dict[t_point+1][i] for i in ids2]
  int1 = [intensities_dict[t_point][i] for i in ids1]
  int2 = [intensities_dict[t_point+1][i] for i in ids2]
  pairs = frame_matching(f1, f2, int1, int2, dist_cutoff=150)
  mg_matchings[t_point] = [(ids1[p1], ids2[p2]) for p1, p2 in pairs]
  
  ids1 = sorted(non_mg_positions_dict[t_point].keys())
  ids2 = sorted(non_mg_positions_dict[t_point+1].keys())
  f1 = [non_mg_positions_dict[t_point][i] for i in ids1]
  f2 = [non_mg_positions_dict[t_point+1][i] for i in ids2]
  int1 = [intensities_dict[t_point][i] for i in ids1]
  int2 = [intensities_dict[t_point+1][i] for i in ids2]
  pairs = frame_matching(f1, f2, int1, int2, dist_cutoff=150)
  non_mg_matchings[t_point] = [(ids1[p1], ids2[p2]) for p1, p2 in pairs]

mat0 = np.zeros((raw_input.shape[0], raw_input.shape[1], 3), dtype='uint8')
mat1 = np.zeros((raw_input.shape[0], raw_input.shape[1], 3), dtype='uint8')
mat0[:, :] = (frame0 / 256).astype('uint8')
mat1[:, :] = (frame1 / 256).astype('uint8')

cmap = matplotlib.cm.get_cmap('Set2')
np.random.seed(123)
plotted = []
for i in np.random.permutation(np.arange(len(mg_matchings[0]))):
  pair = mg_matchings[0][i]
  frame0_position = None
  for mg in cell_positions[0][0]:
    if mg[0] == pair[0]:
      frame0_position = mg[1]
      break  
  frame1_position = None
  for mg in cell_positions[1][0]:
    if mg[0] == pair[1]:
      frame1_position = mg[1]
      break
  if within_range(((128, 940-128), (128, 940-128)), frame0_position) and \
     within_range(((128, 940-128), (128, 940-128)), frame1_position):
    mat0 = add_box(mat0, frame0_position, np.array(cmap.colors[(len(plotted) + 1)%10]) * 255)
    mat1 = add_box(mat1, frame1_position, np.array(cmap.colors[(len(plotted) + 1)%10]) * 255)
    plotted.append((frame0_position, frame1_position))
    if len(plotted) > 3:
      break

cmap = matplotlib.cm.get_cmap('Set1')
for i in np.random.permutation(np.arange(len(non_mg_matchings[0]))):
  pair = non_mg_matchings[0][i]
  frame0_position = None
  for non_mg in cell_positions[0][1]:
    if non_mg[0] == pair[0]:
      frame0_position = non_mg[1]
      break
  frame1_position = None
  for non_mg in cell_positions[1][1]:
    if non_mg[0] == pair[1]:
      frame1_position = non_mg[1]
      break
  if within_range(((128, 940-128), (128, 940-128)), frame0_position) and \
     within_range(((128, 940-128), (128, 940-128)), frame1_position):
    mat0 = add_box(mat0, frame0_position, np.array(cmap.colors[(len(plotted) + 1)%10]) * 255)
    mat1 = add_box(mat1, frame1_position, np.array(cmap.colors[(len(plotted) + 1)%10]) * 255)
    plotted.append((frame0_position, frame1_position))
    if len(plotted) > 5:
      break

cv2.imwrite('/home/michaelwu/fig2_traj_matching_f0.png', mat0[:940, :940])
cv2.imwrite('/home/michaelwu/fig2_traj_matching_f1.png', mat1[:940, :940])

##########

# Fig 2 C2
# Sample plotted traj zoomed in
mg_trajectories, mg_trajectories_positions = pickle.load(open(os.path.split(RAW_DATA_PATH)[0] + '/Data/DynamicPatches/D5-Site_0/mg_traj.pkl', 'rb'))
non_mg_trajectories, non_mg_trajectories_positions = pickle.load(open(os.path.split(RAW_DATA_PATH)[0] + '/Data/DynamicPatches/D5-Site_0/non_mg_traj.pkl', 'rb'))

sample_non_mg_traj = non_mg_trajectories[0]
sample_non_mg_traj_positions = non_mg_trajectories_positions[0]
sample_mg_traj = mg_trajectories[2]
sample_mg_traj_positions = mg_trajectories_positions[2]

for i in [0, 1, 4, 8, 16]:
  mat = raw_input_stack[i][:, :, 0]
  center = sample_mg_traj_positions[i]
  mg_mat = mat[center[0]-128:center[0]+128,
               center[1]-128:center[1]+128]
  cv2.imwrite('/home/michaelwu/fig2_sample_mg_traj_%d.png' % i, enhance_contrast(mg_mat, 1.5, -10000))
  center = sample_non_mg_traj_positions[i]
  non_mg_mat = mat[center[0]-128:center[0]+128,
                   center[1]-128:center[1]+128]
  cv2.imwrite('/home/michaelwu/fig2_sample_non_mg_traj_%d.png' % i, enhance_contrast(non_mg_mat, 1.5, -10000))

##########

# Supp Video 2
# Sample trajectories
inds = [39, 15, 30, 43]
for i in inds:
  traj_name = 'D5-Site_0/%d' % i
  save_traj(traj_name, '/home/michaelwu/supp_video2_sample_traj_%d.gif' % i)
  names = ['/mnt/comp_micro/Projects/CellVAE/Data/StaticPatches/D5-Site_0/%d_%d.h5' % (j, mg_trajectories[i][j]) for j in sorted(mg_trajectories[i].keys())]
  save_movie(names, '/home/michaelwu/supp_video2_sample_traj_movie_%d.gif' % i, masked=False)

############################################################################################################

# Fig 3 A
# VAE illustration
cs = [0, 1]
input_shape = (128, 128)
gpu = True
# Order for `dataset`, `relations`
fs_ = pickle.load(open('./HiddenStateExtractor/file_paths_bkp.pkl', 'rb'))
# Order for `trajs`
fs = sorted(pickle.load(open('./HiddenStateExtractor/file_paths_bkp.pkl', 'rb')))
dataset = torch.load('StaticPatchesAll.pt')
dataset = rescale(dataset)
model = VQ_VAE(alpha=0.0005, gpu=gpu)
model = model.cuda()
model.load_state_dict(torch.load('./HiddenStateExtractor/save_0005_bkp4.pt'))

sample_fs = ['/mnt/comp_micro/Projects/CellVAE/Data/StaticPatches/D3-Site_4/1_45.h5',
             '/mnt/comp_micro/Projects/CellVAE/Data/StaticPatches/D3-Site_6/3_20.h5',
             '/mnt/comp_micro/Projects/CellVAE/Data/StaticPatches/D4-Site_7/50_34.h5',
             '/mnt/comp_micro/Projects/CellVAE/Data/StaticPatches/D5-Site_8/47_14.h5']

for i, f in enumerate(sample_fs):
  sample_ind = fs_.index(f)
  sample = dataset[sample_ind:(sample_ind+1)][0].cuda()
  output = model(sample)[0]
  inp = sample.cpu().data.numpy()
  out = output.cpu().data.numpy()
  input_phase = (inp[0, 0] * 65535).astype('uint16')
  output_phase = (out[0, 0] * 65535).astype('uint16')
  input_retardance = (inp[0, 1] * 65535).astype('uint16')
  output_retardance = (out[0, 1] * 65535).astype('uint16')
  cv2.imwrite('/home/michaelwu/fig3_VAE_pair%d_input_phase.png' % i, enhance_contrast(input_phase, 1., -10000)) # Note dataset has been rescaled
  cv2.imwrite('/home/michaelwu/fig3_VAE_pair%d_output_phase.png' % i, enhance_contrast(output_phase, 1., -10000))
  cv2.imwrite('/home/michaelwu/fig3_VAE_pair%d_input_retardance.png' % i, enhance_contrast(input_retardance, 2., 0.))
  cv2.imwrite('/home/michaelwu/fig3_VAE_pair%d_output_retardance.png' % i, enhance_contrast(output_retardance, 2., 0.))

##########

# Fig 3 B(PCA) & C
# PCA on VAE latent space
z_bs = {}
z_as = {}
for i in range(len(dataset)):
  sample = dataset[i:(i+1)][0].cuda()
  z_b = model.enc(sample)
  z_a, _, _ = model.vq(z_b)
  f_n = fs_[i]
  z_as[f_n] = z_a.cpu().data.numpy()
  z_bs[f_n] = z_b.cpu().data.numpy()
dats = np.stack([z_bs[f] for f in fs], 0).reshape((len(dataset), -1))
pca = PCA(0.5)
dats_ = pca.fit_transform(dats)
with open('./save_0005_bkp4_latent_space_PCAed.pkl', 'wb') as f:
  pickle.dump(dats_, f)
trajs = pickle.load(open('./HiddenStateExtractor/trajectory_in_inds.pkl', 'rb'))
sizes = pickle.load(open(DATA_ROOT + '/Data/EncodedSizes.pkl', 'rb'))
ss = [sizes[f][0] for f in fs]

cmap = matplotlib.cm.get_cmap('BuPu')  
range_min = np.log(min(ss))
range_max = np.log(max(ss))
colors = [cmap(((np.log(s) - range_min)/(range_max - range_min))**1.5) for s in ss]

# Supp Fig 6
cum_explained_var_ratio = list(np.cumsum(pca.explained_variance_ratio_))
cum_explained_var_ratio.insert(0, 0)
plt.clf()
plt.plot(np.arange(len(cum_explained_var_ratio)), cum_explained_var_ratio, '.-')
verts = [(0, 0), *zip(np.arange(5), cum_explained_var_ratio[:5]), (4, 0)]
poly = matplotlib.patches.Polygon(verts, facecolor='0.9', edgecolor='0.5')
plt.gca().add_patch(poly)
plt.ylim(0, 0.48)
plt.xlim(-2, 40)
plt.ylabel("(Cumulative) Explained Variance Ratio", fontsize=16)
plt.xlabel("Principle Components", fontsize=16)
plt.savefig('/home/michaelwu/supp_fig6_PCA_explained_variance.eps')
plt.savefig('/home/michaelwu/supp_fig6_PCA_explained_variance.png', dpi=300)

# Supp Fig 7
import umap
reducer = umap.UMAP()
embedding = reducer.fit_transform(dats)
plt.clf()
plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=0.5, edgecolors='none')
plt.xlim(0, 11)
plt.ylim(-6, 7.5)
plt.savefig('/home/michaelwu/supp_fig7_UMAP.eps')
plt.savefig('/home/michaelwu/supp_fig7_UMAP.png', dpi=300)


plt.clf()
sns.set_style('white')
fig, ax = plt.subplots()
ax.scatter(dats_[:, 0], dats_[:, 1], c=colors, s=0.5, edgecolors='none')
rec1 = plt.Rectangle((-2, 0), 6, 2, color=(228/256, 34/256, 86/256, 0.7), fc='none')
rec2 = plt.Rectangle((0, -2), 2, 6, color=(0/256, 137/256, 123/256, 0.7), fc='none')
ax.add_patch(rec1)
ax.add_patch(rec2)

# Supp Video 3
traj_samples = ['D4-Site_0/18', 'D3-Site_7/62', 'D3-Site_2/24', 'D3-Site_0/38']
for t in traj_samples:
  save_traj(t, '/home/michaelwu/supp_video3_sample_traj_%s.gif' % t.replace('/', '_'))
  names = [fs[i] for i in trajs[t]]
  save_movie(names, '/home/michaelwu/supp_video3_sample_traj_movie_%s.gif' % t.replace('/', '_'), masked=False)

selected_frames = [np.array([1, 7, 16, 27, 43]),
                   np.array([1, 7, 12, 16, 21]),
                   np.array([0, 10, 20, 30, 40]),
                   np.array([1, 10, 20, 30, 43])]
cmap2 = matplotlib.cm.get_cmap('tab10')
colors2 = [cmap2.colors[1],
           cmap2.colors[5], 
           (0.15, 0.5, 0.15), 
           (0.2, 0.2, 0.2)]
for ct, (t, inds, c) in enumerate(zip(traj_samples, selected_frames, colors2)):
  order = np.array(trajs[t])
  ax.plot(dats_[order][:, 0], dats_[order][:, 1], '.--', c=c, linewidth=0.5, markersize=0.5)
  ax.plot(dats_[order][inds][:, 0], dats_[order][inds][:, 1], '.', c=c, markersize=2.0)
  for i in range(len(inds) - 1):
    ind0 = inds[i]
    ind1 = inds[i+1]
    ax.arrow(dats_[order[ind0], 0],
             dats_[order[ind0], 1],
             dats_[order[ind1], 0] - dats_[order[ind0], 0],
             dats_[order[ind1], 1] - dats_[order[ind0], 1],
             fc='none',
             ec=c,
             length_includes_head=True,
             head_width=0.2, 
             head_length=0.3)
  names = []
  output_paths = []
  for j, ind in enumerate(order[inds]):
    f = fs[ind]
    names.append(f)
    output_paths.append('/home/michaelwu/fig3_state_transition_sample_%d_%d.png' % (ct, j))
  plot_patches(names, output_paths, masked=False)

plt.xlim(-6, 8)
plt.ylim(-4, 8)
plt.savefig('/home/michaelwu/fig3_morphology_pca.eps')
plt.savefig('/home/michaelwu/fig3_morphology_pca.png', dpi=300)

plt.clf()
fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)
cb1 = matplotlib.colorbar.ColorbarBase(ax, cmap='BuPu',
                                       norm=matplotlib.colors.Normalize(vmin=range_min, vmax=range_max),
                                       orientation='horizontal')
plt.savefig('/home/michaelwu/fig3_morphology_pca_cbar.eps')

##########

# Fig 3 B(patches)
# PC1&2 samples
# bins_PC1 = {(i, i+0.5): [] for i in np.arange(-2, 4, 0.5)}
# bins_PC2 = {(i, i+0.5): [] for i in np.arange(-2, 4, 0.5)}
# for i in range(84884):
#   val0 = dats_[i, 0] 
#   val1 = dats_[i, 1]
#   for b in bins_PC1:
#     if val0 > b[0] and val0 <= b[1] and val1 > 0. and val1 <= 2.:
#       bins_PC1[b].append(fs[i])
#   for b in bins_PC2:
#     if val0 > 0. and val0 <= 1. and val1 > b[0] and val1 <= b[1]:
#       bins_PC2[b].append(fs[i])

# os.mkdir('/home/michaelwu/fig3_PC1')
# for i, b in enumerate(sorted(bins_PC1.keys())):
#   samples = np.random.choice(bins_PC1[b], (5,), replace=False)
#   prefix = 'b%d' % i
#   for s in samples:
#     name = s.split('/')[-2:]
#     name = prefix + '_' + name[0] + '_' + name[1].split('.')[0] + '.png'
#     plot_patch(s, '/home/michaelwu/fig3_PC1/%s' % name)

# os.mkdir('/home/michaelwu/fig3_PC2')
# for i, b in enumerate(sorted(bins_PC2.keys())):
#   samples = np.random.choice(bins_PC2[b], (5,), replace=False)
#   prefix = 'b%d' % i
#   for s in samples:
#     name = s.split('/')[-2:]
#     name = prefix + '_' + name[0] + '_' + name[1].split('.')[0] + '.png'
#     plot_patch(s, '/home/michaelwu/fig3_PC2/%s' % name)

sample_PC1s = [
    '/mnt/comp_micro/Projects/CellVAE/Data/StaticPatches/D5-Site_2/43_57.h5',
    '/mnt/comp_micro/Projects/CellVAE/Data/StaticPatches/D3-Site_5/19_67.h5',
    '/mnt/comp_micro/Projects/CellVAE/Data/StaticPatches/D3-Site_6/51_55.h5',
    '/mnt/comp_micro/Projects/CellVAE/Data/StaticPatches/D4-Site_0/29_25.h5',
    '/mnt/comp_micro/Projects/CellVAE/Data/StaticPatches/D5-Site_1/3_15.h5'
]
sample_PC2s = [
    '/mnt/comp_micro/Projects/CellVAE/Data/StaticPatches/D5-Site_7/48_28.h5',
    '/mnt/comp_micro/Projects/CellVAE/Data/StaticPatches/D4-Site_5/19_21.h5',
    '/mnt/comp_micro/Projects/CellVAE/Data/StaticPatches/D5-Site_1/20_24.h5',
    '/mnt/comp_micro/Projects/CellVAE/Data/StaticPatches/D4-Site_7/28_14.h5',
    '/mnt/comp_micro/Projects/CellVAE/Data/StaticPatches/D5-Site_4/19_89.h5'
]

plot_patches(sample_PC1s, ['/home/michaelwu/fig3_samples_PC1_%d.png' % i for i in range(len(sample_PC1s))])
plot_patches(sample_PC2s, ['/home/michaelwu/fig3_samples_PC2_%d.png' % i for i in range(len(sample_PC2s))])

##########

# Supp Fig 2
# Scatter plot between PC1 and size
sizes = pickle.load(open('/mnt/comp_micro/Projects/CellVAE/Data/EncodedSizes.pkl', 'rb'))
densities = pickle.load(open('/mnt/comp_micro/Projects/CellVAE/Data/EncodedDensities.pkl', 'rb'))
ss = np.log(np.array([sizes[f][0] for f in fs]))
ds = np.array([densities[f][0][2] for f in fs])
PC1s = dats_[:, 0]
PC2s = dats_[:, 1]
df = pd.DataFrame({'PC1': PC1s,
                   'PC2': PC2s,
                   'Size': ss,
                   'Peak Phase': ds})

sns.set_style('white')
bins_y = np.linspace(6, 9.3, 20)
bins_x = np.linspace(-5, 5, 20)
plt.clf()
g = sns.JointGrid(x='PC1', y='Size', data=df, ylim=(6, 9.3), xlim=(-5, 5))
_ = g.ax_marg_x.hist(df['PC1'], bins=bins_x, color=matplotlib.cm.get_cmap('Blues')(0.5))
_ = g.ax_marg_y.hist(df['Size'], bins=bins_y, orientation='horizontal', color=matplotlib.cm.get_cmap('Blues')(0.5))
g.plot_joint(sns.kdeplot, cmap="Blues", shade=True)
y_ticks = np.array([500, 1000, 2000, 4000, 8000])
g.ax_joint.set_yticks(np.log(y_ticks))
g.ax_joint.set_yticklabels(y_ticks)
g.set_axis_labels('PC1', 'Size', fontsize=16)
plt.tight_layout()
plt.savefig('/home/michaelwu/supp_fig2_PC1_size.eps')
plt.savefig('/home/michaelwu/supp_fig2_PC1_size.png', dpi=300)

sns.set_style('white')
bins_y = np.linspace(0.52, 0.75, 20)
bins_x = np.linspace(-3, 4, 20)
plt.clf()
g = sns.JointGrid(x='PC2', y='Peak Phase', data=df, ylim=(0.52, 0.75), xlim=(-3, 4))
_ = g.ax_marg_x.hist(df['PC2'], bins=bins_x, color=matplotlib.cm.get_cmap('Reds')(0.5))
_ = g.ax_marg_y.hist(df['Peak Phase'], bins=bins_y, orientation='horizontal', color=matplotlib.cm.get_cmap('Reds')(0.5))
g.plot_joint(sns.kdeplot, cmap="Reds", shade=True)
g.set_axis_labels('PC2', 'Peak Phase', fontsize=16)
plt.tight_layout()
plt.savefig('/home/michaelwu/supp_fig2_PC2_density.eps')
plt.savefig('/home/michaelwu/supp_fig2_PC2_density.png', dpi=300)

##########

# Supp Fig 3
# Samples along first 4 PCs
names = []
out_paths = []
np.random.seed(123)

PC1s = dats_[:, 0]
lower_ = np.quantile(PC1s, 0.2)
lower_fs = [f for i, f in enumerate(fs) if PC1s[i] < lower_]
upper_ = np.quantile(PC1s, 0.8)
upper_fs = [f for i, f in enumerate(fs) if PC1s[i] > upper_]
for i, f in enumerate(np.random.choice(lower_fs, (10,), replace=False)):
  names.append(f)
  out_paths.append('/home/michaelwu/supp_fig3_PC1_lower_sample%d.png' % i)
for i, f in enumerate(np.random.choice(upper_fs, (10,), replace=False)):
  names.append(f)
  out_paths.append('/home/michaelwu/supp_fig3_PC1_upper_sample%d.png' % i)

PC2s = dats_[:, 1]
lower_ = np.quantile(PC2s, 0.2)
lower_fs = [f for i, f in enumerate(fs) if PC2s[i] < lower_]
upper_ = np.quantile(PC2s, 0.8)
upper_fs = [f for i, f in enumerate(fs) if PC2s[i] > upper_]
for i, f in enumerate(np.random.choice(lower_fs, (10,), replace=False)):
  names.append(f)
  out_paths.append('/home/michaelwu/supp_fig3_PC2_lower_sample%d.png' % i)
for i, f in enumerate(np.random.choice(upper_fs, (10,), replace=False)):
  names.append(f)
  out_paths.append('/home/michaelwu/supp_fig3_PC2_upper_sample%d.png' % i)

PC1_range = (np.quantile(PC1s, 0.4), np.quantile(PC1s, 0.6))
PC2_range = (np.quantile(PC2s, 0.4), np.quantile(PC2s, 0.6))
PC3s = dats_[:, 2]
lower_ = np.quantile(PC3s, 0.2)
lower_fs = [f for i, f in enumerate(fs) if PC3s[i] < lower_ and PC1_range[0] < PC1s[i] < PC1_range[1] and PC2_range[0] < PC2s[i] < PC2_range[1]]
upper_ = np.quantile(PC3s, 0.8)
upper_fs = [f for i, f in enumerate(fs) if PC3s[i] > upper_ and PC1_range[0] < PC1s[i] < PC1_range[1] and PC2_range[0] < PC2s[i] < PC2_range[1]]
for i, f in enumerate(np.random.choice(lower_fs, (10,), replace=False)):
  names.append(f)
  out_paths.append('/home/michaelwu/supp_fig3_PC3_lower_sample%d.png' % i)
for i, f in enumerate(np.random.choice(upper_fs, (10,), replace=False)):
  names.append(f)
  out_paths.append('/home/michaelwu/supp_fig3_PC3_upper_sample%d.png' % i)

PC4s = dats_[:, 3]
lower_ = np.quantile(PC4s, 0.2)
lower_fs = [f for i, f in enumerate(fs) if PC4s[i] < lower_ and PC1_range[0] < PC1s[i] < PC1_range[1] and PC2_range[0] < PC2s[i] < PC2_range[1]]
upper_ = np.quantile(PC4s, 0.8)
upper_fs = [f for i, f in enumerate(fs) if PC4s[i] > upper_ and PC1_range[0] < PC1s[i] < PC1_range[1] and PC2_range[0] < PC2s[i] < PC2_range[1]]
for i, f in enumerate(np.random.choice(lower_fs, (10,), replace=False)):
  names.append(f)
  out_paths.append('/home/michaelwu/supp_fig3_PC4_lower_sample%d.png' % i)
for i, f in enumerate(np.random.choice(upper_fs, (10,), replace=False)):
  names.append(f)
  out_paths.append('/home/michaelwu/supp_fig3_PC4_upper_sample%d.png' % i)

plot_patches(names, out_paths)

np.random.seed(123)
names = []
out_paths = []
# dats = pickle.load(open('./save_0005_bkp4.pkl', 'rb'))
# sizes = pickle.load(open('/mnt/comp_micro/Projects/CellVAE/Data/EncodedSizes.pkl', 'rb'))
# densities = pickle.load(open('/mnt/comp_micro/Projects/CellVAE/Data/EncodedDensities.pkl', 'rb'))
# aps_nr = pickle.load(open('/mnt/comp_micro/Projects/CellVAE/Data/EncodedAspectRatios_NoRotation.pkl', 'rb'))
# aps = pickle.load(open('/mnt/comp_micro/Projects/CellVAE/Data/EncodedAspectRatios.pkl', 'rb'))
# angle_array = []
# for f in fs:
#   if aps[f][2] >= 0:
#     angle_array.append(aps[f][2] - 90)
#   elif 0.8 < aps[f][0]/aps[f][1] < 1.25:
#     angle_array.append(-90)
#   else:
#     angle_array.append(aps[f][2])
# Properties = [[np.log(sizes[f][0]) for f in fs],
#               [densities[f][0][2] for f in fs],
#               [densities[f][1][2] for f in fs],
#               [aps_nr[f][0]/aps_nr[f][1] for f in fs],
#               angle_array,
#               [aps[f][0]/aps[f][1] for f in fs]]
# X = np.stack(Properties, 1)
# X = sm.add_constant(X)
# dats_residues = []
# for i in range(dats.shape[1]):
#   y = dats[:, i]
#   model = sm.OLS(y, X)
#   results = model.fit()
#   residue = y - results.predict(X)
#   dats_residues.append(residue)
# dats_residues = np.stack(dats_residues, 1)
dats_residues = pickle.load(open('./save_0005_bkp4_residues.pkl', 'rb'))
pca_r = PCA(3)
dats_residues_ = pca_r.fit_transform(dats_residues)
rPC1s = dats_residues_[:, 0]
lower_ = np.quantile(rPC1s, 0.2)
lower_fs = [f for i, f in enumerate(fs) if rPC1s[i] < lower_ and PC1_range[0] < PC1s[i] < PC1_range[1] and PC2_range[0] < PC2s[i] < PC2_range[1]]
upper_ = np.quantile(rPC1s, 0.8)
upper_fs = [f for i, f in enumerate(fs) if rPC1s[i] > upper_ and PC1_range[0] < PC1s[i] < PC1_range[1] and PC2_range[0] < PC2s[i] < PC2_range[1]]
for i, f in enumerate(np.random.choice(lower_fs, (20,), replace=False)):
  names.append(f)
  out_paths.append('/home/michaelwu/supp_fig3_rPC1_lower_sample%d.png' % i)
for i, f in enumerate(np.random.choice(upper_fs, (20,), replace=False)):
  names.append(f)
  out_paths.append('/home/michaelwu/supp_fig3_rPC1_upper_sample%d.png' % i)
plot_patches(names, out_paths, masked=False)

##########

# Supp Fig 4
# Correlation between PC1~6, size, density, aspect ratio, etc.

sizes = pickle.load(open('/mnt/comp_micro/Projects/CellVAE/Data/EncodedSizes.pkl', 'rb'))
densities = pickle.load(open('/mnt/comp_micro/Projects/CellVAE/Data/EncodedDensities.pkl', 'rb'))
aps_nr = pickle.load(open('/mnt/comp_micro/Projects/CellVAE/Data/EncodedAspectRatios_NoRotation.pkl', 'rb'))
aps = pickle.load(open('/mnt/comp_micro/Projects/CellVAE/Data/EncodedAspectRatios.pkl', 'rb'))

angle_array = []
for f in fs:
  if aps[f][2] >= 0:
    angle_array.append(aps[f][2] - 90)
  elif 0.8 < aps[f][0]/aps[f][1] < 1.25:
    angle_array.append(-90)
  else:
    angle_array.append(aps[f][2])
PCs = [PC1s, PC2s, PC3s, PC4s, dats_[:, 4], dats_[:, 5]]
Properties = [[np.log(sizes[f][0]) for f in fs],
              [densities[f][0][2] for f in fs],
              [densities[f][1][2] for f in fs],
              [aps_nr[f][0]/aps_nr[f][1] for f in fs],
              angle_array,
              [aps[f][0]/aps[f][1] for f in fs]]

sr_mat = np.zeros((len(PCs), len(Properties)))
pr_mat = np.zeros((len(PCs), len(Properties)))
for i, PC in enumerate(PCs):
  for j, prop in enumerate(Properties):
    sr_mat[i, j] = spearmanr(PC, prop).correlation
    pr_mat[i, j] = pearsonr(PC, prop)[0]

plt.clf()
fig, ax = plt.subplots()
cmap = matplotlib.cm.get_cmap('RdBu')
im = ax.imshow(np.transpose(sr_mat), cmap=cmap, vmin=-1.5, vmax=1.5)

ax.set_xticks(np.arange(len(PCs)))
ax.set_yticks(np.arange(len(Properties)))
ax.set_xticklabels(['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'])
ax.set_yticklabels(['Size', 'Peak Phase', 'Peak Retardance', 'Aspect Ratio (y-axis)', 'Aspect Ratio', 'Angle (Long axis)'])
for i in range(len(PCs)):
  for j in range(len(Properties)):
    text = ax.text(i, j, "%.2f" % sr_mat[i, j], ha="center", va="center", color="k")
plt.tight_layout()
plt.savefig('/home/michaelwu/supp_fig4_correlations.eps')
plt.savefig('/home/michaelwu/supp_fig4_correlations.png', dpi=300)

##########

# Supp Fig 5
# Distributional difference between trajectories and non-trajectories
traj_PC1_diffs = []
base_diffs = []
for t in trajs:
  traj_PC1 = dats_[np.array(trajs[t])][:, 0]
  traj_PC1_diff = np.abs(traj_PC1[1:] - traj_PC1[:-1])
  traj_PC1_diffs.append(traj_PC1_diff)
  random_PC1 = dats_[np.random.choice(np.arange(dats_.shape[0]), (len(trajs[t]),), replace=False), 0]
  base_diffs.append(np.abs(random_PC1[1:] - random_PC1[:-1]))
traj_PC1_diffs = np.concatenate(traj_PC1_diffs)
base_diffs = np.concatenate(base_diffs)
plt.clf()
plt.hist(traj_PC1_diffs, bins=np.arange(0, 8, 0.2), normed=True, color=(1, 0, 0, 0.5), label='Trajectories')
plt.hist(base_diffs, bins=np.arange(0, 8, 0.2), normed=True, color=(0, 0, 1, 0.5), label='Random pairs')
plt.legend(fontsize=16)
plt.xlabel('PC1 diff', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.savefig('/home/michaelwu/supp_fig5_distri_PC1.eps')
plt.savefig('/home/michaelwu/supp_fig5_distri_PC1.png', dpi=300)

traj_PC2_diffs = []
base_diffs = []
for t in trajs:
  traj_PC2 = dats_[np.array(trajs[t])][:, 1]
  traj_PC2_diff = np.abs(traj_PC2[1:] - traj_PC2[:-1])
  traj_PC2_diffs.append(traj_PC2_diff)
  random_PC2 = dats_[np.random.choice(np.arange(dats_.shape[0]), (len(trajs[t]),), replace=False), 1]
  base_diffs.append(np.abs(random_PC2[1:] - random_PC2[:-1]))
traj_PC2_diffs = np.concatenate(traj_PC2_diffs)
base_diffs = np.concatenate(base_diffs)
plt.clf()
plt.hist(traj_PC2_diffs, bins=np.arange(0, 8, 0.2), normed=True, color=(1, 0, 0, 0.5), label='Trajectories')
plt.hist(base_diffs, bins=np.arange(0, 8, 0.2), normed=True, color=(0, 0, 1, 0.5), label='Random pairs')
plt.legend(fontsize=16)
plt.xlabel('PC2 diff', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.savefig('/home/michaelwu/supp_fig5_distri_PC2.eps')
plt.savefig('/home/michaelwu/supp_fig5_distri_PC2.png', dpi=300)

############################################################################################################

# Fig 4 A
# KDE plot of PC1/speed
feat = 'save_0005_before'
dataset = torch.load('StaticPatchesAll.pt')
fs_ = pickle.load(open('./HiddenStateExtractor/file_paths_bkp.pkl', 'rb'))
fs = sorted(pickle.load(open('./HiddenStateExtractor/file_paths_bkp.pkl', 'rb')))
trajs = pickle.load(open('./HiddenStateExtractor/trajectory_in_inds.pkl', 'rb'))
dats_ = pickle.load(open('./save_0005_bkp4_latent_space_PCAed.pkl', 'rb'))
sizes = pickle.load(open(DATA_ROOT + '/Data/EncodedSizes.pkl', 'rb'))

all_mg_trajs = {}
all_mg_trajs_positions = {}
for site in sites:
  mg_trajectories_inds, mg_trajectories_positions = pickle.load(open(DATA_ROOT + '/Data/DynamicPatches/%s/mg_traj.pkl' % site, 'rb'))
  for i, traj in enumerate(mg_trajectories_positions):
    all_mg_trajs[site + '/%d' % i] = mg_trajectories_inds[i]
    all_mg_trajs_positions[site + '/%d' % i] = traj

traj_average_moving_distances = {}
traj_cell_sizes_mean = {}
traj_PC1 = {}
traj_PC2 = {}
for t in all_mg_trajs:
  t_keys = sorted(all_mg_trajs[t].keys())
  dists = []
  for t_point in range(len(t_keys) - 1):
    d = np.linalg.norm(all_mg_trajs_positions[t][t_keys[t_point+1]] - \
                       all_mg_trajs_positions[t][t_keys[t_point]], ord=2)
    dists.append(d)
  traj_average_moving_distances[t] = np.mean(dists)
  traj_sizes = [sizes[fs[ind]][0] for ind in trajs[t]]
  traj_cell_sizes_mean[t] = np.mean(traj_sizes)
  pc1s = [dats_[ind, 0] for ind in trajs[t]]
  pc2s = [dats_[ind, 1] for ind in trajs[t]]
  traj_PC1[t] = np.mean(pc1s)
  traj_PC2[t] = np.mean(pc2s)

t_arrays = sorted(all_mg_trajs.keys())
df = pd.DataFrame({'PC1': [traj_PC1[t] for t in t_arrays],
                   'PC2': [traj_PC2[t] for t in t_arrays],
                   'sizes': [traj_cell_sizes_mean[t] for t in t_arrays],
                   'dists': [np.log(traj_average_moving_distances[t] * 0.72222) for t in t_arrays]}) #0.72um/h for 1pixel/27min

sns.set_style('white')
bins_y = np.linspace(0.1, 4.3, 20)
bins_x = np.linspace(-4, 4, 20)
plt.clf()
g = sns.JointGrid(x='PC1', y='dists', data=df, ylim=(0.1, 4.3), xlim=(-4, 4))
_ = g.ax_marg_x.hist(df['PC1'], bins=bins_x)
_ = g.ax_marg_y.hist(df['dists'], bins=bins_y, orientation='horizontal')
g.plot_joint(sns.kdeplot, cmap="Blues", shade=True)
y_ticks = np.array([1.5, 3., 6., 12., 24., 48.])
g.ax_joint.set_yticks(np.log(y_ticks))
g.ax_joint.set_yticklabels(y_ticks)
g.set_axis_labels('', '')
plt.savefig('/home/michaelwu/fig4_correlation_kde.eps')
plt.savefig('/home/michaelwu/fig4_correlation_kde.png', dpi=300)

##########

# Fig 4 B
# Sample traj
traj_represented = ['D4-Site_8/16', 'D4-Site_1/14', 'D4-Site_0/15', 
                    'D4-Site_5/1', 'D3-Site_3/56', 'D5-Site_4/33']
colors = [(53, 52, 205)] * 3 + [(176, 177, 0)] * 3
for t, c in zip(traj_represented, colors):
  traj = all_mg_trajs[t]
  frame0_name = fs[trajs[t][0]]
  site_name = frame0_name.split('/')[-2]
  site_dat = pickle.load(open('../data_temp/%s_all_patches.pkl' % site_name, 'rb'))
  frame0 = site_dat[frame0_name]["masked_mat"][:, :, 0]
  frame0 = np.clip(enhance_contrast(frame0, phase_a, phase_b), 0, 65535)
  mat = np.zeros((frame0.shape[0], frame0.shape[1], 3), dtype='uint8')
  mat[:, :] = (np.expand_dims(frame0, 2) / 256).astype('uint8')
  try:
    traj_positions = all_mg_trajs_positions[t]
    positions = np.stack([traj_positions[k] for k in sorted(traj.keys())])
    center_position = positions[0] - np.array([128, 128])
    for i in range(positions.shape[0] - 1):
      start = positions[i] - center_position
      end = positions[i + 1] - center_position
      mat = cv2.line(mat, (start[1], start[0]), (end[1], end[0]), c, thickness=2)
    cv2.imwrite('/home/michaelwu/fig4_sample_%s.png' % t.replace('/', '_'), mat)
  except Exception as e:
    print(e)

##########

# Supp Video 4
# Large/small trajectories
for t in traj_represented:
  save_traj(t, '/home/michaelwu/supp_video4_sample_traj_%s.gif' % t.replace('/', '_'))
  names = [fs[i] for i in trajs[t]]
  save_movie(names, '/home/michaelwu/supp_video4_sample_traj_movie_%s.gif' % t.replace('/', '_'), masked=False)

##########

# Fig 4 C
# Violin plot of two modes
small_trajs = []
large_trajs = []
for t in trajs:
  traj_dats_ = dats_[np.array(trajs[t])]
  if np.quantile(traj_dats_[:, 0], 0.7) < -0.8 and \
     np.quantile(traj_dats_[:, 1], 0.3) < 0 and len(traj_dats_) > 20:
    small_trajs.append(t)
  if np.quantile(traj_dats_[:, 0], 0.3) > 0.8 and len(traj_dats_) > 20:
    large_trajs.append(t)

df = pd.DataFrame({'cluster': ['Small'] * len(small_trajs) + ['Large'] * len(large_trajs), 
                   'aver_dist': [np.log(traj_average_moving_distances[t] * 0.72222) for t in small_trajs + large_trajs]})
plt.clf()
sns.set_style('whitegrid')
g = sns.violinplot(x='cluster', 
                   y='aver_dist', 
                   data=df, 
                   order=['Small', 'Large'], 
                   palette={'Small': '#cd3435', 'Large': '#00b1b0'},
                   orient='v')
g.set_ylim(0.1, 4.3)
y_ticks = np.array([1.5, 3., 6., 12., 24., 48.])
g.set_yticks(np.log(y_ticks))
g.set_yticklabels(y_ticks)
g.set_xticklabels(['', ''])
g.set_xlabel('')
g.set_ylabel('')
plt.savefig('/home/michaelwu/fig4_aver_dist.eps')
plt.savefig('/home/michaelwu/fig4_aver_dist.png', dpi=300)

##########

# Fig 4 D
# MSD plot of two modes
MSD_length = 20

small_traj_ensembles = []
for t in small_trajs:
  t_end = max(all_mg_trajs_positions[t].keys()) + 1
  for t_start in range(t_end - 20):
    if t_start in all_mg_trajs_positions[t]:
      s_traj = {(t_now - t_start): all_mg_trajs_positions[t][t_now] \
          for t_now in range(t_start, t_start+20) if t_now in all_mg_trajs_positions[t]}
      small_traj_ensembles.append(s_traj)

large_traj_ensembles = []
for t in large_trajs:
  t_end = max(all_mg_trajs_positions[t].keys())
  for t_start in range(t_end - 20):
    if t_start in all_mg_trajs_positions[t]:
      l_traj = {(t_now - t_start): all_mg_trajs_positions[t][t_now] \
          for t_now in range(t_start, t_start+20) if t_now in all_mg_trajs_positions[t]}
      large_traj_ensembles.append(l_traj)

small_traj_MSDs = {}
large_traj_MSDs = {}
small_traj_MSDs_trimmed = {}
large_traj_MSDs_trimmed = {}
for i in range(20):
  s_dists = [np.square(t[i] - t[0]).sum() for t in small_traj_ensembles if i in t]
  l_dists = [np.square(t[i] - t[0]).sum() for t in large_traj_ensembles if i in t]
  small_traj_MSDs[i] = s_dists
  large_traj_MSDs[i] = l_dists
  small_traj_MSDs_trimmed[i] = scipy.stats.trimboth(s_dists, 0.25)
  large_traj_MSDs_trimmed[i] = scipy.stats.trimboth(l_dists, 0.25)

def forceAspect(ax,aspect=1):
  im = ax.get_images()
  extent =  im[0].get_extent()
  ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

x = np.arange(1, 20)
y_bins = np.arange(0.9, 11.7, 0.6) # log scale
density_map = np.zeros((20, len(y_bins) - 1))
y = []
for i in range(1, 20):
  for d in small_traj_MSDs[i]:
    if d == 0: 
      continue
    ind_bin = ((np.log(d) - y_bins) > 0).sum() - 1
    if ind_bin < density_map.shape[1] and ind_bin >= 0:
      density_map[i][ind_bin] += 1
  y.append((np.log(np.mean(small_traj_MSDs[i])) - 0.9)/(y_bins[1] - y_bins[0]))
density_map = density_map/density_map.sum(1, keepdims=True)

sns.set_style('white')
plt.clf()
fig = plt.figure()
ax = fig.add_subplot(121)
ax.imshow(np.transpose(density_map), cmap='Reds', origin='lower', vmin=0.01, vmax=0.3, alpha=0.5)
ax.plot(x, np.array(y) - 0.5, '.-', c='#ba4748') # -0.5 is the adjustment for imshow
ax.set_xscale('log')
xticks = np.array([0.5, 1, 2, 4, 8])
xticks_positions = xticks / (27/60)
ax.set_xticks(xticks_positions)
ax.set_xticklabels(xticks)
ax.xaxis.set_minor_locator(NullLocator())
yticks = np.array([0.5, 2, 8, 32, 128, 512, 2048])
yticks_positions = (np.log(yticks / (0.325 * 0.325)) - 0.9)/(y_bins[1] - y_bins[0]) - 0.5 # same adjustment for imshow
ax.set_yticks(yticks_positions)
ax.set_yticklabels(yticks)

density_map = np.zeros((20, len(y_bins) - 1))
y = []
for i in range(1, 20):
  for d in large_traj_MSDs[i]:
    if d == 0: 
      continue
    ind_bin = ((np.log(d) - y_bins) > 0).sum() - 1
    if ind_bin < density_map.shape[1] and ind_bin >= 0:
      density_map[i][ind_bin] += 1
  y.append((np.log(np.mean(large_traj_MSDs[i])) - 0.9)/(y_bins[1] - y_bins[0]))
density_map = density_map/density_map.sum(1, keepdims=True)

ax2 = fig.add_subplot(122)
ax2.imshow(np.transpose(density_map), cmap='BuGn', origin='lower', vmax=0.2, alpha=0.5)
ax2.plot(x, np.array(y) - 0.5, '.-', c='#0b6b6a')
ax2.set_xscale('log')
ax2.set_xticks(xticks_positions)
ax2.set_xticklabels(xticks)
ax2.xaxis.set_minor_locator(NullLocator())
ax2.set_yticks(yticks_positions)
ax2.set_yticklabels(yticks)
plt.tight_layout()
fig.savefig('/home/michaelwu/fig4_MSD.eps')
fig.savefig('/home/michaelwu/fig4_MSD.png', dpi=300)
