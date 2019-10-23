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
import h5py
import pandas as pd
from NNsegmentation.models import Segment
from NNsegmentation.data import predict_whole_map
from SingleCellPatch.extract_patches import instance_clustering, within_range
from SingleCellPatch.generate_trajectories import frame_matching
import matplotlib
from matplotlib import cm
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
from HiddenStateExtractor.vq_vae import VQ_VAE, CHANNEL_MAX, CHANNEL_VAR, prepare_dataset
from HiddenStateExtractor.naive_imagenet import read_file_path, DATA_ROOT
from HiddenStateExtractor.morphology_clustering import select_clean_trajecteories, Kmean_on_short_trajs
from HiddenStateExtractor.movement_clustering import save_traj
import statsmodels.api as sm

RAW_DATA_PATH = '/mnt/comp_micro/Projects/CellVAE/Combined'
color_mg = np.array([240, 94, 56], dtype='uint8')
color_nonmg = np.array([66, 101, 251], dtype='uint8')
color_bg = np.array([150, 150, 150], dtype='uint8')
color_fg = (color_mg * 0.7 + color_nonmg * 0.3).astype('uint8')
sites = ['D%d-Site_%d' % (i, j) for j in range(9) for i in range(3, 6)]

def enhance_contrast(mat, a=1.5, b=-10000):
  mat2 = cv2.addWeighted(mat, 1.5, mat, 0, -10000)
  return mat2

def plot_patch(sample_path, out_path, boundary=False, channel=0):
  with h5py.File(sample_path, 'r') as f:
    mat = np.array(f['masked_mat'][:, :, channel].astype('uint16'))
    mask = np.array(f['masked_mat'][:, :, 2].astype('uint16'))
  mat2 = enhance_contrast(mat, 1.5, -10000)
  cv2.imwrite(out_path, mat2)

############################################################################################################

# Raw input (phase channel)
raw_input_stack = np.load(RAW_DATA_PATH + '/D5-Site_0.npy')
raw_input = raw_input_stack[0, :, :, 0:1]
cv2.imwrite('/home/michaelwu/fig2_raw.png', raw_input)
# raw_movie = [cv2.resize(slic[:, :, 0], (512, 512)) for slic in raw_input_stack]
# imageio.mimsave('/home/michaelwu/sample_movie.gif', np.stack(raw_movie, 0))

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

# Random Forest prediction
RF_predictions_stack = np.load(RAW_DATA_PATH + '/D5-Site_0_RFProbabilities.npy')
RF_predictions = RF_predictions_stack[0]
RF_bg = RF_predictions[:, :, 0:1]
RF_fg = RF_predictions[:, :, 1:2] + RF_predictions[:, :, 2:3]
mat = np.zeros((raw_input.shape[0], raw_input.shape[1], 3), dtype='uint8')
mat[:, :] = (raw_input / 256).astype('uint8')
alpha = 0.7
positions = np.where((RF_bg > 0.5))[:2]
mat[positions] = (1 - alpha) * mat[positions] + alpha * color_bg.reshape((1, 3))
positions = np.where((RF_fg > 0.5))[:2]
mat[positions] = (1 - alpha) * mat[positions] + alpha * color_fg.reshape((1, 3))
cv2.imwrite('/home/michaelwu/fig2_rf_predictions.png', mat)

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

# raw_input_off = raw_input_stack[20, :, :, 0:1]
# RF_predictions_off = RF_predictions_stack[20]
# cv2.imwrite('/home/michaelwu/off_fig2_raw.png', raw_input_off)

# mat = np.zeros((raw_input_off.shape[0], raw_input_off.shape[1], 3), dtype='uint8')
# mat[:, :] = (raw_input_off / 256).astype('uint8')
# alpha = 0.7
# mg_positions = np.where(RF_predictions_off[:, :, 1] > 0.5)[:2]
# nonmg_positions = np.where(RF_predictions_off[:, :, 2] > 0.5)[:2]
# mat = mat * (1 - (alpha * RF_predictions_off[:, :, 0:1])) + np.ones_like(mat) * color_bg.reshape((1, 1, 3)) * (alpha * RF_predictions_off[:, :, 0:1])
# mat[mg_positions] = (mat * (1 - (alpha * RF_predictions_off[:, :, 1:2])) + np.ones_like(mat) * color_mg.reshape((1, 1, 3)) * (alpha * RF_predictions_off[:, :, 1:2]))[mg_positions]
# mat[nonmg_positions] = (mat * (1 - (alpha * RF_predictions_off[:, :, 2:3])) + np.ones_like(mat) * color_nonmg.reshape((1, 1, 3)) * (alpha * RF_predictions_off[:, :, 2:3]))[nonmg_positions]
# cv2.imwrite('/home/michaelwu/off_fig2_rf_predictions_annotation_only.png', mat)

# model = Segment(input_shape=(256, 256, 2), 
#                 unet_feat=32,
#                 fc_layers=[64, 32],
#                 n_classes=3,
#                 model_path='./NNsegmentation/temp_save')
# model.load(model.model_path + '/final.h5')
# NN_predictions_off = predict_whole_map(raw_input_stack[20:21], model, n_supp=20)[0]
# mat = np.zeros((raw_input_off.shape[0], raw_input_off.shape[1], 3), dtype='uint8')
# mat[:, :] = (raw_input_off / 256).astype('uint8')
# alpha = 0.7
# mg_positions = np.where(NN_predictions_off[:, :, 1] > 0.5)[:2]
# nonmg_positions = np.where(NN_predictions_off[:, :, 2] > 0.5)[:2]
# mat = mat * (1 - (alpha * NN_predictions_off[:, :, 0:1])) + np.ones_like(mat) * color_bg.reshape((1, 1, 3)) * (alpha * NN_predictions_off[:, :, 0:1])
# mat[mg_positions] = (mat * (1 - (alpha * NN_predictions_off[:, :, 1:2])) + np.ones_like(mat) * color_mg.reshape((1, 1, 3)) * (alpha * NN_predictions_off[:, :, 1:2]))[mg_positions]
# mat[nonmg_positions] = (mat * (1 - (alpha * NN_predictions_off[:, :, 2:3])) + np.ones_like(mat) * color_nonmg.reshape((1, 1, 3)) * (alpha * NN_predictions_off[:, :, 2:3]))[nonmg_positions]
# cv2.imwrite('/home/michaelwu/off_fig2_nn_predictions.png', mat)

# model.load(model.model_path + '/stage0_0.h5')
# NN_predictions_off2 = predict_whole_map(raw_input_stack[20:21], model, n_supp=20)[0]
# mat = np.zeros((raw_input_off.shape[0], raw_input_off.shape[1], 3), dtype='uint8')
# mat[:, :] = (raw_input_off / 256).astype('uint8')
# alpha = 0.7
# mg_positions = np.where(NN_predictions_off2[:, :, 1] > 0.5)[:2]
# nonmg_positions = np.where(NN_predictions_off2[:, :, 2] > 0.5)[:2]
# mat = mat * (1 - (alpha * NN_predictions_off2[:, :, 0:1])) + np.ones_like(mat) * color_bg.reshape((1, 1, 3)) * (alpha * NN_predictions_off2[:, :, 0:1])
# mat[mg_positions] = (mat * (1 - (alpha * NN_predictions_off2[:, :, 1:2])) + np.ones_like(mat) * color_mg.reshape((1, 1, 3)) * (alpha * NN_predictions_off2[:, :, 1:2]))[mg_positions]
# mat[nonmg_positions] = (mat * (1 - (alpha * NN_predictions_off2[:, :, 2:3])) + np.ones_like(mat) * color_nonmg.reshape((1, 1, 3)) * (alpha * NN_predictions_off2[:, :, 2:3]))[nonmg_positions]
# cv2.imwrite('/home/michaelwu/off_fig2_nn_predictions_annotation_only.png', mat)

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
cv2.imwrite('/home/michaelwu/fig2_nn_predictions_boxed.png', mat)
cv2.imwrite('/home/michaelwu/fig2_nn_predictions_boxed_small.png', mat[:940, :940])

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
  cv2.imwrite('/home/michaelwu/fig2_sample_mg_traj_%d.png' % i, enhance_contrast(mg_mat))
  center = sample_non_mg_traj_positions[i]
  non_mg_mat = mat[center[0]-128:center[0]+128,
                   center[1]-128:center[1]+128]
  cv2.imwrite('/home/michaelwu/fig2_sample_non_mg_traj_%d.png' % i, enhance_contrast(non_mg_mat))


# Sample trajectories
np.random.seed(123)
inds = np.random.choice(np.arange(len(mg_trajectories_positions)), (5,), replace=False)
for i in inds:
  save_traj('D5-Site_0/%d' % i, '/home/michaelwu/sample_traj_%d.gif' % i)
  stacks = []
  for k in sorted(mg_trajectories[i].keys()):
    v = mg_trajectories[i][k]
    with h5py.File(os.path.split(RAW_DATA_PATH)[0] + '/Data/StaticPatches/D5-Site_0/%d_%d.h5' % (k, v), 'r') as f:
      stacks.append(f["masked_mat"][:, :, 0])
  imageio.mimsave('/home/michaelwu/sample_traj_movie_%d.gif' % i, np.stack(stacks, 0))




############################################################################################################

# VAE illustration
cs = [0, 1]
input_shape = (128, 128)
gpu = False
fs = read_file_path(DATA_ROOT + '/Data/StaticPatches')
np.random.seed(1234)
sample_fs = np.random.choice(fs, (4,), replace=False)
dataset = prepare_dataset(sample_fs, cs=cs, input_shape=input_shape, channel_max=CHANNEL_MAX)
model = VQ_VAE(alpha=0.0005, gpu=gpu)
model.load_state_dict(t.load('./HiddenStateExtractor/save_0005.pt', map_location='cpu'))
for i in range(4):
  sample = dataset[i:(i+1)][0]
  output = model(sample)[0]
  inp = sample.data.numpy()
  out = output.data.numpy()
  input_phase = (inp[0, 0] * 65535).astype('uint16')
  output_phase = (out[0, 0] * 65535).astype('uint16')
  input_retardance = (inp[0, 1] * 65535).astype('uint16')
  output_retardance = (out[0, 1] * 65535).astype('uint16')
  cv2.imwrite('/home/michaelwu/fig3_VAE_pair%d_input_phase.png' % i, enhance_contrast(input_phase))
  cv2.imwrite('/home/michaelwu/fig3_VAE_pair%d_output_phase.png' % i, enhance_contrast(output_phase))
  cv2.imwrite('/home/michaelwu/fig3_VAE_pair%d_input_retardance.png' % i, enhance_contrast(input_retardance))
  cv2.imwrite('/home/michaelwu/fig3_VAE_pair%d_output_retardance.png' % i, enhance_contrast(output_retardance))


# PCA on VAE latent space
feat = 'save_0005_before'
fs = sorted(pickle.load(open('./HiddenStateExtractor/file_paths_bkp.pkl', 'rb')))
trajs = pickle.load(open('./HiddenStateExtractor/trajectory_in_inds.pkl', 'rb'))
dats_ = pickle.load(open('./HiddenStateExtractor/%s_PCA.pkl' % feat, 'rb'))
sizes = pickle.load(open(DATA_ROOT + '/Data/EncodedSizes.pkl', 'rb'))
ss = [sizes[f] for f in fs]

cmap = matplotlib.cm.get_cmap('BuPu')  
range_min = np.log(min(ss))
range_max = np.log(max(ss))
colors = [cmap(((np.log(s) - range_min)/(range_max - range_min))**1.5) for s in ss]
plt.clf()
sns.set_style('white')
fig, ax = plt.subplots()
ax.scatter(dats_[:, 0], dats_[:, 1], c=colors, s=0.5, edgecolors='none')

traj_samples = ['D4-Site_0/18', 'D3-Site_7/62', 'D3-Site_2/24', 'D5-Site_7/50']
selected_frames = [np.array([1, 7, 16, 27, 43]),
                   np.array([1, 7, 12, 16, 21]),
                   np.array([0, 10, 20, 30, 40]),
                   np.array([1, 10, 20, 30, 40])]
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
  for j, ind in enumerate(order[inds]):
    plot_patch(fs[ind], '/home/michaelwu/fig3_state_transition_sample_%d_%d.png' % (ct, j))

rec1 = plt.Rectangle((-2, 0), 10, 2, color='#e42256', fc='none')
rec2 = plt.Rectangle((0, -5), 2, 10, color='#00897b', fc='none')
ax.add_patch(rec1)
ax.add_patch(rec2)
plt.savefig('/home/michaelwu/fig3_morphology_pca.eps')

plt.clf()
fig, ax = plt.subplots(figsize=(6, 1))
fig.subplots_adjust(bottom=0.5)
cb1 = matplotlib.colorbar.ColorbarBase(ax, cmap='BuPu',
                                       norm=matplotlib.colors.Normalize(vmin=range_min, vmax=range_max),
                                       orientation='horizontal')
plt.savefig('/home/michaelwu/fig3_morphology_pca_cbar.eps')

# PC1&2 samples
bins_PC1 = {(i, i+0.5): [] for i in np.arange(-2, 8, 0.5)}
bins_PC2 = {(i, i+0.5): [] for i in np.arange(-5, 5, 0.5)}
for i in range(84884):
  val0 = dats_[i, 0] 
  val1 = dats_[i, 1]
  for b in bins_PC1:
    if val0 > b[0] and val0 <= b[1] and val1 > 0. and val1 <= 2.:
      bins_PC1[b].append(fs[i])
  for b in bins_PC2:
    if val0 > 0. and val0 <= 1. and val1 > b[0] and val1 <= b[1]:
      bins_PC2[b].append(fs[i])

sample_PC1s = [
    '/mnt/comp_micro/Projects/CellVAE/Data/StaticPatches/D5-Site_2/39_4.h5',
    '/mnt/comp_micro/Projects/CellVAE/Data/StaticPatches/D4-Site_0/30_47.h5',
    '/mnt/comp_micro/Projects/CellVAE/Data/StaticPatches/D5-Site_0/14_22.h5',
    '/mnt/comp_micro/Projects/CellVAE/Data/StaticPatches/D5-Site_6/13_24.h5',
    '/mnt/comp_micro/Projects/CellVAE/Data/StaticPatches/D3-Site_4/28_44.h5'
]
sample_PC2s = [
    '/mnt/comp_micro/Projects/CellVAE/Data/StaticPatches/D5-Site_8/44_67.h5',
    '/mnt/comp_micro/Projects/CellVAE/Data/StaticPatches/D3-Site_0/47_60.h5',
    '/mnt/comp_micro/Projects/CellVAE/Data/StaticPatches/D5-Site_3/23_0.h5',
    '/mnt/comp_micro/Projects/CellVAE/Data/StaticPatches/D3-Site_4/38_66.h5',
    '/mnt/comp_micro/Projects/CellVAE/Data/StaticPatches/D3-Site_4/45_1.h5'
]

for i, sample in enumerate(sample_PC1s):
  plot_patch(sample, '/home/michaelwu/fig3_samples_PC1_%d.png' % i)
for i, sample in enumerate(sample_PC2s):
  plot_patch(sample, '/home/michaelwu/fig3_samples_PC2_%d.png' % i)

################################################################################################

feat = 'save_0005_before'
fs = sorted(pickle.load(open('./HiddenStateExtractor/file_paths_bkp.pkl', 'rb')))
trajs = pickle.load(open('./HiddenStateExtractor/trajectory_in_inds.pkl', 'rb'))
dats_ = pickle.load(open('./HiddenStateExtractor/%s_PCA.pkl' % feat, 'rb'))
sizes = pickle.load(open(DATA_ROOT + '/Data/EncodedSizes.pkl', 'rb'))
aps = pickle.load(open(DATA_ROOT + '/Data/EncodedAspectRatios.pkl', 'rb'))

all_mg_trajs = {}
all_mg_trajs_positions = {}
for site in sites:
  mg_trajectories_inds, mg_trajectories_positions = pickle.load(open(DATA_ROOT + '/Data/DynamicPatches/%s/mg_traj.pkl' % site, 'rb'))
  for i, traj in enumerate(mg_trajectories_positions):
    all_mg_trajs[site + '/%d' % i] = mg_trajectories_inds[i]
    all_mg_trajs_positions[site + '/%d' % i] = traj

traj_average_moving_distances = {}
traj_aps_mean = {}
traj_cell_sizes_mean = {}
traj_cell_sizes_std = {}
traj_PC1 = {}
traj_PC2 = {}
traj_PC1_std = {}
traj_PC2_std = {}
for t in all_mg_trajs:
  t_keys = sorted(all_mg_trajs[t].keys())
  dists = []
  for t_point in range(len(t_keys) - 1):
    d = np.linalg.norm(all_mg_trajs_positions[t][t_keys[t_point+1]] - \
                       all_mg_trajs_positions[t][t_keys[t_point]], ord=2)
    dists.append(d)
  traj_average_moving_distances[t] = np.mean(dists)

  traj_sizes = [sizes[fs[ind]] for ind in trajs[t]]
  traj_cell_sizes_mean[t] = np.mean(traj_sizes)
  traj_cell_sizes_std[t] = np.std(traj_sizes)
  traj_aps = [min(aps[fs[ind]], 1/aps[fs[ind]]) for ind in trajs[t]]
  traj_aps_mean[t] = np.mean(traj_aps)
  pc1s = [dats_[ind, 0] for ind in trajs[t]]
  pc2s = [dats_[ind, 1] for ind in trajs[t]]
  traj_PC1[t] = np.mean(pc1s)
  traj_PC2[t] = np.mean(pc2s)
  traj_PC1_std[t] = np.std(pc1s)
  traj_PC2_std[t] = np.std(pc2s)

t_arrays = sorted(all_mg_trajs.keys())
df = pd.DataFrame({'PC1': [traj_PC1[t] for t in t_arrays],
                   'PC2': [traj_PC2[t] for t in t_arrays],
                   'sizes': [traj_cell_sizes_mean[t] for t in t_arrays],
                   'aps': [traj_aps_mean[t] for t in t_arrays],
                   'dists': [np.log(traj_average_moving_distances[t] * 0.72222) for t in t_arrays]}) #0.72um/h for 1pixel/27min

sns.set_style('white')
bins_y = np.linspace(0.1, 4.3, 20)
bins_x = np.linspace(-6, 7, 20)
plt.clf()
g = sns.JointGrid(x='PC1', y='dists', data=df, ylim=(0.1, 4.3), xlim=(-6, 7))
_ = g.ax_marg_x.hist(df['PC1'], bins=bins_x)
_ = g.ax_marg_y.hist(df['dists'], bins=bins_y, orientation='horizontal')
g.plot_joint(sns.kdeplot, cmap="Blues", shade=True)
y_ticks = np.array([1.5, 3., 6., 12., 24., 48.])
g.ax_joint.set_yticks(np.log(y_ticks))
g.ax_joint.set_yticklabels(y_ticks)
g.set_axis_labels('', '')
plt.savefig('/home/michaelwu/fig4_correlation_kde.eps')


traj_represented = ['D4-Site_8/16', 'D4-Site_1/14', 'D4-Site_0/15', 
                    'D4-Site_5/1', 'D3-Site_3/56', 'D5-Site_4/33']
colors = [(53, 52, 205)] * 3 + [(176, 177, 0)] * 3
for t, c in zip(traj_represented, colors):
  traj = all_mg_trajs[t]
  with h5py.File(fs[trajs[t][0]], 'r') as f:
    frame0 = np.array(f['masked_mat'][:, :, 0]).astype('uint16')
  frame0 = enhance_contrast(frame0)
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

small_trajs = []
large_trajs = []
for t in trajs:
  traj_dats_ = dats_[np.array(trajs[t])]
  if np.quantile(traj_dats_[:, 0], 0.7) < -0.5 and \
     np.quantile(traj_dats_[:, 1], 0.3) > 0.5 and len(traj_dats_) > 20:
    small_trajs.append(t)
  if np.quantile(traj_dats_[:, 0], 0.3) > 1.8 and len(traj_dats_) > 20:
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



# MSD_length = 20
# def generate_MSD_distri(trajectories_positions, length=MSD_length):
#   MSD = {i * 0.45: [] for i in range(1, length)} # 0.45h for 27min
#   for t in trajectories_positions:
#     for t1 in sorted(t.keys()):
#       for t2 in range(t1+1, min(max(t.keys())+1, t1+length)):
#         if t2 in t:
#           dist = np.linalg.norm(t[t2] - t[t1], ord=2)
#           MSD[(t2-t1) * 0.45].append((dist*0.325)**2) # 0.325um for 1pixel
#   return MSD
# MSD_small = generate_MSD_distri([all_mg_trajs_positions[t] for t in small_trajs])
# MSD_large = generate_MSD_distri([all_mg_trajs_positions[t] for t in large_trajs])


# fit_length = 8
# X = np.log(np.array(sorted(MSD_small.keys()))[:8])
# y1 = [np.log(np.mean(MSD_small[i*0.45])) for i in np.arange(1, fit_length + 1)]
# y2 = [np.log(np.mean(MSD_large[i*0.45])) for i in np.arange(1, fit_length + 1)]
# X = sm.add_constant(X)
# intercept1, slope1 = sm.OLS(y1, X).fit().params
# intercept2, slope2 = sm.OLS(y2, X).fit().params

# df = pd.DataFrame({'t': np.log(np.array(sorted(MSD_small.keys()))),
#                    'MSD (Small)': [np.log(np.mean(MSD_small[i*0.45])) for i in np.arange(1, MSD_length)],
#                    'MSD_std (Small)': [np.log(np.std(MSD_small[i*0.45])) for i in np.arange(1, MSD_length)],
#                    'MSD (Large)': [np.log(np.mean(MSD_large[i*0.45])) for i in np.arange(1, MSD_length)],
#                    'MSD_std (Large)': [np.log(np.std(MSD_large[i*0.45])) for i in np.arange(1, MSD_length)],
#                    'Linear Fit (Small)': np.log(np.array(sorted(MSD_small.keys()))) * slope1 + intercept1,
#                    'Linear Fit (Large)': np.log(np.array(sorted(MSD_small.keys()))) * slope2 + intercept2})
# plt.clf()
# sns.set_style('whitegrid')

# #plt.plot(df['t'], df['MSD (Small)'], '.-', c='#cd3435', linewidth=2, markersize=10, label='MSD (Small)')
# plt.errorbar(df['t'], df['MSD (Small)'], yerr=df['MSD_std (Small)'], c='#cd3435', linewidth=2, markersize=10, label='MSD (Small)')
# plt.plot(df['t'], df['Linear Fit (Small)'], '--', c='#cd3435', linewidth=2)
# plt.plot(df['t'], df['MSD (Large)'], '.-', c='#00b1b0', linewidth=2, markersize=10, label='MSD (Large)')
# #plt.errorbar(df['t'], df['MSD (Large)'], yerr=df['MSD_std (Large)'], c='#00b1b0', linewidth=2, markersize=10, label='MSD (Large)')
# plt.plot(df['t'], df['Linear Fit (Large)'], '--', c='#00b1b0', linewidth=2)

# x_ticks = np.array([0.5, 1., 2., 4, 8])
# plt.gca().set_xticks(np.log(x_ticks))
# plt.gca().set_xticklabels(x_ticks)

# y_ticks = np.array([8, 16, 32, 64, 128, 256, 512])
# plt.gca().set_yticks(np.log(y_ticks))
# plt.gca().set_yticklabels(y_ticks)

# plt.legend()
# plt.savefig('/home/michaelwu/fig4_MSD.eps')

MSD_length = 20

# small_traj_ensembles = []
# for t in small_trajs:
#   t_start = min(all_mg_trajs_positions[t].keys()) + 1
#   t_end = max(all_mg_trajs_positions[t].keys())
#   while (t_start + 20 <= t_end):
#     s_traj = {(t_now - t_start): all_mg_trajs_positions[t][t_now] \
#         for t_now in range(t_start, t_start+20) if t_now in all_mg_trajs_positions[t]}
#     small_traj_ensembles.append(s_traj)
#     for t_start in range(max(s_traj.keys()) + t_start + 1, t_end + 1):
#       if t_start in all_mg_trajs_positions[t]:
#         break

# large_traj_ensembles = []
# for t in large_trajs:
#   t_start = min(all_mg_trajs_positions[t].keys()) + 1
#   t_end = max(all_mg_trajs_positions[t].keys())
#   while (t_start + 20 <= t_end):
#     l_traj = {(t_now - t_start): all_mg_trajs_positions[t][t_now] \
#         for t_now in range(t_start, t_start+20) if t_now in all_mg_trajs_positions[t]}
#     large_traj_ensembles.append(l_traj)
#     for t_start in range(max(l_traj.keys()) + t_start + 1, t_end + 1):
#       if t_start in all_mg_trajs_positions[t]:
#         break


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

x = np.arange(20)
y_bins = np.linspace(0, np.quantile(small_traj_MSDs[19], 0.95), 41)
density_map = np.zeros((20, 40))
y = []
for i in range(20):
  for d in small_traj_MSDs[i]:
    ind_bin = 40 - (y_bins > d).sum()
    if ind_bin < 40:
      density_map[i][ind_bin] += 1
  y.append(np.mean(small_traj_MSDs[i])/(y_bins[1] - y_bins[0]))

density_map = density_map/density_map.sum(1, keepdims=True)
plt.clf()
plt.imshow(np.transpose(np.log(density_map + 1e-5)), cmap='Reds', vmin=-7.5, origin='lower')
plt.plot(x, y, 'r-')

xticks = np.array([0, 2, 4, 6, 8])
xticks_positions = xticks / (27/60)
plt.gca().set_xticks(xticks_positions)
plt.gca().set_xticklabels(xticks)

yticks = np.array([0, 40, 80, 120, 160, 200])
yticks_positions = (yticks / (0.325 * 0.325))/(y_bins[1] - y_bins[0])
plt.gca().set_yticks(yticks_positions)
plt.gca().set_yticklabels(yticks)

plt.savefig('/home/michaelwu/temp.png', dpi=300)



x = np.arange(20)
y_bins = np.linspace(0, np.quantile(large_traj_MSDs[19], 0.95), 41)
density_map = np.zeros((20, 40))
y = []
for i in range(20):
  for d in large_traj_MSDs[i]:
    ind_bin = 40 - (y_bins > d).sum()
    if ind_bin < 40:
      density_map[i][ind_bin] += 1
  y.append(np.mean(large_traj_MSDs[i])/(y_bins[1] - y_bins[0]))

density_map = density_map/density_map.sum(1, keepdims=True)
plt.clf()
plt.imshow(np.transpose(np.log(density_map + 1e-5)), cmap='Blues', vmin=-8, origin='lower')
plt.plot(x, y, 'b-')

xticks = np.array([0, 2, 4, 6, 8])
xticks_positions = xticks / (27/60)
plt.gca().set_xticks(xticks_positions)
plt.gca().set_xticklabels(xticks)

yticks = np.array([0, 400, 800, 1200, 1600])
yticks_positions = (yticks / (0.325 * 0.325))/(y_bins[1] - y_bins[0])
plt.gca().set_yticks(yticks_positions)
plt.gca().set_yticklabels(yticks)

plt.savefig('/home/michaelwu/temp2.png', dpi=300)





# plt.plot(x, [small_traj_MSDs[i][0] for i in x], 'b-')
# plt.fill_between(x, 
#                  [small_traj_MSDs[i][0] - small_traj_MSDs[i][1] for i in x],
#                  [small_traj_MSDs[i][0] + small_traj_MSDs[i][1] for i in x], facecolor='blue', alpha=0.2)
# plt.plot(x, [large_traj_MSDs[i][0] for i in x], 'r-')
# plt.fill_between(x, 
#                  [large_traj_MSDs[i][0] - large_traj_MSDs[i][1] for i in x],
#                  [large_traj_MSDs[i][0] + large_traj_MSDs[i][1] for i in x], facecolor='red', alpha=0.2)
# plt.savefig('/home/michaelwu/MSD.png')

# plt.clf()
# plt.plot(x, [small_traj_MSDs_trimmed[i][0] for i in x], 'b-')
# plt.fill_between(x, 
#                  [small_traj_MSDs_trimmed[i][0] - small_traj_MSDs_trimmed[i][1] for i in x],
#                  [small_traj_MSDs_trimmed[i][0] + small_traj_MSDs_trimmed[i][1] for i in x], facecolor='blue', alpha=0.2)
# plt.plot(x, [large_traj_MSDs_trimmed[i][0] for i in x], 'r-')
# plt.fill_between(x, 
#                  [large_traj_MSDs_trimmed[i][0] - large_traj_MSDs_trimmed[i][1] for i in x],
#                  [large_traj_MSDs_trimmed[i][0] + large_traj_MSDs_trimmed[i][1] for i in x], facecolor='red', alpha=0.2)
# plt.savefig('/home/michaelwu/MSD_trimmed.png')