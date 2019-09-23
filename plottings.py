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
from NNsegmentation.models import Segment
from NNsegmentation.data import predict_whole_map
from SingleCellPatch.extract_patches import instance_clustering
from SingleCellPatch.generate_trajectories import frame_matching
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

RAW_DATA_PATH = '/mnt/comp_micro/Projects/CellVAE/Combined'
color_mg = np.array([240, 94, 56], dtype='uint8')
color_nonmg = np.array([66, 101, 251], dtype='uint8')
color_bg = np.array([150, 150, 150], dtype='uint8')
color_fg = (color_mg * 0.7 + color_nonmg * 0.3).astype('uint8')

raw_input_stack = np.load(RAW_DATA_PATH + '/D5-Site_0.npy')
raw_input = raw_input_stack[0, :, :, 0:1]
cv2.imwrite('/home/michaelwu/fig2_raw.png', raw_input)

annotations = np.load(RAW_DATA_PATH + '/D5-Site_0_Annotations.npy')
annotations = annotations[0]
mat = np.zeros((annotations.shape[0], annotations.shape[1], 3), dtype='uint8')
mat[:, :] = (raw_input / 256).astype('uint8')
alpha = 0.7
mat[np.where(annotations == 3)[:2]] = (1 - alpha) * mat[np.where(annotations == 3)[:2]] + alpha * color_nonmg.reshape((1, 3))
mat[np.where(annotations == 2)[:2]] = (1 - alpha) * mat[np.where(annotations == 2)[:2]] + alpha * color_mg.reshape((1, 3))
mat[np.where(annotations == 1)[:2]] = (1 - alpha) * mat[np.where(annotations == 1)[:2]] + alpha * color_bg.reshape((1, 3))
cv2.imwrite('/home/michaelwu/fig2_annotations.png', mat)

RF_predictions = np.load(RAW_DATA_PATH + '/D5-Site_0_RFProbabilities.npy')[0]
RF_bg = RF_predictions[:, :, 0:1]
RF_fg = RF_predictions[:, :, 1:2] + RF_predictions[:, :, 2:3]
mat = np.zeros((annotations.shape[0], annotations.shape[1], 3), dtype='uint8')
mat[:, :] = (raw_input / 256).astype('uint8')
positions = np.where((RF_bg > 0.5))[:2]
alpha = 0.7
mat[positions] = (1 - alpha) * mat[positions] + alpha * color_bg.reshape((1, 3))
cv2.imwrite('/home/michaelwu/fig2_rf_predictions.png', mat)

# model = Segment(input_shape=(256, 256, 2), 
#                 unet_feat=32,
#                 fc_layers=[64, 32],
#                 n_classes=3,
#                 model_path='./NNsegmentation/temp_save')
# model.load(model.model_path + '/final.h5')
# NN_predictions = predict_whole_map(np.load(RAW_DATA_PATH + '/D5-Site_0.npy')[0:1], model, n_supp=20)[0]
NN_predictions_stack = np.load(RAW_DATA_PATH + '/D5-Site_0_NNProbabilities.npy')
NN_predictions = NN_predictions_stack[0]

mat = np.zeros((annotations.shape[0], annotations.shape[1], 3), dtype='uint8')
mat[:, :] = (raw_input / 256).astype('uint8')
mg_positions = np.where(NN_predictions[:, :, 1] > 0.5)[:2]
nonmg_positions = np.where(NN_predictions[:, :, 2] > 0.5)[:2]
alpha = 0.7
mat = mat * (1 - (alpha * NN_predictions[:, :, 0:1])) + np.ones_like(mat) * color_bg.reshape((1, 1, 3)) * (alpha * NN_predictions[:, :, 0:1])
# mat[nonmg_positions] = (mat * (1 - (alpha * NN_predictions[:, :, 2:3])) + np.ones_like(mat) * color_nonmg.reshape((1, 1, 3)) * (alpha * NN_predictions[:, :, 2:3]))[nonmg_positions]
# mat[mg_positions] = (mat * (1 - (alpha * NN_predictions[:, :, 1:2])) + np.ones_like(mat) * color_mg.reshape((1, 1, 3)) * (alpha * NN_predictions[:, :, 1:2]))[mg_positions]
mat = (mat * (1 - (alpha * NN_predictions[:, :, 2:3])) + np.ones_like(mat) * color_nonmg.reshape((1, 1, 3)) * (alpha * NN_predictions[:, :, 2:3]))
mat = (mat * (1 - (alpha * NN_predictions[:, :, 1:2])) + np.ones_like(mat) * color_mg.reshape((1, 1, 3)) * (alpha * NN_predictions[:, :, 1:2]))
cv2.imwrite('/home/michaelwu/fig2_nn_predictions.png', mat)

# mat = np.zeros((annotations.shape[0], annotations.shape[1], 3), dtype='uint8')
# mat[:, :] = np.expand_dims((raw_input / 256).astype('uint8'), 2)
# mg_positions = np.where(RF_predictions[:, :, 1] > 0.5)[:2]
# nonmg_positions = np.where(RF_predictions[:, :, 2] > 0.5)[:2]
# alpha = 0.7
# mat = mat * (1 - (alpha * RF_predictions[:, :, 0:1])) + np.ones_like(mat) * color_bg.reshape((1, 1, 3)) * (alpha * RF_predictions[:, :, 0:1])
# mat[mg_positions] = (mat * (1 - (alpha * RF_predictions[:, :, 1:2])) + np.ones_like(mat) * color_mg.reshape((1, 1, 3)) * (alpha * RF_predictions[:, :, 1:2]))[mg_positions]
# mat[nonmg_positions] = (mat * (1 - (alpha * RF_predictions[:, :, 2:3])) + np.ones_like(mat) * color_nonmg.reshape((1, 1, 3)) * (alpha * RF_predictions[:, :, 2:3]))[nonmg_positions]
# cv2.imwrite('/home/michaelwu/off_fig2_rf_predictions_annotation_only.png', mat)

# model = Segment(input_shape=(256, 256, 2), 
#                 unet_feat=32,
#                 fc_layers=[64, 32],
#                 n_classes=3,
#                 model_path='./NNsegmentation/temp_save')
# model.load(model.model_path + '/final.h5')
# NN_predictions = predict_whole_map(np.load(RAW_DATA_PATH + '/D5-Site_0.npy')[20:21], model, n_supp=20)[0]
# mat = np.zeros((annotations.shape[0], annotations.shape[1], 3), dtype='uint8')
# mat[:, :] = np.expand_dims((raw_input / 256).astype('uint8'), 2)
# mg_positions = np.where(NN_predictions[:, :, 1] > 0.5)[:2]
# nonmg_positions = np.where(NN_predictions[:, :, 2] > 0.5)[:2]
# alpha = 0.7
# mat = mat * (1 - (alpha * NN_predictions[:, :, 0:1])) + np.ones_like(mat) * color_bg.reshape((1, 1, 3)) * (alpha * NN_predictions[:, :, 0:1])
# mat[mg_positions] = (mat * (1 - (alpha * NN_predictions[:, :, 1:2])) + np.ones_like(mat) * color_mg.reshape((1, 1, 3)) * (alpha * NN_predictions[:, :, 1:2]))[mg_positions]
# mat[nonmg_positions] = (mat * (1 - (alpha * NN_predictions[:, :, 2:3])) + np.ones_like(mat) * color_nonmg.reshape((1, 1, 3)) * (alpha * NN_predictions[:, :, 2:3]))[nonmg_positions]
# cv2.imwrite('/home/michaelwu/off_fig2_nn_predictions.png', mat)

# model.load(model.model_path + '/stage0_0.h5')
# NN_predictions = predict_whole_map(np.load(RAW_DATA_PATH + '/D5-Site_0.npy')[20:21], model, n_supp=20)[0]
# mat = np.zeros((annotations.shape[0], annotations.shape[1], 3), dtype='uint8')
# mat[:, :] = np.expand_dims((raw_input / 256).astype('uint8'), 2)
# mg_positions = np.where(NN_predictions[:, :, 1] > 0.5)[:2]
# nonmg_positions = np.where(NN_predictions[:, :, 2] > 0.5)[:2]
# alpha = 0.7
# mat = mat * (1 - (alpha * NN_predictions[:, :, 0:1])) + np.ones_like(mat) * color_bg.reshape((1, 1, 3)) * (alpha * NN_predictions[:, :, 0:1])
# mat[mg_positions] = (mat * (1 - (alpha * NN_predictions[:, :, 1:2])) + np.ones_like(mat) * color_mg.reshape((1, 1, 3)) * (alpha * NN_predictions[:, :, 1:2]))[mg_positions]
# mat[nonmg_positions] = (mat * (1 - (alpha * NN_predictions[:, :, 2:3])) + np.ones_like(mat) * color_nonmg.reshape((1, 1, 3)) * (alpha * NN_predictions[:, :, 2:3]))[nonmg_positions]
# cv2.imwrite('/home/michaelwu/off_fig2_nn_predictions_annotation_only.png', mat)




cells, positions, positions_labels = instance_clustering(NN_predictions, fg_thr=0.2)
mg_cell_positions, non_mg_cell_positions, other_cells = cells
mat = np.zeros((annotations.shape[0], annotations.shape[1], 3), dtype='uint8')
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



mat = np.zeros((annotations.shape[0], annotations.shape[1], 3), dtype='uint8')
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

# Generate Frame-frame matching
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

mat0 = np.zeros((annotations.shape[0], annotations.shape[1], 3), dtype='uint8')
mat1 = np.zeros((annotations.shape[0], annotations.shape[1], 3), dtype='uint8')
mat0[:, :] = (frame0 / 256).astype('uint8')
mat1[:, :] = (frame1 / 256).astype('uint8')

cmap = matplotlib.cm.get_cmap('tab10')
np.random.seed(123)
plotted = []
for i in np.random.choice(np.arange(len(mg_matchings[0])), (5,), replace=False):
  pair = mg_matchings[0][i]
  frame0_position = None
  for mg in cell_positions[0][0]:
    if mg[0] == pair[0]:
      frame0_position = mg[1]
      break
  mat0 = add_box(mat0, frame0_position, np.array(cmap.colors[i%10]) * 255)
  frame1_position = None
  for mg in cell_positions[1][0]:
    if mg[0] == pair[1]:
      frame1_position = mg[1]
      break
  mat1 = add_box(mat1, frame1_position, np.array(cmap.colors[i%10]) * 255)
  plotted.append((frame0_position, frame1_position))

for i in np.random.choice(np.arange(len(non_mg_matchings[0])), (2,), replace=False):
  pair = non_mg_matchings[0][i]
  frame0_position = None
  for non_mg in cell_positions[0][1]:
    if non_mg[0] == pair[0]:
      frame0_position = non_mg[1]
      break
  mat0 = add_box(mat0, frame0_position, np.array(cmap.colors[i%10]) * 255)
  frame1_position = None
  for non_mg in cell_positions[1][1]:
    if non_mg[0] == pair[1]:
      frame1_position = non_mg[1]
      break
  mat1 = add_box(mat1, frame1_position, np.array(cmap.colors[i%10]) * 255)
cv2.imwrite('/home/michaelwu/fig2_traj_matching_f0.png', mat0)
cv2.imwrite('/home/michaelwu/fig2_traj_matching_f1.png', mat1)



all_saved_trajs = pickle.load(open(os.path.split(RAW_DATA_PATH)[0] + '/Data/DynamicPatches/D5-Site_0/mg_traj.pkl', 'rb'))
mg_trajectories, mg_trajectories_positions = all_saved_trajs0
def distance(p1, p2):
  return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
target_traj = plotted[1]
for i, traj in enumerate(mg_trajectories_positions):
  if 0 in traj and 1 in traj:
    if distance(traj[0], target_traj[0]) < 50 and distance(traj[1], target_traj[1]) < 50:
      break
mats = []
for t in sorted(mg_trajectories[i].keys()):
  with h5py.File(os.path.split(RAW_DATA_PATH)[0] + '/Data/StaticPatches/D5-Site_0/%d_%d.h5' % (t, mg_trajectories[i][t]), 'r') as f:
    mats.append(np.array(f['masked_mat'][:, :, 0]))
for i in [0, 1, 5, 10, 20]:
  cv2.imwrite('/home/michaelwu/fig2_sample_traj_%d.png' % i, mats[i].astype('uint16'))




################################################################################################
from HiddenStateExtractor.vq_vae import VQ_VAE, CHANNEL_MAX, CHANNEL_VAR, prepare_dataset
from HiddenStateExtractor.naive_imagenet import read_file_path, DATA_ROOT
from HiddenStateExtractor.morphology_clustering import select_clean_trajecteories, Kmean_on_short_trajs
import torch as t
import seaborn as sns
import h5py
import pandas as pd


cs = [0, 1]
input_shape = (128, 128)
gpu = False

fs = read_file_path(DATA_ROOT + '/Data/StaticPatches')
np.random.seed(1234)
sample_fs = np.random.choice(fs, (7,), replace=False)
dataset = prepare_dataset(sample_fs, cs=cs, input_shape=input_shape, channel_max=CHANNEL_MAX)

model = VQ_VAE(alpha=0.0005, gpu=gpu)
model.load_state_dict(t.load('./HiddenStateExtractor/save_0005.pt', map_location='cpu'))
for i in range(7):
  sample = dataset[i:(i+1)][0]
  output = model(sample)[0]
  inp = sample.data.numpy()
  out = output.data.numpy()
  cv2.imwrite('/home/michaelwu/fig3_VAE_pair%d_input.png' % i, inp[0, 0] * 255)
  cv2.imwrite('/home/michaelwu/fig3_VAE_pair%d_output.png' % i, out[0, 0] * 255)







feat = 'save_0005_before'
sites = ['D%d-Site_%d' % (i, j) for j in range(9) for i in range(3, 6)]
fs = sorted(read_file_path(DATA_ROOT + '/Data/StaticPatches'))
trajs = pickle.load(open('./HiddenStateExtractor/trajectory_in_inds.pkl', 'rb'))

dats = pickle.load(open(DATA_ROOT + '/Data/%s.pkl' % feat, 'rb'))
ks = sorted([k for k in dats.keys() if dats[k] is not None])
assert ks == fs
sizes = pickle.load(open(DATA_ROOT + '/Data/EncodedSizes.pkl', 'rb'))
ss = [sizes[k] for k in ks]

length = 5
n_clusters = 3
seed = 123
dats_ = pickle.load(open('./HiddenStateExtractor/%s_PCA.pkl' % feat, 'rb')) 

np.random.seed(seed)
clean_trajs = select_clean_trajecteories(dats_, trajs)
traj_classes = Kmean_on_short_trajs(dats_, trajs, length=length, n_clusters=n_clusters)
representative_trajs = {}
traj_names = list(clean_trajs.keys())
np.random.shuffle(traj_names)
for t in traj_names:
  if np.unique(traj_classes[t]).shape[0] == 1:
    cl = str(traj_classes[t][0])
  elif np.unique(traj_classes[t]).shape[0] == 2:
    if np.unique(traj_classes[t][:5]).shape[0] == 1 and \
       np.unique(traj_classes[t][-5:]).shape[0] == 1 and \
       traj_classes[t][0] != traj_classes[t][-1]:
      cl = str(traj_classes[t][0]) + '_' + str(traj_classes[t][-1])
    else:
      continue
  else:
    continue
  if not cl in representative_trajs:
    representative_trajs[cl] = []
  representative_trajs[cl].append(t)

cmap = matplotlib.cm.get_cmap('binary')  
range_min = np.log(min(ss))
range_max = np.log(max(ss))
colors = [cmap(((np.log(s) - range_min)/(range_max - range_min))**1.5) for s in ss]

plt.clf()
sns.set_style('white')
sns.despine()
plt.scatter(dats_[:, 0], dats_[:, 1], c=colors, s=0.3, edgecolors='none')

np.random.seed(seed)
c0_trajs = np.random.choice(representative_trajs['0'], (5,), replace=False)
for t in c0_trajs:
  order = np.array(trajs[t])
  if t==c0_trajs[0]:
    plt.plot(dats_[order][:, 0], dats_[order][:, 1], c='#ffd700', marker='.', linewidth=0.5, markersize=1., label='Small')
  else:
    plt.plot(dats_[order][:, 0], dats_[order][:, 1], c='#ffd700', marker='.', linewidth=0.5, markersize=1.)
with h5py.File(fs[trajs[c0_trajs[4]][0]], 'r') as f:
  example_fig = np.array(f["masked_mat"])[:, :, 0]
  cv2.imwrite('/home/michaelwu/fig3_morphology_pca_example_small0.png', example_fig.astype('uint16'))
with h5py.File(fs[trajs[c0_trajs[4]][-1]], 'r') as f:
  example_fig = np.array(f["masked_mat"])[:, :, 0]
  cv2.imwrite('/home/michaelwu/fig3_morphology_pca_example_small1.png', example_fig.astype('uint16'))

c2_trajs = np.random.choice(representative_trajs['2'], (5,), replace=False)
for t in c2_trajs:
  order = np.array(trajs[t])
  if t==c2_trajs[0]:
    plt.plot(dats_[order][:, 0], dats_[order][:, 1], c='#e42256', marker='.', linewidth=0.5, markersize=1., label='Medium')
  else:
    plt.plot(dats_[order][:, 0], dats_[order][:, 1], c='#e42256', marker='.', linewidth=0.5, markersize=1.)
with h5py.File(fs[trajs[c2_trajs[2]][0]], 'r') as f:
  example_fig = np.array(f["masked_mat"])[:, :, 0]
  cv2.imwrite('/home/michaelwu/fig3_morphology_pca_example_medium0.png', example_fig.astype('uint16'))
with h5py.File(fs[trajs[c2_trajs[2]][-1]], 'r') as f:
  example_fig = np.array(f["masked_mat"])[:, :, 0]
  cv2.imwrite('/home/michaelwu/fig3_morphology_pca_example_medium1.png', example_fig.astype('uint16'))

c1_trajs = np.random.choice(representative_trajs['1'], (5,), replace=False)
for t in c1_trajs:
  order = np.array(trajs[t])
  if t==c1_trajs[0]:
    plt.plot(dats_[order][:, 0], dats_[order][:, 1], c='#00b1b0', marker='.', linewidth=0.5, markersize=1., label='Large')
  else:
    plt.plot(dats_[order][:, 0], dats_[order][:, 1], c='#00b1b0', marker='.', linewidth=0.5, markersize=1.)
with h5py.File(fs[trajs[c1_trajs[4]][0]], 'r') as f:
  example_fig = np.array(f["masked_mat"])[:, :, 0]
  cv2.imwrite('/home/michaelwu/fig3_morphology_pca_example_large0.png', example_fig.astype('uint16'))
with h5py.File(fs[trajs[c1_trajs[4]][-1]], 'r') as f:
  example_fig = np.array(f["masked_mat"])[:, :, 0]
  cv2.imwrite('/home/michaelwu/fig3_morphology_pca_example_large1.png', example_fig.astype('uint16'))


plt.legend(fontsize=14)
plt.xlabel("PC1", fontsize=18)
plt.ylabel("PC2", fontsize=18)

plt.savefig('/home/michaelwu/fig3_morphology_pca.eps')



all_mg_trajs = {}
for site in sites:
  _, mg_trajectories_positions = pickle.load(open(DATA_ROOT + '/Data/DynamicPatches/%s/mg_traj.pkl' % site, 'rb'))
  for i, traj in enumerate(mg_trajectories_positions):
    all_mg_trajs[site + '/%d' % i] = traj

traj_average_moving_distances = {}
for t in representative_trajs['0'] + representative_trajs['1'] + representative_trajs['2']: 
  traj_average_moving_distances[t] = []
  t_keys = sorted(all_mg_trajs[t].keys())

  assert len(t_keys) > length
  dists = []
  for t_point in range(len(t_keys) - 1):
    d = np.linalg.norm(all_mg_trajs[t][t_keys[t_point+1]] - all_mg_trajs[t][t_keys[t_point]], ord=2)
    dists.append(d)
  traj_average_moving_distances[t] = np.log(np.mean(dists) + 1e-9)

cluster_dict = {0: 'Small', 1: 'Large', 2: 'Medium'}
cluster_assign_col = []
average_dist_col = []
for t in representative_trajs['0']:
  cluster_assign_col.append(cluster_dict[0])
  average_dist_col.append(traj_average_moving_distances[t])
for t in representative_trajs['1']:
  cluster_assign_col.append(cluster_dict[1])
  average_dist_col.append(traj_average_moving_distances[t])
for t in representative_trajs['2']:
  cluster_assign_col.append(cluster_dict[2])
  average_dist_col.append(traj_average_moving_distances[t])

df = pd.DataFrame({'cluster': cluster_assign_col, 'aver_dist': average_dist_col})
plt.clf()
sns.set_style('white')
sns.violinplot(y='cluster', 
               x='aver_dist', 
               data=df, 
               order=['Small', 'Medium', 'Large'], 
               palette={'Small': '#ffd700', 'Medium': '#e42256', 'Large': '#00b1b0'},
               orient='h')
#plt.xlim(0, 5)
plt.yticks(fontsize=14, rotation=45)
plt.ylabel('')
plt.xlabel('Average Movement', fontsize=18)
plt.savefig('/home/michaelwu/fig3_aver_dist.eps')


small_traj_represented = ['D5-Site_0/8', 'D5-Site_0/19']
medium_traj_represented = ['D5-Site_0/16', 'D5-Site_0/21']
large_traj_represented = ['D5-Site_0/0', 'D5-Site_0/24']

mat = np.zeros((raw_input.shape[0], raw_input.shape[1], 3), dtype='uint8')
mat[:, :] = (raw_input / 256).astype('uint8')

for t in small_traj_represented:
  traj = all_mg_trajs[t]
  positions = np.stack([traj[k] for k in sorted(traj.keys())])
  for i in range(positions.shape[0] - 1):
    start = positions[i]
    end = positions[i + 1]
    mat = cv2.line(mat, (start[1], start[0]), (end[1], end[0]), (0, 215, 255), thickness=2)

for t in medium_traj_represented:
  traj = all_mg_trajs[t]
  positions = np.stack([traj[k] for k in sorted(traj.keys())])
  for i in range(positions.shape[0] - 1):
    start = positions[i]
    end = positions[i + 1]
    mat = cv2.line(mat, (start[1], start[0]), (end[1], end[0]), (86, 34, 228), thickness=2)

for t in large_traj_represented:
  traj = all_mg_trajs[t]
  positions = np.stack([traj[k] for k in sorted(traj.keys())])
  for i in range(positions.shape[0] - 1):
    start = positions[i]
    end = positions[i + 1]
    mat = cv2.line(mat, (start[1], start[0]), (end[1], end[0]), (176, 177, 0), thickness=2)
cv2.imwrite('/home/michaelwu/fig3_movement_samles.png', mat)