import numpy as np
import cv2
import os
import pickle
import torch as t
import h5py
import pandas as pd
from NNsegmentation.models import Segment
from NNsegmentation.data import predict_whole_map
from SingleCellPatch.extract_patches import within_range
from pipeline.segmentation import instance_clustering
from SingleCellPatch.generate_trajectories import frame_matching
import matplotlib
from matplotlib import cm
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import seaborn as sns
import imageio
from HiddenStateExtractor.vq_vae import VQ_VAE, CHANNEL_MAX, CHANNEL_VAR, prepare_dataset
from HiddenStateExtractor.naive_imagenet import read_file_path, DATA_ROOT
from HiddenStateExtractor.morphology_clustering import select_clean_trajecteories, Kmean_on_short_trajs
from HiddenStateExtractor.movement_clustering import save_traj
import statsmodels.api as sm
import scipy

RAW_DATA_PATH = '/mnt/comp_micro/Projects/CellVAE/Combined'
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

feat = 'save_0005_before'
fs = sorted(pickle.load(open('./HiddenStateExtractor/file_paths_bkp.pkl', 'rb')))
trajs = pickle.load(open('./HiddenStateExtractor/trajectory_in_inds.pkl', 'rb'))
dats_ = pickle.load(open('./HiddenStateExtractor/%s_PCA.pkl' % feat, 'rb'))
sizes = pickle.load(open(DATA_ROOT + '/Data/EncodedSizes.pkl', 'rb'))
ss = [sizes[f][0] for f in fs]


PC1_vals = dats_[:, 0]
PC1_range = (np.quantile(PC1_vals, 0.4), np.quantile(PC1_vals, 0.6))
PC2_vals = dats_[:, 1]
PC2_range = (np.quantile(PC2_vals, 0.4), np.quantile(PC2_vals, 0.6))

# PC1
vals = dats_[:, 0]
path = '/data/michaelwu/CellVAE/PC_samples/PC1'
val_std = np.std(vals)

thr0 = np.quantile(vals, 0.1)
thr1 = np.quantile(vals, 0.9)
samples0 = [f for i, f in enumerate(fs) if vals[i] < thr0]
samples1 = [f for i, f in enumerate(fs) if vals[i] > thr1]
sample_ts = []
for t in trajs:
  traj_PCs = np.array([vals[ind] for ind in trajs[t]])
  start = np.mean(traj_PCs[:3])
  end = np.mean(traj_PCs[-3:])
  traj_PC_diff = traj_PCs[1:] - traj_PCs[:-1]
  if np.abs(end - start) > 1.2 * val_std and np.median(traj_PC_diff) < 0.5 * val_std:
    sample_ts.append(t)

np.random.seed(123)
for i, f in enumerate(np.random.choice(samples0, (10,), replace=False)):
  plot_patch(f, path + '/sample_low_%d.png' % i)
for i, f in enumerate(np.random.choice(samples1, (10,), replace=False)):
  plot_patch(f, path + '/sample_high_%d.png' % i)
for t in np.random.choice(sample_ts, (10,), replace=False):
  save_traj(t, path + '/sample_traj_%s.gif' % t.replace('/', '_'))

# PC2, controlling for PC1
vals = dats_[:, 1]
path = '/data/michaelwu/CellVAE/PC_samples/PC2'
vals_filtered = [v for i, v in enumerate(vals) if PC1_range[0] < PC1_vals[i] < PC1_range[1]]
val_std = np.std(vals_filtered)

thr0 = np.quantile(vals_filtered, 0.1)
thr1 = np.quantile(vals_filtered, 0.9)
samples0 = [f for i, f in enumerate(fs) if vals[i] < thr0 and PC1_range[0] < PC1_vals[i] < PC1_range[1]]
samples1 = [f for i, f in enumerate(fs) if vals[i] > thr1 and PC1_range[0] < PC1_vals[i] < PC1_range[1]]
sample_ts = []
for t in trajs:
  traj_PCs = np.array([vals[ind] for ind in trajs[t]])
  start = np.mean(traj_PCs[:3])
  end = np.mean(traj_PCs[-3:])
  traj_PC_diff = traj_PCs[1:] - traj_PCs[:-1]
  if np.abs(end - start) > 1.2 * val_std and np.median(traj_PC_diff) < 0.5 * val_std:
    sample_ts.append(t)

np.random.seed(123)
for i, f in enumerate(np.random.choice(samples0, (10,), replace=False)):
  plot_patch(f, path + '/sample_low_%d.png' % i)
for i, f in enumerate(np.random.choice(samples1, (10,), replace=False)):
  plot_patch(f, path + '/sample_high_%d.png' % i)
for t in np.random.choice(sample_ts, (10,), replace=False):
  save_traj(t, path + '/sample_traj_%s.gif' % t.replace('/', '_'))


# PC3, controlling for PC1, PC2
vals = dats_[:, 2]
path = '/data/michaelwu/CellVAE/PC_samples/PC3'
vals_filtered = [v for i, v in enumerate(vals) \
    if PC1_range[0] < PC1_vals[i] < PC1_range[1] and PC2_range[0] < PC2_vals[i] < PC2_range[1]]
val_std = np.std(vals_filtered)

thr0 = np.quantile(vals_filtered, 0.1)
thr1 = np.quantile(vals_filtered, 0.9)
samples0 = [f for i, f in enumerate(fs) if vals[i] < thr0 and \
    PC1_range[0] < PC1_vals[i] < PC1_range[1] and PC2_range[0] < PC2_vals[i] < PC2_range[1]]
samples1 = [f for i, f in enumerate(fs) if vals[i] > thr1 and \
    PC1_range[0] < PC1_vals[i] < PC1_range[1] and PC2_range[0] < PC2_vals[i] < PC2_range[1]]
sample_ts = []
for t in trajs:
  traj_PCs = np.array([vals[ind] for ind in trajs[t]])
  start = np.mean(traj_PCs[:3])
  end = np.mean(traj_PCs[-3:])
  traj_PC_diff = traj_PCs[1:] - traj_PCs[:-1]
  if np.abs(end - start) > 1.2 * val_std and np.median(traj_PC_diff) < 0.5 * val_std:
    sample_ts.append(t)

np.random.seed(123)
for i, f in enumerate(np.random.choice(samples0, (10,), replace=False)):
  plot_patch(f, path + '/sample_low_%d.png' % i)
for i, f in enumerate(np.random.choice(samples1, (10,), replace=False)):
  plot_patch(f, path + '/sample_high_%d.png' % i)
for t in np.random.choice(sample_ts, (10,), replace=False):
  save_traj(t, path + '/sample_traj_%s.gif' % t.replace('/', '_'))


# PC4, controlling for PC1, PC2
vals = dats_[:, 3]
path = '/data/michaelwu/CellVAE/PC_samples/PC4'
vals_filtered = [v for i, v in enumerate(vals) \
    if PC1_range[0] < PC1_vals[i] < PC1_range[1] and PC2_range[0] < PC2_vals[i] < PC2_range[1]]
val_std = np.std(vals_filtered)

thr0 = np.quantile(vals_filtered, 0.1)
thr1 = np.quantile(vals_filtered, 0.9)
samples0 = [f for i, f in enumerate(fs) if vals[i] < thr0 and \
    PC1_range[0] < PC1_vals[i] < PC1_range[1] and PC2_range[0] < PC2_vals[i] < PC2_range[1]]
samples1 = [f for i, f in enumerate(fs) if vals[i] > thr1 and \
    PC1_range[0] < PC1_vals[i] < PC1_range[1] and PC2_range[0] < PC2_vals[i] < PC2_range[1]]
sample_ts = []
for t in trajs:
  traj_PCs = np.array([vals[ind] for ind in trajs[t]])
  start = np.mean(traj_PCs[:3])
  end = np.mean(traj_PCs[-3:])
  traj_PC_diff = traj_PCs[1:] - traj_PCs[:-1]
  if np.abs(end - start) > 1.2 * val_std and np.median(traj_PC_diff) < 0.5 * val_std:
    sample_ts.append(t)

np.random.seed(123)
for i, f in enumerate(np.random.choice(samples0, (10,), replace=False)):
  plot_patch(f, path + '/sample_low_%d.png' % i)
for i, f in enumerate(np.random.choice(samples1, (10,), replace=False)):
  plot_patch(f, path + '/sample_high_%d.png' % i)
for t in np.random.choice(sample_ts, (10,), replace=False):
  save_traj(t, path + '/sample_traj_%s.gif' % t.replace('/', '_'))