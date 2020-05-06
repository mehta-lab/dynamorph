#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 16:38:26 2019

@author: michaelwu
"""
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import cm
import imageio
# import tifffile
# import statsmodels.api as sm
from .naive_imagenet import DATA_ROOT

def generate_MSD_distri(trajectories_positions):
  MSD = {i: [] for i in range(1, 15)}
  for t in trajectories_positions:
    for t1 in sorted(t.keys()):
      for t2 in range(t1+1, min(max(t.keys())+1, t1+14)):
        if t2 in t:
          dist = np.linalg.norm(t[t2] - t[t1], ord=2)
          MSD[t2-t1].append(dist**2)
  return MSD

def plot_MSD(trajectories_positions, fit=True, with_intercept=False, first_n_points=5):
  MSD = generate_MSD_distri(trajectories_positions)
  ks = sorted(MSD.keys())
  points = np.array([(k, np.mean(MSD[k])) for k in ks])
  
  plt.plot(points[:, 0], points[:, 1], '.-', label='MSD')
  
  X = points[:first_n_points, 0]
  y = points[:first_n_points, 1]
  if with_intercept:
    X = sm.add_constant(X)
    res = sm.OLS(y, X).fit()
    slope = res.params[1]
    intercept = res.params[0]
    plt.plot(points[:, 0], points[:, 0] * slope + intercept, '--', label='Linear Control')
  else:
    res = sm.OLS(y, X).fit()
    slope = res.params[0]
    plt.plot(points[:, 0], points[:, 0] * slope, '--', label='Linear Control')
  plt.legend()
  return

def generate_short_traj_collections(trajectories_positions, length=5, raw=False):
  short_trajs = []
  for t in trajectories_positions:
    t_keys = sorted(t.keys())
    assert len(t_keys) > length
    for t_point in range(len(t_keys) - (length - 1)):
      if raw:
        short_trajs.append({t_keys[t_point+i]: t[t_keys[t_point+i]] for i in range(length)})
      else:
        short_t = [t[t_keys[t_point + i]] for i in range(length)]
        
        short_t_ = []
        #initial_position = short_t[0]
        for i in range(length - 1):
          d = np.linalg.norm(short_t[i+1] - short_t[i], ord=2)
          #d2 = np.linalg.norm(short_t[i+1] - initial_position, ord=2)
          short_t_.append(d)
          #short_t_2.append(d2/np.sqrt(i+1))
        short_trajs.append(short_t_)
  return short_trajs

def save_traj(k, output_path=None):
  input_path = DATA_ROOT + '/Data/DynamicPatches/%s/mg_traj_%s.tif' % (k.split('/')[0], k.split('/')[1])
  # images = tifffile.imread(input_path)
  _, images = cv2.imreadmulti(input_path, flags=cv2.IMREAD_ANYDEPTH)
  images = np.array(images)
  if output_path is None:
    output_path = './%s.gif' % (t, k[:9] + '_' + k[10:])
  imageio.mimsave(output_path, images)
  return

if __name__ == '__main__':
  sites = ['D%d-Site_%d' % (i, j) for j in range(9) for i in range(3, 6)]
  
    
  all_mg_trajs = {}
  all_non_mg_trajs = {}
  for site in sites:
    _, mg_trajectories_positions = pickle.load(open(DATA_ROOT + '/Data/DynamicPatches/%s/mg_traj.pkl' % site, 'rb'))
    _, non_mg_trajectories_positions = pickle.load(open(DATA_ROOT + '/Data/DynamicPatches/%s/non_mg_traj.pkl' % site, 'rb'))
    
    for i, traj in enumerate(mg_trajectories_positions):
      all_mg_trajs[site + '/%d' % i] = traj
    for i, traj in enumerate(non_mg_trajectories_positions):
      all_non_mg_trajs[site + '/%d' % i] = traj
  
  # Clustering
  np.random.seed(123)
  traj_length = 9
  n_clusters = 3
  short_trajs = generate_short_traj_collections(all_mg_trajs.values(), length=traj_length)
  clustering = KMeans(n_clusters=n_clusters)
  clustering.fit(short_trajs)
  
  
  clustering_labels = {
      0: '00',
      1: '0',
      2: '000'}
  
  # PCA
  pca = PCA(n_components=3)
  short_trajs_ = pca.fit_transform(short_trajs)
  short_trajs_labels = clustering.predict(short_trajs)
  cmap = cm.get_cmap('tab10')
  plt.clf()
  for i in range(n_clusters):
    plt.scatter(short_trajs_[:, 0][np.where(short_trajs_labels == i)],
                short_trajs_[:, 2][np.where(short_trajs_labels == i)],
                s=0.1,
                color=cmap.colors[i],
                label='cluster_%s' % clustering_labels[i])
  plt.legend()
  plt.xlabel("PC1")
  plt.ylabel("PC2")
  plt.savefig('/home/michaelwu/pca_movement.png', dpi=300)
  
  plt.clf()
  plt.plot(pca.components_[0]);
  plt.savefig('/home/michaelwu/pc1_movement_components.png', dpi=300)
  plt.clf()
  plt.plot(pca.components_[1]);
  plt.savefig('/home/michaelwu/pc2_movement_components.png', dpi=300)
  
  # Generate representative trajs

  stagnant_trajs = {}
  minor_moving_trajs = {}
  moving_trajs = {}
  other_trajs = {}
  for k in all_mg_trajs:
    sub_trajs = generate_short_traj_collections([all_mg_trajs[k]], length=traj_length)
    labels = [clustering_labels[l] for l in clustering.predict(sub_trajs)]
    
    if labels.count('0') > 0.7 * len(labels):
      stagnant_trajs[k] = labels
    elif set(labels) <= set(['0', '00']) or labels.count('00') > 0.7 * len(labels):
      # Contains ('00' only) and ('0' and '00')
      minor_moving_trajs[k] = labels
    elif set(labels) <= set(['00', '000']) or labels.count('000') > 0.4 * len(labels):
      # Contains all trajectories with '000' and '0000' but not '0'
      moving_trajs[k] = labels
    else:
      other_trajs[k] = labels
  
  clustered = {"stagnant": list(stagnant_trajs.keys()),
               "minor_moving": list(minor_moving_trajs.keys()),
               "moving": list(moving_trajs.keys())}
  
  
#  os.mkdir('./movement_clustered_trajs')
#  os.mkdir('./movement_clustered_trajs/stagnant')
#  os.mkdir('./movement_clustered_trajs/minor_moving')
#  os.mkdir('./movement_clustered_trajs/moving')
#  os.mkdir('./movement_clustered_trajs/other')
#  for k in np.random.choice(list(stagnant_trajs.keys()), (30,), replace=False):
#    save_traj(k, './movement_clustered_trajs/stagnant/%s.gif' % (k[:9] + '_' + k[10:]))
#  for k in np.random.choice(list(minor_moving_trajs.keys()), (30,), replace=False):
#    save_traj(k, './movement_clustered_trajs/minor_moving/%s.gif' % (k[:9] + '_' + k[10:]))
#  for k in np.random.choice(list(moving_trajs.keys()), (30,), replace=False):
#    save_traj(k, './movement_clustered_trajs/moving/%s.gif' % (k[:9] + '_' + k[10:]))
#  for k in np.random.choice(list(other_trajs.keys()), (30,), replace=False):
#    save_traj(k, './movement_clustered_trajs/other/%s.gif' % (k[:9] + '_' + k[10:]))
  
  # MSD curve
  plt.clf()
  plot_MSD(list(all_mg_trajs.values()))
  plt.xlabel("time step")
  plt.ylabel("distance^2")
  plt.savefig("/home/michaelwu/all_microglia_combined.png", dpi=300)
  plt.clf()
  plot_MSD([all_mg_trajs[t] for t in clustered["stagnant"]])
  plt.xlabel("time step")
  plt.ylabel("distance^2")
  plt.savefig("/home/michaelwu/mg_stagnant.png", dpi=300)
  plt.clf()
  plot_MSD([all_mg_trajs[t] for t in clustered["minor_moving"]])
  plt.xlabel("time step")
  plt.ylabel("distance^2")
  plt.savefig("/home/michaelwu/mg_minor_moving.png", dpi=300)
  plt.clf()
  plot_MSD([all_mg_trajs[t] for t in clustered["moving"]])
  plt.xlabel("time step")
  plt.ylabel("distance^2")
  plt.savefig("/home/michaelwu/mg_moving.png", dpi=300)
  plt.clf()
  plot_MSD(list(all_non_mg_trajs.values()))
  plt.xlabel("time step")
  plt.ylabel("distance^2")
  plt.savefig("/home/michaelwu/all_non_microglia_combined.png", dpi=300)
  

