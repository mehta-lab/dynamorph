#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 16:19:26 2019

@author: michaelwu
"""
import pickle
from naive_imagenet import read_file_path, DATA_ROOT

fs = read_file_path(DATA_ROOT + '/Data/StaticPatches')
relations = {}
sites = ['D%d-Site_%d' % (i, j) for j in range(9) for i in range(3, 6)]

path = '/mnt/comp_micro/Projects/CellVAE'
for site in sites:
  print(site)
  trajectories = pickle.load(open(path + '/Data/DynamicPatches/%s/mg_traj.pkl' % site, 'rb'))[0]
  for t in trajectories:
    keys = sorted(t.keys())
    t_inds = []
    for k in keys:
      a_name = path + '/Data/StaticPatches/%s/%d_%d.h5' % (site, k, t[k])
      t_inds.append(fs.index(a_name))
      if k-1 in keys:
        a_name = path + '/Data/StaticPatches/%s/%d_%d.h5' % (site, k, t[k])
        b_name = path + '/Data/StaticPatches/%s/%d_%d.h5' % (site, k-1, t[k-1])
        relations[(fs.index(a_name), fs.index(b_name))] = 2
      if k+1 in keys:
        a_name = path + '/Data/StaticPatches/%s/%d_%d.h5' % (site, k, t[k])
        b_name = path + '/Data/StaticPatches/%s/%d_%d.h5' % (site, k+1, t[k+1])
        relations[(fs.index(a_name), fs.index(b_name))] = 2
    
    for i in t_inds:
      for j in t_inds:
        if not (i, j) in relations:
          relations[(i, j)] = 1
        
with open(path + '/Data/StaticPatchesAllRelations.pkl', 'wb') as f:
  pickle.dump(relations, f)
