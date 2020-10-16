#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 16:19:26 2019

@author: michaelwu
"""
import pickle
import os
from .naive_imagenet import read_file_path

# Support script for generating relation dict, which is used for 
# dataset re-ordering (necessary for matching loss)

# `relations` is a dict recording relations (2 - adjacent frame, 1 - same 
# trajectory) of pairs of frames

def generate_trajectory_relations(sites, raw_folder, supp_folder):
    """ Find pair relations (adjacent frame, same trajectory) in static patches

    Results will be saved under `raw_folder`

    Args:
        sites (list of str): sites from the same well
        raw_folder (str): path to save image stacks, segmentation stacks, etc.
        supp_folder (str): path to save supplementary data

    """
    assert len(set(s[:2] for s in sites)) == 1
    well = sites[0][:2]
    fs = pickle.load(open(os.path.join(raw_folder, "%s_file_paths.pkl" % well), 'rb'))
    relations = {}

    def find_key(fs, key):
        inds = []
        for i, f in enumerate(fs):
            if key in f:
                inds.append(i)
        return inds[0] if len(inds) == 1 else None

    for site in sites:
        print(site)
        trajectories = pickle.load(open(
            os.path.join(supp_folder, "%s-supps" % well, site, "cell_traj.pkl"), 'rb'))[0]

        for t in trajectories:
            keys = sorted(t.keys())
            t_inds = []
            for k in keys:
                a_ind = find_key(fs, '/%s/%d_%d.' % (site, k, t[k]))
                assert not a_ind is None
                t_inds.append(a_ind)
                # Adjacent frames
                if k-1 in keys:
                    b_ind = find_key(fs, '/%s/%d_%d.' % (site, k-1, t[k-1]))
                    relations[(a_ind, b_ind)] = 2
                if k+1 in keys:
                    b_ind = find_key(fs, '/%s/%d_%d.' % (site, k+1, t[k+1]))
                    relations[(a_ind, b_ind)] = 2
            
            # Same trajectory
            for i in t_inds:
                for j in t_inds:
                    if not (i, j) in relations:
                        relations[(i, j)] = 1

    with open(os.path.join(raw_folder, "%s_static_patches_relations.pkl" % well), 'wb') as f:
        pickle.dump(relations, f)
    return