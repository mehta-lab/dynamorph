# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 21:27:26 2021

@author: Zhenqin Wu
"""

import cv2
import numpy as np
import os
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import DBSCAN
from copy import copy

""" Functions for clustering single cells from semantic segmentation """

def within_range(r, pos):
    """ Check if a given position is in window

    Args:
        r (tuple): window, ((int, int), (int, int)) in the form of 
            ((x_low, x_up), (y_low, y_up))
        pos (tuple): (int, int) in the form of (x, y)

    Returns:
        bool: True if `pos` is in `r`, False otherwise

    """
    if pos[0] >= r[0][1] or pos[0] < r[0][0]:
        return False
    if pos[1] >= r[1][1] or pos[1] < r[1][0]:
        return False
    return True


def check_segmentation_dim(segmentation):
    """ Check segmentation mask dimension.
    Add a background channel if n(channels)==1

    Args:
        segmentation: (np.array): segmentation mask for the frame

    """

    assert len(segmentation.shape) == 4, "Semantic segmentation should be formatted with dimension (c, z, x, y)"
    n_channels, _, _, _ = segmentation.shape

    # binary segmentation has only foreground channel, add background channel
    if n_channels == 1:
        segmentation = np.concatenate([1 - segmentation, segmentation], axis=0)
    assert np.allclose(segmentation.sum(0), 1.), "Semantic segmentation doens't sum up to 1"    
    return segmentation


def instance_clustering(cell_segmentation, 
                        ct_thr=(500, 12000), 
                        instance_map=True, 
                        map_path=None, 
                        fg_thr=0.3,
                        dbscan_thr=(10, 250),
                        channel=0):
    """ Perform instance clustering on a static frame

    Args:
        cell_segmentation (np.array): segmentation mask for the frame, 
            size (n_classes(3), z(1), x, y)
        ct_thr (tuple, optional): lower and upper threshold for cell size 
            (number of pixels in segmentation mask)
        instance_map (bool, optional): if to save instance segmentation as an 
            image
        map_path (str or None, optional): path to the image (if `instance_map` 
            is True)
        fg_thr (float, optional): threshold of foreground, any pixel with 
            predicted background prob less than this value would be regarded as
            foreground (MG or Non-MG)
        dbscan_thr (tuple, optional): parameters for DBSCAN, (eps, min_samples)

    Returns:
        (list * 3): 3 lists (MG, Non-MG, intermediate) of cell identifiers
            each entry in the list is a tuple of cell ID and cell center position
        np.array: array of x, y coordinates of foreground pixels
        np.array: array of cell IDs of foreground pixels

    """
    cell_segmentation = check_segmentation_dim(cell_segmentation)
    all_cells = np.mean(cell_segmentation[channel], axis=0) < fg_thr
    positions = np.array(list(zip(*np.where(all_cells))))
    if len(positions) < 1000:
        # No cell detected
        return [], np.zeros((0, 2), dtype=int), np.zeros((0,), dtype=int)

    # DBSCAN clustering of cell pixels
    clustering = DBSCAN(eps=dbscan_thr[0], min_samples=dbscan_thr[1]).fit(positions)
    positions_labels = clustering.labels_
    cell_ids, point_cts = np.unique(positions_labels, return_counts=True)
    
    cell_positions = []
    for cell_id, ct in zip(cell_ids, point_cts):
        if cell_id < 0:
            # neglect unclustered pixels
            continue
        if ct <= ct_thr[0] or ct >= ct_thr[1]:
            # neglect cells that are too small/big
            continue
        points = positions[np.where(positions_labels == cell_id)[0]]
        # calculate cell center
        mean_pos = np.mean(points, 0).astype(int)
        # define window
        window = [(mean_pos[0]-128, mean_pos[0]+128), (mean_pos[1]-128, mean_pos[1]+128)]
        # skip if cell has too many outlying points
        outliers = [p for p in points if not within_range(window, p)]
        if len(outliers) > len(points) * 0.05:
            continue
        cell_positions.append((cell_id, mean_pos))

    # Save instance segmentation results as image
    if instance_map and map_path is not None:
        x_size, y_size = cell_segmentation.shape[-2:]
        # bg as -1
        segmented = np.zeros((x_size, y_size)) - 1
        for cell_id, mean_pos in cell_positions:
            points = positions[np.where(positions_labels == cell_id)[0]]
            for p in points:
                segmented[p[0], p[1]] = cell_id%10
        plt.clf()
        # cmap = matplotlib.cm.get_cmap('tab10')
        cmap = copy(matplotlib.cm.get_cmap("tab10"))
        cmap.set_under(color='k')
        plt.imshow(segmented, cmap=cmap, vmin=-0.001, vmax=10.001)
        font = {'color': 'white', 'size': 4}
        for cell_id, mean_pos in cell_positions:
            plt.text(mean_pos[1], mean_pos[0], str(cell_id), fontdict=font)
        plt.axis('off')
        plt.savefig(map_path, dpi=300)
    return cell_positions, positions, positions_labels


def process_site_instance_segmentation(raw_data,
                                       raw_data_segmented,
                                       site_supp_files_folder,
                                       config_):
    """
    Wrapper method for instance segmentation

    Results will be saved to the supplementary data folder as:
        "cell_positions.pkl": list of cells in each frame (IDs and positions);
        "cell_pixel_assignments.pkl": pixel compositions of cells;
        "segmentation_*.png": image of instance segmentation results.


    :param raw_data: (str) path to image stack (.npy)
    :param raw_data_segmented: (str) path to semantic segmentation stack (.npy)
    :param site_supp_files_folder: (str) path to the folder where supplementary files will be saved
    :param config_: config file parameters
    :return:
    """

    ct_thr = (config_.patch.count_threshold_low, config_.patch.count_threshold_high)
    fg_thr = config_.patch.foreground_threshold
    DBSCAN_thr = (config_.patch.dbscan_eps, config_.patch.dbscan_min_samples)
    channel = config_.patch.channel

    # TODO: Size is hardcoded here
    # Should be of size (n_frame, n_channels, z(1), x(2048), y(2048)), uint16
    print(f"\tLoading {raw_data}")
    image_stack = np.load(raw_data)
    print(f"\traw_data has shape {image_stack.shape}")
    # Should be of size (n_frame, n_classes, z(1), x(2048), y(2048)), float
    print(f"\tLoading {raw_data_segmented}")
    segmentation_stack = np.load(raw_data_segmented)
    print(f"\tsegmentation stack has shape {segmentation_stack.shape}")

    # reshape if numpy file is too large, assume index=1 is redundant
    if len(segmentation_stack.shape) > 4:
        shp = segmentation_stack.shape
        # skip index 1 which is blank
        segmentation_stack = segmentation_stack.reshape((shp[0], shp[2], shp[3], shp[4], shp[5]))

    cell_positions = {}
    cell_pixel_assignments = {}

    # if the number of timepoints between images and predictions mismatch, choose the smaller one
    endpt = image_stack.shape[0] if image_stack.shape[0] < segmentation_stack.shape[0] else segmentation_stack.shape[0]
    for t_point in range(endpt):
        print("\tClustering time %d" % t_point)
        cell_segmentation = segmentation_stack[t_point]
        instance_map_path = os.path.join(site_supp_files_folder, 'segmentation_%d.png' % t_point)
        res = instance_clustering(cell_segmentation,
                                  instance_map=True,
                                  map_path=instance_map_path,
                                  ct_thr=ct_thr,
                                  fg_thr=fg_thr,
                                  dbscan_thr=DBSCAN_thr,
                                  channel=channel)
        print(f"\tfound {len(res[0])} cells for timepoint {t_point}")
        cell_positions[t_point] = res[0]  # List of cell: (cell_id, mean_pos)
        cell_pixel_assignments[t_point] = res[1:]
    with open(os.path.join(site_supp_files_folder, 'cell_positions.pkl'), 'wb') as f:
        pickle.dump(cell_positions, f)
    with open(os.path.join(site_supp_files_folder, 'cell_pixel_assignments.pkl'), 'wb') as f:
        pickle.dump(cell_pixel_assignments, f)
    return
