# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 21:27:26 2021

@author: Zhenqin Wu
"""

import cv2
import numpy as np
import h5py
import os
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import DBSCAN
from copy import copy
import logging
log = logging.getLogger(__name__)

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
    # if segmentation.ndim == 4:
    #     n_channels, n_z, x_full_size, y_full_size = segmentation.shape
    if segmentation.ndim == 3:
        n_channels, x_full_size, y_full_size = segmentation.shape
        # segmentation = segmentation[:, np.newaxis, ...]
    elif segmentation.ndim == 2:
        n_channels = 1
        segmentation = segmentation[np.newaxis, ...]
    else:
        raise ValueError('Semantic segmentation mask must be 2 or 3D, not {}'.format(segmentation.ndim))

    # binary segmentation has only foreground channel, add background channel
    if n_channels == 1:
        segmentation = np.concatenate([1 - segmentation, segmentation], axis=0)
    assert np.allclose(segmentation.sum(0), 1.), "Semantic segmentation doens't sum up to 1"    
    return segmentation


def instance_clustering(cell_segmentation,
                        # ct_thr=(500, 12000),
                        ct_thr=(0, np.inf),
                        save_fig=True,
                        map_path=None,
                        fg_thr=0.5,
                        DBSCAN_thr=(10, 250),
                        # DBSCAN_thr=(2, 15),
                        n_jobs=10):
    """ Perform instance clustering on a static frame

    Args:
        cell_segmentation (np.array): segmentation mask for the frame, 
            size (n_classes(3), z(1), x, y)
        ct_thr (tuple, optional): lower and upper threshold for cell size 
            (number of pixels in segmentation mask)
        save_fig (bool, optional): if to save instance segmentation as an
            image
        map_path (str or None, optional): path to the image (if `save_fig`
            is True)
        fg_thr (float, optional): threshold of foreground, any pixel with 
            predicted background prob less than this value would be regarded as
            foreground (MG or Non-MG)
        DBSCAN_thr (tuple, optional): parameters for DBSCAN, (eps, min_samples)

    Returns:
        (list * 3): 3 lists (MG, Non-MG, intermediate) of cell identifiers
            each entry in the list is a tuple of cell ID and cell center position
        np.array: array of x, y coordinates of foreground pixels
        np.array: array of cell IDs of foreground pixels

    """
    cell_segmentation = check_segmentation_dim(cell_segmentation)
    all_cells = cell_segmentation[0] < fg_thr
    pixel_ids = np.array(list(zip(*np.nonzero(all_cells))))
    if len(pixel_ids) < 1000:
        # No cell detected
        return [], [], np.zeros((0, 2), dtype=int), np.zeros((0,), dtype=int)

    # DBSCAN clustering of cell pixels
    clustering = DBSCAN(eps=DBSCAN_thr[0], min_samples=DBSCAN_thr[1], n_jobs=n_jobs).fit(pixel_ids)
    positions_labels = clustering.labels_
    cell_ids, cell_sizes = np.unique(positions_labels, return_counts=True)
    # neglect unclustered pixels
    cell_sizes = cell_sizes[cell_ids >= 0]
    cell_ids = cell_ids[cell_ids >= 0]
    cell_positions = []
    cell_ids = list(cell_ids)
    cell_ids_new = []
    cell_sizes_new = []
    for cell_id, cell_size in zip(cell_ids, cell_sizes):
        # if cell_size <= ct_thr[0] or cell_size >= ct_thr[1]:
        #     # neglect cells that are too small/big
        #     continue
        points = pixel_ids[np.nonzero(positions_labels == cell_id)[0]]
        # calculate cell center
        mean_pos = np.mean(points, 0).astype(int)
        # define window
        window = [(mean_pos[0]-128, mean_pos[0]+128), (mean_pos[1]-128, mean_pos[1]+128)]
        # skip if cell has too many outlying points
        outliers = [p for p in points if not within_range(window, p)]
        if len(outliers) > len(points) * 0.05:
            continue
        cell_positions.append(mean_pos)
        cell_ids_new.append(cell_id)
        cell_sizes_new.append(cell_size)

    # Save instance segmentation results as image
    if save_fig and map_path is not None:
        x_size, y_size = cell_segmentation.shape[-2:]
        # bg as -1
        segmented = np.zeros((x_size, y_size)) - 1
        for cell_id, mean_pos in zip(cell_ids_new, cell_positions):
            points = pixel_ids[np.where(positions_labels == cell_id)[0]]
            for p in points:
                segmented[p[0], p[1]] = cell_id%10
        plt.clf()
        # cmap = matplotlib.cm.get_cmap('tab10')
        cmap = copy(matplotlib.cm.get_cmap("tab10"))
        cmap.set_under(color='k')
        plt.imshow(segmented, cmap=cmap, vmin=-0.001, vmax=10.001)
        font = {'color': 'white', 'size': 4}
        for cell_id, mean_pos in zip(cell_ids_new, cell_positions):
            plt.text(mean_pos[1], mean_pos[0], str(cell_id), fontdict=font)
        plt.axis('off')
        plt.savefig(map_path, dpi=300)
    return cell_ids_new, cell_positions, cell_sizes_new, pixel_ids, positions_labels


def process_site_instance_segmentation(site,
                                       raw_data,
                                       raw_data_segmented,
                                       site_supp_files_folder,
                                       **kwargs):
    """
    Wrapper method for instance segmentation

    Results will be saved to the supplementary data folder as:
        "cell_positions.pkl": list of cells in each frame (IDs and positions);
        "cell_pixel_assignments.pkl": pixel compositions of cells;
        "segmentation_*.png": image of instance segmentation results.


    :param raw_data: (str) path to image stack (.npy)
    :param raw_data_segmented: (str) path to semantic segmentation stack (.npy)
    :param site_supp_files_folder: (str) path to the folder where supplementary files will be saved
    :param kwargs:
    :return:
    """

    # TODO: Size is hardcoded here
    # Should be of size (n_frame, n_channels, z(1), x(2048), y(2048)), uint16
    print(f"\tLoading {raw_data}")
    image_stack = np.load(raw_data)
    # Should be of size (n_frame, n_classes, z(1), x(2048), y(2048)), float
    print(f"\tLoading {raw_data_segmented}")
    segmentation_stack = np.load(raw_data_segmented)
    meta_list = []
    cell_positions = {}
    cell_pixel_assignments = {}
    for t_point in range(image_stack.shape[0]):
        cell_positions[t_point] = {}
        cell_pixel_assignments[t_point] = {}
        for z in range(image_stack.shape[2]):
            print("\tClustering time {} z {}".format(t_point, z))
            cell_segmentation = segmentation_stack[t_point, :, z, ...]
            instance_map_path = os.path.join(site_supp_files_folder, 'segmentation_t{}_z{}.png'.format(t_point, z))
            #TODO: expose instance clustering parameters in config
            cell_ids, positions, cell_sizes, pixel_ids, positions_labels = \
                instance_clustering(cell_segmentation, save_fig=False, map_path=instance_map_path)
            cell_positions[t_point][z] = list(zip(cell_ids, positions)) # List of cell: (cell_id, mean_pos)
            cell_pixel_assignments[t_point][z] = (pixel_ids, positions_labels)
            # new metadata format
            for cell_id, cell_pos, cell_size in zip(cell_ids, positions, cell_sizes):
                meta_row = {'FOV': site,
                            'time': t_point,
                            'slice': z,
                            'cell ID': cell_id,
                            'cell position': cell_pos,
                            'cell size': cell_size}
                meta_list.append(meta_row)
    with open(os.path.join(site_supp_files_folder, 'cell_positions.pkl'), 'wb') as f:
        pickle.dump(cell_positions, f)
    with open(os.path.join(site_supp_files_folder, 'cell_pixel_assignments.pkl'), 'wb') as f:
        pickle.dump(cell_pixel_assignments, f)
    return meta_list
