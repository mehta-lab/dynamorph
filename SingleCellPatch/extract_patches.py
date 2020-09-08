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

""" Functions for extracting single cells from static frames """


def select_window(mat, window, padding=0.):
    """ Extract submatrix

    Submatrix of `window` will be extracted from `mat`,
    negative boundaries are allowed (padded)
    
    Args:
        mat (np.array): target matrix, size should be 2048 * 2048 * C
            TODO: size is hardcoded now
        window (tuple): area-of-interest for submatrix, ((int, int), (int, int))
            in the form of ((x_low, x_up), (y_low, y_up))
        padding (float, optional): padding value for negative boundaries

    Returns:
        np.array: submatrix-of-interest
    
    """
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


# filter 1 is for the masking of surrounding cells
size1 = 11
filter1 = np.zeros((size1, size1), dtype=int)
for i in range(size1):
    for j in range(size1):
        if np.sqrt((i-size1//2)**2 + (j-size1//2)**2) <= size1//2:
            filter1[i, j] = 1

# filter 2 is for (un-)masking of center cell
size2 = 21 # any pixel within a distance of 21 to center cell will not be masked
filter2 = np.zeros((size2, size2), dtype=int)
for i in range(size2):
    for j in range(size2):
        if np.sqrt((i-size2//2)**2 + (j-size2//2)**2) < size2//2:
            filter2[i, j] = 1

def generate_mask(positions, positions_labels, cell_id, window, window_segmentation):
    """ Generate mask matrix for surrounding cells

    Args:
        positions (np.array): array of x, y coordinates, size (N, 2)
        positions_labels (np.array): cell IDs of each pixel, size (N,)
        cell_id (int): target cell ID
        window (tuple): window centered around the target cell, 
            ((int, int), (int, int)) in the form of 
            ((x_low, x_up), (y_low, y_up))
        window_segmentation (np.array): pixel-level semantic segmentation in 
            the window area, size (window_x, window_y, n_classes)

    Returns:
        np.array: remove mask (1s will be filled by median)
        np.array: target mask
        np.array: enlarged target mask

    """
    x_size = window[0][1] - window[0][0]
    y_size = window[1][1] - window[1][0]
    remove_mask = np.zeros((x_size, y_size), dtype=int)
    target_mask = np.zeros((x_size, y_size), dtype=int)

    for i, p in enumerate(positions):
        if not within_range(window, p):
            continue
        if positions_labels[i] != cell_id and positions_labels[i] >= 0:
            # pixels belonging to other cells
            remove_mask[p[0] - window[0][0], p[1] - window[1][0]] = 1
        if positions_labels[i] == cell_id:
            # pixels belonging to target cell
            target_mask[p[0] - window[0][0], p[1] - window[1][0]] = 1

    remove_mask = np.sign(convolve2d(remove_mask, filter1, mode='same'))
    target_mask2 = np.sign(convolve2d(target_mask, filter2, mode='same'))
    # Target mask override remove mask
    remove_mask = ((remove_mask - target_mask2) > 0) * 1
    # Add negative boundary paddings
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
    """ Perform instance clustering on a static frame

    Args:
        cell_segmentation (np.array): segmentation mask for the frame
        ct_thr (tuple, optional): lower and upper threshold for cell size (number 
            of pixels in segmentation mask)
        instance_map (bool, optional): if to save instance segmentation as an 
            image
        map_path (str or None, optional): path to the image (if `instance_map` 
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
    all_cells = cell_segmentation[:, :, 0] < fg_thr
    positions = np.array(list(zip(*np.where(all_cells))))
    if len(positions) < 1000:
        # No cell detected
        return ([], [], []), np.zeros((0, 2), dtype=int), np.zeros((0,), dtype=int)

    # DBSCAN clustering of cell pixels
    clustering = DBSCAN(eps=DBSCAN_thr[0], min_samples=DBSCAN_thr[1]).fit(positions)
    positions_labels = clustering.labels_
    cell_ids, point_cts = np.unique(positions_labels, return_counts=True)
    
    mg_cell_positions = []
    non_mg_cell_positions = []
    other_cells = []
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
        cell_segmentation_labels = cell_segmentation[points[:, 0], points[:, 1]]
        # Calculate if MG/Non-MG/intermediate
        mg_ratio = (np.argmax(cell_segmentation_labels, 1) == 1).sum()/len(points)
        non_mg_ratio = (np.argmax(cell_segmentation_labels, 1) == 2).sum()/len(points)
        if mg_ratio > 0.9:
            mg_cell_positions.append((cell_id, mean_pos))
        elif non_mg_ratio > 0.9:
            non_mg_cell_positions.append((cell_id, mean_pos))
        else:
            other_cells.append((cell_id, mean_pos))

    # Save instance segmentation results as image
    if instance_map and map_path is not None:
        # bg as -1
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
        # MG will be marked with white text, Non-MG with red text
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
    """ Calculate the rotation angle for long axis alignment

    Args:
        tm (np.array): target mask

    Returns:
        float: long axis angle

    """
    contours, _ = cv2.findContours(tm.astype('uint8'), 1, 2)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    rect = cv2.minAreaRect(contours[np.argmax(areas)])
    w, h = rect[1]
    ang = rect[2]
    if w < h:
        ang = ang - 90
    return ang


def process_site_instance_segmentation(site_path, 
                                       site_segmentation_path, 
                                       site_supp_files_folder):
    """ Wrapper method for instance segmentation

    Results will be saved to the supplementary data folder as:
        "cell_positions.pkl": list of cells in each frame (IDs and positions);
        "cell_pixel_assignments.pkl": pixel compositions of cells;
        "segmentation_*.png": image of instance segmentation results.
    
    Args:
        site_path (str): path to image stack (.npy)
        site_segmentation_path (str): path to semantic segmentation stack (.npy)
        site_supp_files_folder (str): path to the folder where supplementary 
            files will be saved

    """

    # TODO: Size is hardcoded here
    # Should be of size (n_time_points, 2048, 2048, 2), uint16
    image_stack = np.load(site_path)
    # Should be of size (n_time_points, 2048, 2048, n_classes), float
    segmentation_stack = np.load(site_segmentation_path)

    cell_positions = {}
    cell_pixel_assignments = {}
    for t_point in range(image_stack.shape[0]):
        print("\tClustering time %d" % t_point)
        cell_segmentation = segmentation_stack[t_point, :, :]
        instance_map_path = os.path.join(site_supp_files_folder, 'segmentation_%d.png' % t_point)
        res = instance_clustering(cell_segmentation, instance_map=True, map_path=instance_map_path)
        cell_positions[t_point] = res[0] # MG, Non-MG, Chimeric Cells
        cell_pixel_assignments[t_point] = res[1:]
    with open(os.path.join(site_supp_files_folder, 'cell_positions.pkl'), 'wb') as f:
        pickle.dump(cell_positions, f)
    with open(os.path.join(site_supp_files_folder, 'cell_pixel_assignments.pkl'), 'wb') as f:
        pickle.dump(cell_pixel_assignments, f)
    return


def process_site_extract_patches(site_path, 
                                 site_segmentation_path, 
                                 site_supp_files_folder,
                                 window_size=256):
    """ Wrapper method for patch extraction

    Supplementary files generated by `process_site_instance_segmentation` will
    be loaded for each site, then individual cells from static frames will be
    extracted and saved.

    Results will be saved in supplementary data folder, including:
        "stacks_*.pkl": single cell patches for each time slice

    Args:
        site_path (str): path to image stack (.npy)
        site_segmentation_path (str): path to semantic segmentation stack (.npy)
        site_supp_files_folder (str): path to the folder where supplementary 
            files will be saved
        window_size (int, optional): default=256, x, y size of the patch

    """

    # Load data
    image_stack = np.load(site_path)
    segmentation_stack = np.load(site_segmentation_path)
    with open(os.path.join(site_supp_files_folder, 'cell_positions.pkl'), 'rb') as f:
        cell_positions = pickle.load(f)
    with open(os.path.join(site_supp_files_folder, 'cell_pixel_assignments.pkl'), 'rb') as f:
        cell_pixel_assignments = pickle.load(f)

    for t_point in range(image_stack.shape[0]):
        if os.path.exists(os.path.join(site_supp_files_folder, 'stacks_%d.pkl' % t_point)):
            try:
                site_data = pickle.load(open(os.path.join(site_supp_files_folder, 'stacks_%d.pkl' % t_point), 'rb'))
                continue
            except Exception as e:
                print(e)
                site_data = {}
        else:
            site_data = {}
        print("\tWriting time %d" % t_point)
        raw_image = image_stack[t_point, :, :]
        cell_segmentation = segmentation_stack[t_point, :, :]
        
        positions, positions_labels = cell_pixel_assignments[t_point]
        mg_cells, non_mg_cells, other_cells = cell_positions[t_point]

        # Define fillings for the masked pixels in this slice
        background_pool = raw_image[np.where(cell_segmentation[:, :, 0] > 0.9)]
        background_pool = np.median(background_pool, 0)
        background_filling = np.ones((window_size, window_size, 1)) * background_pool.reshape((1, 1, -1))

        # Save all cells in this step, filtering will be performed during analysis
        for cell_id, cell_position in mg_cells + non_mg_cells + other_cells:
            cell_name = os.path.join(site_supp_files_folder, '%d_%d.h5' % (t_point, cell_id))
            if cell_name in site_data:
                continue
            # Define window based on cell center and extract mask
            window = [(cell_position[0]-window_size//2, cell_position[0]+window_size//2),
                      (cell_position[1]-window_size//2, cell_position[1]+window_size//2)]
            window_segmentation = select_window(cell_segmentation, window, padding=-1)
            remove_mask, tm, tm2 = generate_mask(positions, 
                                                 positions_labels, 
                                                 cell_id, 
                                                 window, 
                                                 window_segmentation)
            # Select submatrix from the whole slice
            output_mat = select_window(raw_image, window, padding=0)
            masked_output_mat = output_mat * (1 - remove_mask) + background_filling * remove_mask
            # To prevent backward compatibility issue cast to int64 and float 64 respectively
            # TODO: solve compatibility issue here
            output_mat = np.concatenate([output_mat, tm, tm2], 2).astype('int64')
            masked_output_mat = np.concatenate([masked_output_mat, tm, tm2], 2).astype('float64')
            site_data[cell_name] = {"mat": output_mat, "masked_mat": masked_output_mat}
        with open(os.path.join(site_supp_files_folder, 'stacks_%d.pkl' % t_point), 'wb') as f:
            pickle.dump(site_data, f)


def process_site_extract_patches_align_axis(site_path, 
                                            site_segmentation_path, 
                                            site_supp_files_folder,
                                            window_size=256):
    """ Wrapper method for long-axis-aligned patch extraction

    Supplementary files generated by `process_site_instance_segmentation` will
    be loaded for each site, then individual cells from static frames will be
    extracted and saved. An extra step of aligning cell long axis to x-axis is
    performed.

    Results will be saved in supplementary data folder, including:
        "stacks_rotated_*.pkl": long-axis-aligned single cell patches for 
            each time slice

    Args:
        site_path (str): path to image stack (.npy)
        site_segmentation_path (str): path to semantic segmentation stack (.npy)
        site_supp_files_folder (str): path to the folder where supplementary 
            files will be saved
        window_size (int, optional): default=256, x, y size of the patch

    """

    # Use a larger window (for rotation)
    window_size = int(np.ceil(window_size * np.sqrt(2)) + 1)
    # Load data
    image_stack = np.load(site_path)
    segmentation_stack = np.load(site_segmentation_path)
    with open(os.path.join(site_supp_files_folder, 'cell_positions.pkl'), 'rb') as f:
        cell_positions = pickle.load(f)
    with open(os.path.join(site_supp_files_folder, 'cell_pixel_assignments.pkl'), 'rb') as f:
        cell_pixel_assignments = pickle.load(f)

    for t_point in range(image_stack.shape[0]):
        site_data = {}
        print("\tWriting time %d" % t_point)
        raw_image = image_stack[t_point, :, :]
        cell_segmentation = segmentation_stack[t_point, :, :]
        
        positions, positions_labels = cell_pixel_assignments[t_point]      
        mg_cells, non_mg_cells, other_cells = cell_positions[t_point]

        # Define fillings for the masked pixels in this slice
        background_pool = raw_image[np.where(cell_segmentation[:, :, 0] > 0.9)]
        background_pool = np.median(background_pool, 0 )
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
            # Find long axis and rotate
            ang = get_cell_rect_angle(tm)
            M = cv2.getRotationMatrix2D((window_size/2, window_size/2), ang, 1)
            tm_ = cv2.warpAffine(tm.astype('uint8'), M, (window_size, window_size)).reshape((window_size, window_size, 1))
            tm2_ = cv2.warpAffine(tm2.astype('uint8'), M, (window_size, window_size)).reshape((window_size, window_size, 1))
            output_mat_ = cv2.warpAffine(output_mat.astype('uint16'), M, (window_size, window_size))
            masked_output_mat_ = cv2.warpAffine(masked_output_mat.astype('uint16'), M, (window_size, window_size))

            # Hardcoded for size to be 256 * 256
            tm = tm_[(window_size//2 - 128):(window_size//2 + 128),
                     (window_size//2 - 128):(window_size//2 + 128)]
            tm2 = tm2_[(window_size//2 - 128):(window_size//2 + 128),
                       (window_size//2 - 128):(window_size//2 + 128)]
            output_mat = output_mat_[(window_size//2 - 128):(window_size//2 + 128),
                                     (window_size//2 - 128):(window_size//2 + 128)]
            masked_output_mat = masked_output_mat_[(window_size//2 - 128):(window_size//2 + 128),
                                                   (window_size//2 - 128):(window_size//2 + 128)]
            output_mat = np.concatenate([output_mat, tm, tm2], 2).astype('int64')
            masked_output_mat = np.concatenate([masked_output_mat, tm, tm2], 2).astype('float64')
            site_data[os.path.join(site_supp_files_folder, '%d_%d.h5' % (t_point, cell_id))] = {"mat": output_mat, "masked_mat": masked_output_mat}
        with open(os.path.join(site_supp_files_folder, 'stacks_rotated_%d.pkl' % t_point), 'wb') as f:
            pickle.dump(site_data, f)
