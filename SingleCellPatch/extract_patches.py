#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:43:41 2019

@author: zqwu
"""

import cv2
import numpy as np
import os
import matplotlib

matplotlib.use('AGG')
import matplotlib.pyplot as plt
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
    if padding is None and ((window[0][0] < 0) or
                            (window[1][0] < 0) or
                            (window[0][1] > 2048) or
                            (window[1][1] > 2048)):
        return None

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

def check_segmentation_dim(segmentation):
    """ Check segmentation mask dimension. Add a background channel if n(channels)==1
    Args:
        segmentation: (np.array): segmentation mask for the frame

    Returns:

    """
    segmentation = np.squeeze(segmentation)
    # binary segmentation has only foreground channel, add background channel
    if segmentation.ndim == 2:
        segmentation = np.stack([1 - segmentation, segmentation], axis=-1)
    # assueming the first channel to be background for multi-class segmentation
    elif segmentation.ndim == 3:
        pass
    else:
        raise ValueError('segmentation mask can only be 2 or 3 dimension, not {}'.
                         format(segmentation.ndim))
    return segmentation


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


def process_site_extract_patches(site_path,
                                 site_segmentation_path, 
                                 site_supp_files_folder,
                                 window_size=256,
                                 save_fig=False):
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
        site_data = {}
        print("\tWriting time %d" % t_point)
        raw_image = image_stack[t_point]
        cell_segmentation = segmentation_stack[t_point]
        cell_segmentation = check_segmentation_dim(cell_segmentation)
        positions, positions_labels = cell_pixel_assignments[t_point]
        mg_cells, non_mg_cells, other_cells = cell_positions[t_point]
        all_cells = mg_cells + non_mg_cells + other_cells
        # Define fillings for the masked pixels in this slice
        background_pool = raw_image[np.where(cell_segmentation[:, :, 0] > 0.9)]
        background_pool = np.median(background_pool, 0)
        background_filling = np.ones((window_size, window_size, 1)) * background_pool.reshape((1, 1, -1))
        cells_to_keep = []
        # Save all cells in this step, filtering will be performed during analysis
        for cell_id, cell_position in all_cells:
            cell_name = os.path.join(site_supp_files_folder, '%d_%d.h5' % (t_point, cell_id))
            if cell_name in site_data:
                continue
            # Define window based on cell center and extract mask
            window = [(cell_position[0]-window_size//2, cell_position[0]+window_size//2),
                      (cell_position[1]-window_size//2, cell_position[1]+window_size//2)]
            # window_segmentation = select_window(cell_segmentation, window, padding=-1)
            window_segmentation = select_window(cell_segmentation, window, padding=None)
            if window_segmentation is None:
                continue
            # only keep the cells that has patches
            cells_to_keep.append(cell_id)
            remove_mask, tm, tm2 = generate_mask(positions, 
                                                 positions_labels, 
                                                 cell_id, 
                                                 window, 
                                                 window_segmentation)
            # Select submatrix from the whole slice
            # output_mat = select_window(raw_image, window, padding=0)
            output_mat = select_window(raw_image, window, padding=None)
            masked_output_mat = output_mat * (1 - remove_mask) + background_filling * remove_mask
            # To prevent backward compatibility issue cast to int64 and float 64 respectively
            # TODO: solve compatibility issue here
            output_mat = np.concatenate([output_mat, tm, tm2], 2).astype('int64')
            masked_output_mat = np.concatenate([masked_output_mat, tm, tm2], 2).astype('float64')
            site_data[cell_name] = {"mat": output_mat, "masked_mat": masked_output_mat}
            if save_fig:
                im_phase = output_mat[:, :, 0]
                im_phase_masked = masked_output_mat[:, :, 0]
                # replace zero-padding with min for display
                im_phase[im_phase == 0] = np.nanmin(im_phase[im_phase != 0])
                im_phase_masked[im_phase_masked == 0] = np.nanmin(im_phase_masked[im_phase_masked != 0])
                im_phase = im_adjust(im_phase)
                im_phase_masked = im_adjust(im_phase_masked)
                n_rows = 2
                n_cols = 2
                fig, ax = plt.subplots(n_rows, n_cols, squeeze=False)
                ax = ax.flatten()
                fig.set_size_inches((15, 5 * n_rows))
                axis_count = 0
                for im, name in zip([im_phase, im_phase_masked, tm, tm2],
                                    ['output_mat', 'masked_output_mat', 'tm', 'tm2']):
                    ax[axis_count].imshow(np.squeeze(im), cmap='gray')
                    ax[axis_count].axis('off')
                    ax[axis_count].set_title(name, fontsize=12)
                    axis_count += 1
                fig.savefig(os.path.join(site_supp_files_folder, 'patch_t%d_id%d.jpg' % (t_point, cell_id)),
                            dpi=300, bbox_inches='tight')
                plt.close(fig)
        # remove cells that don't have patches, update cell_positions
        cell_positions_t = []
        for cells in [mg_cells, non_mg_cells, other_cells]:
            cells = [cell for cell in cells if cell[0] in cells_to_keep]
            cell_positions_t.append(cells)
        cell_positions[t_point] = cell_positions_t
        with open(os.path.join(site_supp_files_folder, 'stacks_%d.pkl' % t_point), 'wb') as f:
            pickle.dump(site_data, f)
        with open(os.path.join(site_supp_files_folder, 'cell_positions.pkl'), 'wb') as f:
            pickle.dump(cell_positions, f)


def im_bit_convert(im, bit=16, norm=False, limit=[]):
    im = im.astype(np.float32, copy=False) # convert to float32 without making a copy to save memory
    if norm:
        if not limit:
            limit = [np.nanmin(im[:]), np.nanmax(im[:])] # scale each image individually based on its min and max
        im = (im-limit[0])/(limit[1]-limit[0])*(2**bit-1)
    im = np.clip(im, 0, 2**bit-1) # clip the values to avoid wrap-around by np.astype
    if bit==8:
        im = im.astype(np.uint8, copy=False) # convert to 8 bit
    else:
        im = im.astype(np.uint16, copy=False) # convert to 16 bit
    return im

def im_adjust(img, tol=1, bit=8):
    """
    Adjust contrast of the image

    """
    limit = np.percentile(img, [tol, 100 - tol])
    im_adjusted = im_bit_convert(img, bit=bit, norm=True, limit=limit.tolist())
    return im_adjusted


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
