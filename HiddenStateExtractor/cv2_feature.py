#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 09:53:46 2019

@author: michaelwu
"""
import cv2
import h5py
import numpy as np
import scipy
import pickle
import random
import os
import cmath
import matplotlib.pyplot as plt
from .naive_imagenet import preprocess, read_file_path, CHANNEL_MAX
import multiprocessing as mp


def extract_features(x, vector_size=32):
    """ Calculate KAZE features for input image

    Args:
        x (np.array): input image mat
        vector_size (int, optional): feature vector size

    Returns:
        np.array: KAZE features

    """  
    x = x.astype('uint8')
    try:
        dscs = []
        alg = cv2.KAZE_create()
        for x_slice in x:
            # finding image keypoints
            kps = alg.detect(x_slice)
            kps = sorted(kps, key=lambda x: -x.response)[:vector_size]
            # computing descriptors vector
            kps, dsc = alg.compute(x_slice, kps)
            dsc = dsc.flatten()
            # Padding
            needed_size = (vector_size * 64)
            if dsc.size < needed_size:
                dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
            dscs.append(dsc)
        dscs = np.stack(dscs, 0)
    except Exception as e:
        print('Error: ' + str(e))
        return None
    return dscs


def worker(f_n):
    """ Helper function for parallelization """
    x = preprocess(f_n, cs=[0, 1], channel_max=CHANNEL_MAX)
    y = extract_features(x, vector_size=32)
    return y


def get_size(mask):
    """ Calculate cell size based on mask
    
    Args:
        mask (np.array): segmentation mask of a single cell

    Returns:
        int: number of pixels in the cell area
        int: size of the cell contour

    """

    _, contours, _ = cv2.findContours(mask.astype('uint8'), 1, 2)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    return mask.sum(), np.max(areas)


def get_intensity_profile(dat, mask=None):
    """ Calculate peak phase/retardance values
    
    See docs of `get_size` for input details

    Args:
        dat (list): list of 2D np.arrays for each channel of the patch
        mask (np.array, optional): segmentation mask

    Returns:
        list (of tuples): list of intensity properties for each channel
            max phase intensity;
            95th percentile phase intensity;
            200-th value of top phase intensities;
            summed phase intensities.

    """
    
    output = []
    for channel_ind in range(len(dat)):
        channel_slice = dat[channel_ind]
        channel_slice = channel_slice / 65535.
        
        # bg = np.median(channel_slice[np.where(mask == 0)])
        bg = 0.

        peak_int = ((channel_slice - bg) * mask).max()
        sum_int = ((channel_slice - bg) * mask).sum()
        intensities = (channel_slice - bg)[np.where(mask)]
        quantile_int = np.percentile(intensities, 95)
        top200_int = np.mean(sorted(intensities)[-200:])

        output.append((peak_int, quantile_int, top200_int, sum_int))

    return output


# def get_aspect_ratio(dat):
#     """ Calcualte aspect ratio (cv2.minAreaRect)
      
#     This function is deprecated and should be replaced by `get_angle_apr`

#     See docs of `get_size` for input details.

#     Args:
#         dat (np.array): single cell patch

#     Returns:
#         float: width
#         float: height
#         float: angle of long axis

#     """
#     _, contours, _ = cv2.findContours(dat[:, :, 2].astype('uint8'), 1, 2)
#     areas = [cv2.contourArea(cnt) for cnt in contours]
#     rect = cv2.minAreaRect(contours[np.argmax(areas)])
#     w, h = rect[1]
#     ang = rect[2]
#     if w < h:
#         ang = ang - 90
#     return w, h, ang


def rotate_bound(image, angle):
    """ Rotate target mask

    Args:
        image (np.array): target mask of single cell patch
        angle (float): rotation angle

    Returns:
        np.array: rotated mask

    """
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


def get_angle_apr(mask):
    """ Find long axis and calcualte aspect ratio

    See docs of `get_size` for input details.

    Args:
        mask (np.array): segmentation mask of a single cell

    Returns:
        float: aspect ratio
        float: angle of long axis

    """
    y, x = np.nonzero(mask)
    x = x - np.mean(x)
    y = y - np.mean(y)
    coords = np.stack([x, y], 0)
    cov = np.cov(coords)
    evals, evecs = np.linalg.eig(cov)
    main_axis = evecs[:, np.argmax(evals)]  # Eigenvector with largest eigenvalue
    angle = cmath.polar(complex(*main_axis))[1]
      
    rotated = rotate_bound(mask, -angle/np.pi * 180)
    _, contours, _ = cv2.findContours(rotated.astype('uint8'), 1, 2)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    rect = cv2.boundingRect(contours[np.argmax(areas)])
    return rect[2], rect[3], angle


def get_aspect_ratio_no_rotation(mask):
    """ Calcualte aspect ratio of untouched target mask

    See docs of `get_size` for input details.

    Args:
        mask (np.array): segmentation mask of a single cell

    Returns:
        float: width
        float: height

    """
    _, contours, _ = cv2.findContours(mask.astype('uint8'), 1, 2)
    areas = [cv2.contourArea(cnt) for cnt in contours]
    rect = cv2.boundingRect(contours[np.argmax(areas)])
    return rect[2], rect[3]


# if __name__ == '__main__':
#     path = '/mnt/comp_micro/Projects/CellVAE'
#     fs = read_file_path(os.path.join(path, 'Data', 'StaticPatches'))
#     sites = ['D%d-Site_%d' % (i, j) for j in range(9) for i in range(3, 6)]
    
#     sizes = {}
#     densities = {}
#     aprs = {}
#     aprs_nr = {}
#     for site in sites:
#         dat = pickle.load(open('./%s_all_patches.pkl' % site, 'rb'))
#         for f in dat:
#             d = dat[f]["masked_mat"]
#             sizes[f] = get_size(d)
#             densities[f] = get_density(d)
#             aprs[f] = get_aspect_ratio(d)
#             aprs_nr[f] = get_aspect_ratio_no_rotation(d)