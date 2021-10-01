import numpy as np
import os
import argparse
import tifffile as tf

from skimage.filters import threshold_multiotsu
from skimage.measure import regionprops, label
from skimage.morphology import disk, ball, binary_opening, binary_erosion
import skimage.exposure as exposure
import skimage.filters as filters

"""
python binary_from_fluorescence.py 
-i /Users/bryant.chhun/Desktop/Data/learningCellState/microglia/2020-10-29/FLUOR/10222020_MG_MGCoculture-x2_FLUOR_A2_1 
-o /Users/bryant.chhun/Desktop/Data/learningCellState/microglia/2020-10-29/target_binary 
-c 1 -dx 6 -dy 11 -m unimodal

python binary_from_fluorescence.py 
-i /Users/bryant.chhun/Desktop/Data/learningCellState/microglia/2020-10-29/FLUOR/10222020_MG_MGCoculture-x2_FLUOR_B2_1 
-o /Users/bryant.chhun/Desktop/Data/learningCellState/microglia/2020-10-29/target_binary 
-c 2 -dx 6 -dy 11 -m multiotsu

python binary_from_fluorescence.py 
-i /Users/bryant.chhun/Desktop/Data/learningCellState/microglia/2020-10-29/FLUOR/10222020_MG_MGCoculture-x2_FLUOR_C2_1 
-o /Users/bryant.chhun/Desktop/Data/learningCellState/microglia/2020-10-29/target_binary 
-c 2 -dx 6 -dy 11 -m multiotsu
"""


def in_range(num, ref, delta):
    """
    given num, return boolean if num is within delta of ref
    :param num:
    :param ref:
    :param delta:
    :return:
    """
    return ref-delta<num < ref+delta


def filter_area(prop, tot_pix):
    """
    given list of regionprops, return list of regionprops whose areas are less than tot_pix
    :param prop:
    :param tot_pix:
    :return:
    """
    out = []
    for p in prop:
        if p.area >= tot_pix:
            out.append(p)
    return out


def shift_image(im, delta_=(0,0)):
    """

    :param im:
    :param delta_:
    :return:
    """
    blank = np.zeros(shape=im.shape)
    blank[delta_[1]:, delta_[0]:] = im[:-delta_[1], :-delta_[0]]
    return blank.astype('uint16')


def get_unimodal_threshold(input_image):
    """Determines optimal unimodal threshold

    https://users.cs.cf.ac.uk/Paul.Rosin/resources/papers/unimodal2.pdf
    https://www.mathworks.com/matlabcentral/fileexchange/45443-rosin-thresholding

    :param np.array input_image: generate mask for this image
    :return float best_threshold: optimal lower threshold for the foreground
     hist
    """

    hist_counts, bin_edges = np.histogram(
        input_image,
        bins=256,
        range=(input_image.min(), np.percentile(input_image, 99.5))
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # assuming that background has the max count
    max_idx = np.argmax(hist_counts)
    int_with_max_count = bin_centers[max_idx]
    p1 = [int_with_max_count, hist_counts[max_idx]]

    # find last non-empty bin
    pos_counts_idx = np.where(hist_counts > 0)[0]
    last_binedge = pos_counts_idx[-1]
    p2 = [bin_centers[last_binedge], hist_counts[last_binedge]]

    best_threshold = -np.inf
    max_dist = -np.inf
    for idx in range(max_idx, last_binedge, 1):
        x0 = bin_centers[idx]
        y0 = hist_counts[idx]
        a = [p1[0] - p2[0], p1[1] - p2[1]]
        b = [x0 - p2[0], y0 - p2[1]]
        cross_ab = a[0] * b[1] - b[0] * a[1]
        per_dist = np.linalg.norm(cross_ab) / np.linalg.norm(a)
        if per_dist > max_dist:
            best_threshold = x0
            max_dist = per_dist
    assert best_threshold > -np.inf, 'Error in unimodal thresholding'
    return best_threshold


def create_unimodal_mask(input_image, str_elem_size=3):
    """Create a mask with unimodal thresholding and morphological operations

    unimodal thresholding seems to oversegment, erode it by a fraction

    :param np.array input_image: generate masks from this image
    :param int str_elem_size: size of the structuring element. typically 3, 5
    :return: mask of input_image, np.array
    """

    if np.min(input_image) == np.max(input_image):
        thr = np.unique(input_image)
    else:
        thr = get_unimodal_threshold(input_image)
    if len(input_image.shape) == 2:
        str_elem = disk(str_elem_size)
    else:
        str_elem = ball(str_elem_size)
    # remove small objects in mask
    thr_image = binary_opening(input_image > thr, str_elem)
    mask = binary_erosion(thr_image, str_elem)
    return mask.astype('uint16')


def multi_otsu_pipeline(image, gamma_=0.1, sigma_=1, classes_=3, nbins_=512, min_area_=35, joint_tolerance_=30):
    """

    :param image:
    :param gamma_:
    :param sigma_:
    :param classes_:
    :param nbins_:
    :param min_area_:
    :param joint_tolerance_:
    :return:
    """

    # ===== PRE FILTERS ============== #
    image = exposure.adjust_gamma(image, gamma_)
    image = filters.gaussian(image, sigma=sigma_)

    # ===== MULTI OTSU THRESHOLDING == #
    thr = threshold_multiotsu(image, classes=classes_, nbins=nbins_)
    im_label = np.digitize(image, bins=thr)
    mask0 = im_label == 0
    mask1 = im_label == 1
    mask2 = im_label == 2

    # == generate region props for each structure in mask1 and mask2
    label1 = label(mask1)
    label2 = label(mask2)
    prop1 = regionprops(label1, intensity_image=image)
    prop2 = regionprops(label2, intensity_image=image)

    # filter region props by size
    prop1 = filter_area(prop1, min_area_)
    prop2 = filter_area(prop2, min_area_)

    # extract only the centroids of the region props ** this method is faster than calling props directly
    # many
    p1_centroids = {(int(np.round(p.centroid[0])), int(np.round(p.centroid[1]))): p for p in prop1}

    # fewer
    p2_centroids = {(int(np.round(p.centroid[0])), int(np.round(p.centroid[1]))): p for p in prop2}

    # ======= COMMON STRUCTURE PROPS =========
    # use centroids as a proxy to determine common structures
    region = joint_tolerance_
    p1_nearest = []
    for p2 in p2_centroids:
        for p1 in p1_centroids:
            if in_range(p1[0], p2[0], region) and in_range(p1[1], p2[1], region):
                p1_nearest.append(p1_centroids[p1])
    print(f"found {len(p1_nearest)} matches")
    # ========================================

    # build target images from "common structure" approach
    blank1 = np.zeros(shape=image.shape)
    blank2 = np.zeros(shape=image.shape)

    for idx, p1 in enumerate(p1_nearest):
        (minrow, mincol, maxrow, maxcol) = p1_nearest[idx].bbox
        blank1[minrow:maxrow, mincol:maxcol] = p1_nearest[idx].intensity_image

    for idx, p2 in enumerate(prop2):
        (minrow, mincol, maxrow, maxcol) = prop2[idx].bbox
        blank2[minrow:maxrow, mincol:maxcol] = prop2[idx].intensity_image

    combined = blank1 + blank2
    combined[combined > 0] = 1

    return combined.astype('uint16')


def main(arguments_):
    src = arguments_.input
    dst = arguments_.output
    chan = arguments_.channel
    method = arguments_.method
    delta = (arguments_.deltax, arguments_.deltay)

    positions = [pos for pos in os.listdir(src) if "Site" in pos]

    for site in positions:
        sitepath = os.path.join(src, site)
        fluor = sorted([file for file in os.listdir(sitepath) if f"channel{chan:03d}" in file])

        # if index=2 is the focal plane
        # image = tf.imread(os.path.join(sitepath, fluor[2]))

        # max-intensity z-projection of full stack
        image_stack = []
        for f in sorted(fluor):
            image_stack.append(tf.imread(os.path.join(sitepath, f)))
        image_stack = np.stack(image_stack)
        image = np.max(image_stack, axis=0)

        if method == 'multiotsu':
            # multi-otsu segmentation
            segmentation = multi_otsu_pipeline(image)
        elif method == 'unimodal':
            segmentation = create_unimodal_mask(image, str_elem_size=1)
        else:
            raise AttributeError("method flag is not supplied or not implemented")

        # shift the result to align with reference data
        if delta is not (None, None):
            segmentation = shift_image(segmentation, delta)

        # write files
        dstpath = os.path.join(dst, site)
        if not os.path.isdir(dstpath):
            os.makedirs(dstpath)
        dstfile = os.path.join(dstpath, fluor[0][:-4]+"_segmentation")
        tf.imsave(dstfile+".tif", segmentation)


def parse_args():
    """
    Parse command line arguments for CLI.

    :return: namespace containing the arguments passed.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help="Path to experiment folder that contains multiple positions",
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help="Path to write results",
    )
    parser.add_argument(
        '-m', '--method',
        type=str,
        required=True,
        help="segmentation method, either 'unimodal' or 'multiotsu'",
    )
    # channel idx.  ch= {0,1,2,3,} = {405, 488, 561, 647}
    parser.add_argument(
        '-c', '--channel',
        type=int,
        required=True,
        help="channel index to segment on",
    )
    parser.add_argument(
        '-dx', '--deltax',
        type=int,
        default=None,
        required=False,
        help="if there is a shift between these masks and reference data, shift by dx in PIXELS",
    )
    parser.add_argument(
        '-dy', '--deltay',
        type=int,
        default=None,
        required=False,
        help="if there is a shift between these masks and reference data, shift by dy in PIXELS",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    main(arguments)
