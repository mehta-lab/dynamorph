# bchhun, {2020-02-21}

'''
for pipeline steps 1-3

'''

# pipeline:
# 1. check input: (n_frames * 2048 * 2048 * 2) channel 0 - phase, channel 1 - retardance
# 2. adjust channel range
#     a. phase: 32767 plus/minus 1600~2000
#     b. retardance: 1400~1600 plus/minus 1500~1800
# 3. save as '$SITE_NAME.npy' numpy array, dtype=uint16
# 4. run segmentation using saved model: `/data/michaelwu/CellVAE/NNSegmentation/temp_save_unsaturated/final.h5`
# 5. run instance segmentation
# 6. save individual cell patches
# 7. connect individual cells into trajectories
# 8. collect patches and assemble for VAE encoding
# 9. PCA of VAE encoded latent vectors


import numpy as np
import cv2


# 1, 2, 3
def load_raw(path, site, multipage=True):
    """
    This function takes a filepath to an EXPERIMENT
        then generates .npy files of the concatenated arrays of shape:
        (n_frames, 2048, 2048, 2): channel 0 - phase, channel 1 - retardance

    :param path: str
        path to experiment
    :param site: str
        position type ex: "C5-Site_0", or "pos0"
    :param multipage: bool
        if folder contains stabilized multipage tiffs

    :return:
    """

    if not multipage:
        raise NotImplementedError("loading non-stabilized, non-multipage tiffs not supported")

    fullpath = path+'/'+site

    multi_tif_retard = 'img__Retardance__stabilized.tif'
    multi_tif_phase = 'img_Phase2D_stabilized.tif'
    multi_tif_bf = 'img_Brightfield_computed_stabilized.tif'

    _, ret = cv2.imreadmulti(fullpath+'/'+multi_tif_retard, flags=cv2.IMREAD_ANYDEPTH)
    _, phase = cv2.imreadmulti(fullpath+'/'+multi_tif_phase, flags=cv2.IMREAD_ANYDEPTH)
    ret = np.array(ret)
    phase = np.array(phase)

    assert(ret.shape == phase.shape)

    out = np.empty(shape=ret.shape + (2,))
    out[:,:,:,0] = phase
    out[:,:,:,1] = ret

    return out


# 2
def adjust_range(arr):

    arr_adjusted = arr
    # nothing implemented right now

    return arr_adjusted


# 3
def write_raw_to_npy(path, site, output, multipage=True):
    """
    loads a single site's retardance, phase, and outputs to a single npy of the same name

    :param path:
    :param site:
    :param output:
    :param multipage:
    :return:
    """

    output_name = output + '/' + site + '.npy'

    raw = load_raw(path, site, multipage=multipage)

    raw_adjusted = adjust_range(raw)

    np.save(output_name, raw_adjusted)


# 4
def segmentation(site_path):
    '''

    :param path:
    :return:
    '''

    pass


# 5
def instance_segmentation(site_path, site_segmentation_path, site_supp_files_folder):
    '''

    :param site_path:
    :param site_segmentation_path:
    :return:
    '''
    pass


# 6
def extract_patches(site_path, site_segmentation_path, site_supp_files_folder):
    '''

    :param site_path:
    :param site_segmentation_path:
    :param site_supp_files_folder:
    :return:
    '''
    pass


# 7
def build_trajectories(site_path, site_segmentation_path, site_supp_files_folder):
    '''

    :param site_path:
    :param site_segmentation_path:
    :param site_supp_files_folder:
    :return:
    '''

    pass


# 8
def collect_patches():
    pass

