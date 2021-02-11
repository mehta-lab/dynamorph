# bchhun, {2020-02-21}

import numpy as np
import cv2
import os

def load_raw(path: str, 
             site: str, 
             multipage: bool = True):
    """Raw data loader

    This function takes a path to an experiment folder and 
    loads specified site data into a numpy array.

    Output array will be of shape: (n_frames, 2048, 2048, 2), where
    channel 0 (last dimension) is phase and channel 1 is retardance

    Args:
        path (str):
            path to the experiment folder
        site (str):
            position type ex: "C5-Site_0", "pos0", etc.
        multipage (bool, optional): default=True
            if folder contains stabilized multipage tiffs
            only multipage tiff is supported now

    Returns:
        np.array: numpy array as described above

    """
    if not multipage:
        raise NotImplementedError("loading non-stabilized, non-multipage tiffs not supported")

    fullpath = path+'/'+site

    multi_tif_retard = 'img__Retardance__stabilized.tif'
    multi_tif_phase = 'img_Phase2D_stabilized.tif'
    multi_tif_bf = 'img_Brightfield_computed_stabilized.tif'

    _, ret = cv2.imreadmulti(fullpath + '/' + multi_tif_retard, 
                             flags=cv2.IMREAD_ANYDEPTH)
    _, phase = cv2.imreadmulti(fullpath + '/' + multi_tif_phase, 
                               flags=cv2.IMREAD_ANYDEPTH)
    _, bf = cv2.imreadmulti(fullpath + '/' + multi_tif_bf, 
                            flags=cv2.IMREAD_ANYDEPTH)
    ret = np.array(ret)
    phase = np.array(phase)
    bf = np.array(bf)

    assert ret.shape == phase.shape == bf.shape

    n_frame, x_size, y_size = ret.shape[:3]
    out = np.empty(shape=(n_frame, 3, 1, x_size, y_size))
    out[:, 0, 0] = phase
    out[:, 1, 0] = ret
    out[:, 2, 0] = bf

    return out


def adjust_range(arr):
    """Check value range for both channels

    To maintain stability, input arrays should be within:
        phase channel: mean - 32767, std - 1600~2000
        retardance channel: mean - 1400~1600, std ~ 1500~1800

    Args:
        arr (np.array):
            input data array

    Returns:
        np.array: numpy array with value range adjusted

    """

    mean_c0 = arr[:, 0, 0].mean()
    mean_c1 = arr[:, 1, 0].mean()
    mean_c2 = arr[:, 2, 0].mean()
    std_c0 = arr[:, 0, 0].std()
    std_c1 = arr[:, 1, 0].std()
    std_c2 = arr[:, 2, 0].std()
    print("\tPhase: %d plus/minus %d" % (mean_c0, std_c0))
    print("\tRetardance: %d plus/minus %d" % (mean_c1, std_c1))
    print("\tBrightfield: %d plus/minus %d" % (mean_c2, std_c2))
    #TODO: manually adjust range if input doesn't satisfy
    return arr


def write_raw_to_npy(path: str, 
                     site: str, 
                     output: str, 
                     multipage: bool = True):
    """Wrapper method for data loading

    This function takes a path to an experiment folder, loads specified 
    site data into a numpy array, and saves it under specified output path.

    Args:
        path (str):
            path to the experiment folder
        site (str):
            position type ex: "C5-Site_0", "pos0", etc.
        output (str):
            path to the output folder
        multipage (bool, optional): default=True
            if folder contains stabilized multipage tiffs
            only multipage tiff is supported now

    """

    output_name = output + '/' + site + '.npy'
    raw = load_raw(path, site, multipage=multipage)
    raw_adjusted = adjust_range(raw)
    np.save(output_name, raw_adjusted)
    return
