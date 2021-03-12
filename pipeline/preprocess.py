# bchhun, {2020-02-21}

import numpy as np
import cv2
from typing import Union
import logging
log = logging.getLogger(__name__)


def read_image(file_path):
    """
    Read 2D grayscale image from file.
    Checks file extension for npy and load array if true. Otherwise
    reads regular image using OpenCV (png, tif, jpg, see OpenCV for supported
    files) of any bit depth.
    :param str file_path: Full path to image
    :return array im: 2D image
    :raise IOError if image can't be opened
    """
    if file_path[-3:] == 'npy':
        im = np.load(file_path)
    else:
        im = cv2.imread(file_path, cv2.IMREAD_ANYDEPTH)
        if im is None:
            raise IOError('Image "{}" cannot be found.'.format(file_path))
    return im


def load_raw(fullpaths: list,
             chans: list,
             z_slice: int,
             multipage: bool = True):
    """Raw data loader

    This function takes a list of paths to an experiment folder and
    loads specified site data into a numpy array.

    Output array will be of shape: (n_frames, 2048, 2048, 2), where
    channel 0 (last dimension) is phase and channel 1 is retardance

    Args:
        fullpaths (list):
            list of full paths to singlepage or multipage tiffs
        chans (list):
            list of strings corresponding to channel names
        z_slice: (int)
            specific slice to extract if multiple exist
        multipage (bool, optional): default=True
            if folder contains stabilized multipage tiffs
            only multipage tiff is supported now

    Returns:
        np.array: numpy array as described above

    """

    # store list of every image shape in the dataset for later validation
    shapes = []

    if not multipage:
        log.info(f"single-page tiffs specified")
        # load singlepage tiffs.  String parse assuming time series and z### format
        for chan in chans:
            # files maps (key:value) = (z_index, t_y_x array)
            # files = []
            # for z in z_indicies:
            #     files.append([c for c in sorted(os.listdir(fullpath)) if chan in c and f"z{z:03d}" in c])
            # files = np.array(files).flatten()
            files = [c for c in fullpaths if chan in c.split('/')[-1] and f"z{z_slice:03d}" in c.split('/')[-1]]
            files = sorted(files)
            if not files:
                log.warning(f"no files with {chan} identified")
                continue

            # resulting shapes are in (t, y, x) order
            if "Phase" in chan:
                phase = np.stack([read_image(f) for f in files])
                # phase = phase.reshape((len(z_indicies), -1, phase.shape[-2], phase.shape[-1]))
                shapes.append(phase.shape)
            elif "Retardance" in chan:
                ret = np.stack([read_image(f) for f in files])
                # ret = ret.reshape((len(z_indicies), -1, ret.shape[-2], ret.shape[-1]))
                shapes.append(ret.shape)
            elif "Brightfield" in chan:
                bf = np.stack([read_image(f) for f in files])
                # bf = bf.reshape((len(z_indicies), -1, bf.shape[-2], bf.shape[-1]))
                shapes.append(bf.shape)
            else:
                log.warning(f'not implemented: {chan} parse from single page files')

    else:
        log.info(f"multi-page tiffs specified")
        # load stabilized multipage tiffs.
        for chan in chans:
            files = [c for c in fullpaths if chan in c.split('/')[-1] and '.tif' in c.split('/')[-1]]
            files = sorted(files)
            if not files:
                log.warning(f"no files with {chan} identified")
                continue
            if len(files) > 1:
                log.warning(f"duplicate matches for channel name in folder, skipping channel")
                continue

            if "Phase" in chan:
                # multi_tif_phase = 'img_Phase2D_stabilized.tif'
                _, phase = cv2.imreadmulti(files[0],
                                           flags=cv2.IMREAD_ANYDEPTH)
                phase = np.array(phase)
                shapes.append(phase.shape)
            if "Retardance" in chan:
                # multi_tif_retard = 'img__Retardance__stabilized.tif'
                _, ret = cv2.imreadmulti(files[0],
                                         flags=cv2.IMREAD_ANYDEPTH)
                ret = np.array(ret)
                shapes.append(ret.shape)
            if "Brightfield" in chan:
                # multi_tif_bf = 'img_Brightfield_computed_stabilized.tif'
                _, bf = cv2.imreadmulti(files[0],
                                        flags=cv2.IMREAD_ANYDEPTH)
                bf = np.array(bf)
                shapes.append(bf.shape)

    # check that all shapes are the same
    assert shapes.count(shapes[0]) == len(shapes)

    # insert images into a composite array.  Composite always has 3 channels
    n_frame, x_size, y_size = shapes[0][:3]
    out = np.empty(shape=(n_frame, 3, 1, x_size, y_size))
    log.info(f"writing channels ({chans}) to composite array")
    for chan in chans:
        try:
            if "Phase" in chan:
                out[:, 0, 0] = phase
            if "Retardance" in chan:
                out[:, 1, 0] = ret
            if "Brightfield" in chan:
                out[:, 2, 0] = bf
        except UnboundLocalError:
            log.warning('variable referenced before assignment')

    return out


def adjust_range(arr):
    """Check value range for both channels
    *** currently does nothing but report mean and std ***
    *** image z-scoring is done at a later stage ***

    To maintain stability, input arrays should be within:
        phase channel: mean - 32767, std - 1600~2000
        retardance channel: mean - 1400~1600, std ~ 1500~1800

    Args:
        arr (np.array):
            input data array

    Returns:
        np.array: numpy array with value range adjusted

    """
    log.info(f"z scoring data")

    mean_c0 = arr[:, 0, 0].mean()
    mean_c1 = arr[:, 1, 0].mean()
    mean_c2 = arr[:, 2, 0].mean()
    std_c0 = arr[:, 0, 0].std()
    std_c1 = arr[:, 1, 0].std()
    std_c2 = arr[:, 2, 0].std()
    log.info("\tPhase: %d plus/minus %d" % (mean_c0, std_c0))
    log.info("\tRetardance: %d plus/minus %d" % (mean_c1, std_c1))
    log.info("\tBrightfield: %d plus/minus %d" % (mean_c2, std_c2))
    #TODO: manually adjust range if input doesn't satisfy
    return arr


def write_raw_to_npy(site: Union[int, str],
                     site_list: list,
                     output: str,
                     chans: list,
                     z_slice: int,
                     multipage: bool = True):
    """Wrapper method for data loading

    This function takes a path to an experiment folder, loads specified 
    site data into a numpy array, and saves it under specified output path.

    Args:
        site: (int or str)
            name of specific position/site being processed
        site_list (list):
            list of files for this position/site
        output (str):
            path to the output folder
        chans (list):
            list of strings corresponding to channel names
        z_slice (int):
            specific z slice to stack
        multipage (bool, optional): default=True
            if folder contains stabilized multipage tiffs
            only multipage tiff is supported now

    """

    raw = load_raw(site_list, chans, z_slice=z_slice, multipage=multipage)
    raw_adjusted = adjust_range(raw)

    output_name = output + '/' + str(site) + '.npy'
    log.info(f"saving image stack to {output_name}")
    np.save(output_name, raw_adjusted)
    return
