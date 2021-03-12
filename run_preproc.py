
# 1. check input: (n_frames * 2048 * 2048 * 2) channel 0 - phase, channel 1 - retardance
# 2. adjust channel range
#     a. phase: 32767 plus/minus 1600~2000
#     b. retardance: 1400~1600 plus/minus 1500~1800
# 3. save as '$SITE_NAME.npy' numpy array, dtype=uint16

from pipeline.preprocess import write_raw_to_npy
import os
import fnmatch
import re

import argparse
from configs.config_reader import YamlReader
import logging
log = logging.getLogger(__name__)


def main(input_, output_, config_):
    """
    Using supplied config file parameters, prepare specified datasets for downstream analysis

    :param input_: str
        Path to a single experiment
    :param output_: str
        Path to output directory for prepared datasets
    :param config_: YamlReader
        YamlReader object containing parsed configuration values
    :return:
    """

    chans = config_.preprocess.channels
    multi = config_.preprocess.multipage
    z_slice = config_.preprocess.z_slice
    fovs = config_.preprocess.fov

    # === build list or dict of all sites we wish to process ===

    # positions are identified by subfolder names
    if config_.preprocess.pos_dir:
        log.info("pos dir, identifying all subfolders")
        if fovs == 'all':
            sites = [site for site in os.listdir(input_) if os.path.isdir(os.path.join(input_, site))]
        elif type(fovs) is list:
            sites = [site for site in os.listdir(input_) if os.path.isdir(os.path.join(input_, site)) and site in fovs]
        else:
            raise NotImplementedError("FOV subfolder expected, or preprocess FOVs must be 'all' or list of positions")

    # positions are identified by indicies
    # assume files have name structure "t###_p###_z###"
    elif not config_.preprocess.pos_dir:
        log.info("no pos dir, identifiying all files")
        sites = {}
        all_files = [f for f in os.listdir(input_)
                     if os.path.isfile(os.path.join(input_, f)) and '_p' in f and '.tif' in f]

        if fovs == 'all':
            log.info("fovs = all, looping ")
            # for every position index in the file, assign the image to a dict key
            while all_files:
                pos = [int(p_idx.strip('p')) for p_idx in all_files[0].split('_') if 'p' in p_idx][0]
                if not pos:
                    all_files.pop(0)
                if pos in sites.keys():
                    sites[pos].append(os.path.join(input_, all_files.pop(0)))
                else:
                    sites[pos] = [os.path.join(input_, all_files.pop(0))]

        elif type(fovs) is list:
            for fov in fovs:
                sites[fov] = [os.path.join(input_, f) for f in sorted(fnmatch.filter(all_files, f'*p{fov:03d}*'))]
        else:
            raise NotImplementedError("FOV index expected, or preprocess FOVs must be 'all' or list of positions")
    else:
        raise NotImplementedError("pos_dir must be boolean True/False")

    # write sites
    for site in sorted(sites):
        if not os.path.exists(output_):
            os.makedirs(output_)

        # site represents a position folder
        if type(site) is str:
            s_list = [os.path.join(input_, site, f) for f in sorted(os.listdir(os.path.join(input_, site)))]

        # site represents a position index
        elif type(site) is int:
            s_list = sites[site]
        else:
            log.warning(f"no files found for position = {site}")
            continue

        write_raw_to_npy(site, s_list, output_, chans, z_slice, multipage=multi)


def parse_args():
    """
    Parse command line arguments for CLI.

    :return: namespace containing the arguments passed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config',
        type=str,
        required=True,
        help='path to yaml configuration file'
    )

    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_args()
    config = YamlReader()
    config.read_config(arguments.config)

    for (src, target) in list(zip(config.preprocess.image_dirs, config.preprocess.target_dirs)):
        main(src, target, config)

