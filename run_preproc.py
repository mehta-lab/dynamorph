# bchhun, {2020-02-21}

# 1. check input: (n_frames * 2048 * 2048 * 2) channel 0 - phase, channel 1 - retardance
# 2. adjust channel range
#     a. phase: 32767 plus/minus 1600~2000
#     b. retardance: 1400~1600 plus/minus 1500~1800
# 3. save as '$SITE_NAME.npy' numpy array, dtype=uint16

from pipeline.preprocess import write_raw_to_npy
import os
import fnmatch

import argparse
from configs.config_reader import YamlReader


def main(input_, output_, config_):

    chans = config_.preprocess.channels
    multi = config_.preprocess.multipage
    z_slice = config_.preprocess.z_slice
    fovs = config_.preprocess.fov

    # build list of all sites we wish to process

    # positions are identified by subfolder names
    if config_.preprocess.pos_dir:
        if fovs is 'all':
            sites = [site for site in os.listdir(input_) if os.path.isdir(os.path.join(input_, site))]
        elif type(fovs) is list:
            sites = [site for site in os.listdir(input_) if os.path.isdir(os.path.join(input_, site)) and site in fovs]
        else:
            raise NotImplementedError("FOV subfolder expected, or preprocess FOVs must be 'all' or list of positions")

    # positions are identified by indicies
    # assume files have name structure "t###_p###_z###"
    elif not config_.preprocess.pos_dir:
        sites = {}
        all_files = [f for f in os.listdir(input_)
                     if os.path.isfile(os.path.join(input_, f)) and '_p' in f and '.tif' in f]

        if fovs is 'all':
            # for every position index in the file, assign the image to a dict key
            while all_files:
                pos = [int(p_idx.strip('p')) for p_idx in all_files[0].split('_') if 'p' in p_idx][0]
                if not pos:
                    all_files.pop(0)
                if pos in sites.keys():
                    sites[pos].append(all_files.pop(0))
                else:
                    sites[pos] = [all_files.pop(0)]

        elif type(fovs) is list:
            for fov in fovs:
                sites[fov] = fnmatch.filter(all_files, f'*p{fov:03d}*')
        else:
            raise NotImplementedError("FOV index expected, or preprocess FOVs must be 'all' or list of positions")
    else:
        raise NotImplementedError("pos_dir must be boolean True/False")

    # write sites
    for site in sites:
        if not os.path.exists(output_):
            os.makedirs(output_)

        # site represents a folder path
        if type(site) is str:
            s_list = [os.path.join(input_, f) for f in sorted(os.listdir(site))]

        # site represents a list of files filtered by position index
        elif type(site) is list:
            s_list = site
        else:
            print(f"no files found for {site}")
            continue

        write_raw_to_npy(site, s_list, output_, chans, z_slice, multipage=multi)


def parse_args():
    """
    Parse command line arguments for CLI.

    :return: namespace containing the arguments passed.
    """
    parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     '-i', '--input',
    #     type=str,
    #     required=False,
    #     help="Path to multipage-tiff file of format [t, x, y], or to single-page-tiffs",
    # )
    # parser.add_argument(
    #     '-o', '--output',
    #     type=str,
    #     required=False,
    #     help="Path to write results",
    # )
    parser.add_argument(
        '-c', '--config',
        type=str,
        required=True,
        help='path to yaml configuration file'
    )

    return parser.parse_args()


if __name__ == '__main__':
    # print(time.asctime(time.localtime(time.time())), flush=True)
    arguments = parse_args()
    config = YamlReader()
    config.read_config(arguments.config)

    for (src, target) in list(zip(config.preprocess.image_dirs, config.preprocess.target_dirs)):
        main(src, target, config)

    # print(time.asctime(time.localtime(time.time())), flush=True)

