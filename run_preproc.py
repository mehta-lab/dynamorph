# bchhun, {2020-02-21}

# 1. check input: (n_frames * 2048 * 2048 * 2) channel 0 - phase, channel 1 - retardance
# 2. adjust channel range
#     a. phase: 32767 plus/minus 1600~2000
#     b. retardance: 1400~1600 plus/minus 1500~1800
# 3. save as '$SITE_NAME.npy' numpy array, dtype=uint16

from pipeline.preprocess import write_raw_to_npy
import os
import time

import argparse


def main(arguments_):

    path = arguments_.input
    outputs = arguments_.output

    if arguments_.sites:
        sites = arguments_.sites
    else:
        # assume all subdirectories are site/FOVs
        sites = [site for site in os.listdir(path) if os.path.isdir(os.path.join(path, site))]

    for site in sites:
        if not os.path.exists(outputs):
            os.makedirs(outputs)

        out = outputs

        try:
            print(f"writing {site} to {out}", flush=True)
            write_raw_to_npy(path, site, out, multipage=True)
        except Exception as e:
            print(f"\terror in writing {site}", flush=True)


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
        help="Path to multipage-tiff file of format [t, x, y]",
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help="Path to write results",
    )
    # sites argument is a list of strings
    parser.add_argument(
        '-s', '--sites',
        type=lambda s: [str(item.strip(' ').strip("'")) for item in s.split(',')],
        required=False,
        help="list of field-of-views to process (subfolders in raw data directory)",
    )
    return parser.parse_args()


if __name__ == '__main__':
    print(time.asctime(time.localtime(time.time())), flush=True)
    arguments = parse_args()
    main(arguments)
    print(time.asctime(time.localtime(time.time())), flush=True)

