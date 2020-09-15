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

# Paths of RAW experiment data (ESS from hulk)
input_NOVEMBER = '/gpfs/CompMicro/projects/dynamorph/microglia/20191107_1209_1_GW23/blank_bg_stabilized'
input_JANUARY = '/gpfs/CompMicro/rawdata/hummingbird/Processed/Galina/2020_01_28/SM_GW22_2020_0128_1404_1_SM_GW22_2020_0128_1404_1/blank_bg_stabilized'
input_JANUARY_FAST = '/gpfs/CompMicro/rawdata/hummingbird/Processed/Galina/2020_01_28/SM_GW22_2020_0128_1143_2hr_fastTimeSeries_1_SM_GW22_2020_0128_1143_2hr_fastTimeSeries_1/blank_bg_stabilized'


# Sites for each experiment
sites_NOVEMBER = [
    'B2-Site_0', 'B2-Site_1',  'B2-Site_2',  'B2-Site_3',  'B2-Site_4', 'B2-Site_5', 'B2-Site_6', 'B2-Site_7', 'B2-Site_8',
    'B4-Site_0', 'B4-Site_1',  'B4-Site_2',  'B4-Site_3',  'B4-Site_4', 'B4-Site_5', 'B4-Site_6', 'B4-Site_7', 'B4-Site_8',
    'B5-Site_0', 'B5-Site_1',  'B5-Site_2',  'B5-Site_3',  'B5-Site_4', 'B5-Site_5', 'B5-Site_6', 'B5-Site_7', 'B5-Site_8',
    'C4-Site_0', 'C4-Site_1',  'C4-Site_2',  'C4-Site_3',  'C4-Site_4', 'C4-Site_5', 'C4-Site_6', 'C4-Site_7', 'C4-Site_8',
    'C5-Site_0', 'C5-Site_1',  'C5-Site_2',  'C5-Site_3',  'C5-Site_4', 'C5-Site_5', 'C5-Site_6', 'C5-Site_7', 'C5-Site_8'
]

sites_JANUARY = [
    'B2-Site_0', 'B2-Site_1', 'B2-Site_2', 'B2-Site_3', 'B2-Site_4', 'B2-Site_5', 'B2-Site_6', 'B2-Site_7', 'B2-Site_8',
    'B4-Site_0', 'B4-Site_1', 'B4-Site_2', 'B4-Site_3', 'B4-Site_4', 'B4-Site_5', 'B4-Site_6', 'B4-Site_7', 'B4-Site_8',
    'B5-Site_0', 'B5-Site_1', 'B5-Site_2', 'B5-Site_3', 'B5-Site_4', 'B5-Site_5', 'B5-Site_6', 'B5-Site_7', 'B5-Site_8',
    'C3-Site_0', 'C3-Site_1', 'C3-Site_2', 'C3-Site_3', 'C3-Site_4', 'C3-Site_5', 'C3-Site_6', 'C3-Site_7', 'C3-Site_8',
    'C4-Site_0', 'C4-Site_1', 'C4-Site_2', 'C4-Site_3', 'C4-Site_4', 'C4-Site_5', 'C4-Site_6', 'C4-Site_7', 'C4-Site_8',
    'C5-Site_0', 'C5-Site_1', 'C5-Site_2', 'C5-Site_3', 'C5-Site_4', 'C5-Site_5', 'C5-Site_6', 'C5-Site_7', 'C5-Site_8'
]

sites_JANUARY_FAST = [
    'C5-Site_0', 'C5-Site_1', 'C5-Site_2', 'C5-Site_3', 'C5-Site_4', 'C5-Site_5', 'C5-Site_6', 'C5-Site_7', 'C5-Site_8'
]


# Output paths for each experiment
# DATA_PREP = '/gpfs/CompMicro/Hummingbird/Processed/Galina/VAE/data_temp'
# output = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation'
output_NOVEMBER = '/gpfs/CompMicro/projects/dynamorph/microglia/raw_for_segmentation/NOVEMBER/raw'
output_JANUARY = '/gpfs/CompMicro/projects/dynamorph/microglia/raw_for_segmentation/JANUARY/raw'
output_JANUARY_FAST = '/gpfs/CompMicro/projects/dynamorph/microglia/raw_for_segmentation/JANUARY_FAST/raw'


def main(arguments_):

    if not arguments_.input or not arguments_.output:
        print('no input or output supplied, using hard coded paths')

        for sites, inputs, outputs in zip([sites_NOVEMBER, sites_JANUARY, sites_JANUARY_FAST],
                                        [input_NOVEMBER, input_JANUARY, input_JANUARY_FAST],
                                        [output_NOVEMBER, output_JANUARY, output_JANUARY_FAST],
                                        ):
            for site in sites:

                if not os.path.exists(outputs):
                    os.makedirs(outputs)

                out = outputs

                try:
                    print(f"writing {site} to {out}", flush=True)
                    write_raw_to_npy(inputs, site, out, multipage=True)
                except Exception as e:
                    print(f"\terror in writing {site}", flush=True)

    else:
        path = arguments_.input
        outputs = arguments_.output

        # files are written to subfolder "raw"
        outputs = os.path.join(outputs, 'raw')
        if not os.path.isdir(outputs):
            os.makedirs(outputs, exist_ok=True)

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
        required=False,
        help="Path to multipage-tiff file of format [t, x, y]",
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=False,
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

