# bchhun, {2020-02-21}

from pipeline.patch_VAE import extract_patches, build_trajectories
from multiprocessing import Pool, Queue, Process
import os
from run_preproc import sites_NOVEMBER, sites_JANUARY, sites_JANUARY_FAST
import numpy as np
import argparse

# ESS from hulk

# SITES = ['B4-Site_0', 'B4-Site_1',  'B4-Site_2',  'B4-Site_3',  'B4-Site_4', 'B4-Site_5', 'B4-Site_6', 'B4-Site_7', 'B4-Site_8',
#          'B5-Site_0', 'B5-Site_1',  'B5-Site_2',  'B5-Site_3',  'B5-Site_4', 'B5-Site_5', 'B5-Site_6', 'B5-Site_7', 'B5-Site_8',
#          'C3-Site_0', 'C3-Site_1',  'C3-Site_2',  'C3-Site_3',  'C3-Site_4', 'C3-Site_5', 'C3-Site_6', 'C3-Site_7', 'C3-Site_8',
#          'C4-Site_0', 'C4-Site_1',  'C4-Site_2',  'C4-Site_3',  'C4-Site_4', 'C4-Site_5', 'C4-Site_6', 'C4-Site_7', 'C4-Site_8',
#          'C5-Site_0', 'C5-Site_1',  'C5-Site_2',  'C5-Site_3',  'C5-Site_4', 'C5-Site_5', 'C5-Site_6', 'C5-Site_7', 'C5-Site_8']

#SITES_ctrl = ['C5-Site_0', 'C5-Site_4']
#SITES_GBM = ['B2-Site_0', 'B2-Site_4']
#SITES_IL17 = ['B4-Site_0', 'B4-Site_4']
#SITES_IFbeta=['B5-Site_0', 'B5-Site_4']
#ITES_fast = ['C5-Site_0', 'C5-Site_4']

RAW_NOVEMBER = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/NOVEMBER/raw'
RAW_JANUARY = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/JANUARY/raw'
RAW_JANUARY_FAST = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/JANUARY_FAST/raw'

INTERMEDIATE_NOVEMBER = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/NOVEMBER/supp'
INTERMEDIATE_JANUARY = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/JANUARY/supp'
INTERMEDIATE_JANUARY_FAST = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/JANUARY_FAST/supp'

#TARGET = '/gpfs/CompMicro/Projects/learningCellState/microglia/segmentation_experiments/expt_001'
TARGET = '/data_sm/home/michaelwu/VALIDATION'


class Worker(Process):
    def __init__(self, inputs, gpuid=0, method='extract_patches'):
        super().__init__()
        self.gpuid = gpuid
        self.inputs = inputs
        self.method = method

    def run(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpuid)

        if self.method == 'extract_patches':
            extract_patches(self.inputs)
        elif self.method == 'build_trajectories':
            build_trajectories(self.inputs)


def main(arguments):

    if not arguments.input or arguments.output:
        print('no input or output supplied, using hard coded paths')
        # loads 'Site.npy',
        #       '_NNProbabilities.npy',
        #       '/Site-supps/Site/cell_positions.pkl',
        #       '/Site-supps/site/cell_pixel_assignments.pkl',

        # generates 'stacks_%d.pkl' % timepoint

        # prints: "writing time %d"
        n_gpu = 1
        for sites, inputs, outputs in zip([sites_NOVEMBER, sites_JANUARY, sites_JANUARY_FAST],
                                [RAW_NOVEMBER, RAW_JANUARY, RAW_JANUARY_FAST],
                                [INTERMEDIATE_NOVEMBER, INTERMEDIATE_JANUARY, INTERMEDIATE_JANUARY_FAST]):

            # if probabilities and formatted stack exist
            segment_sites = [site for site in sites if os.path.exists(os.path.join(inputs, "%s.npy" % site)) and \
                                         os.path.exists(os.path.join(inputs, "%s_NNProbabilities.npy" % site))]

            # process each site on a different GPU if using multi-gpu
            sep = np.linspace(0, len(segment_sites), n_gpu+1).astype(int)

            processes = []
            for i in range(n_gpu):
                _sites = segment_sites[sep[i]:sep[i+1]]
                args = (inputs, outputs, TARGET, _sites)
                p = Worker(args, gpuid=i)
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

        # *** NOT USED WITH VAE ***
        # *** USED IN POST-PCA TRAJ MATCHING ***
        # loads 'cell_positions.pkl', 'cell_pixel_assignments.pkl'
        # generates 'cell_traj.pkl'
    else:
        print("CLI arguments provided")
        inputs = arguments.input
        outputs = arguments.output

        # results are written to subfolder "supp"
        outputs = os.path.join(outputs, "supp")
        if not os.path.isdir(outputs):
            os.mkdir(outputs)

        n_gpu = arguments.gpus
        method = arguments.method

        if arguments.sites:
            sites = arguments.sites
        else:
            sites = [site for site in os.listdir(inputs) if os.path.isdir(os.path.join(inputs, site))]

        # if probabilities and formatted stack exist
        segment_sites = [site for site in sites if os.path.exists(os.path.join(inputs, "%s.npy" % site)) and \
                         os.path.exists(os.path.join(inputs, "%s_NNProbabilities.npy" % site))]

        # process each site on a different GPU if using multi-gpu
        sep = np.linspace(0, len(segment_sites), n_gpu + 1).astype(int)

        # TARGET is never used in either extract_patches or build_trajectory
        processes = []
        for i in range(n_gpu):
            _sites = segment_sites[sep[i]:sep[i + 1]]
            args = (inputs, outputs, TARGET, _sites)
            p = Worker(args, gpuid=i, method=method)
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


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
    parser.add_argument(
        '-m', '--method',
        type=str,
        required=False,
        choices=['extract_patches', 'build_trajectories'],
        default='extract_patches',
        help="Method: one of 'extract_patches', 'build_trajectories'",
    )
    parser.add_argument(
        '-g', '--gpus',
        type=int,
        required=False,
        default=1,
        help="Number of GPS to use",
    )
    parser.add_argument(
        '-s', '--sites',
        type=lambda s: [str(item.strip(' ').strip("'")) for item in s.split(',')],
        required=False,
        help="list of field-of-views to process (subfolders in raw data directory)",
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
