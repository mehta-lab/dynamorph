# bchhun, {2020-02-21}

from pipeline.segmentation import segmentation, instance_segmentation
from pipeline.segmentation_validation import segmentation_validation_bryant, segmentation_validation_michael
from run_preproc import sites_NOVEMBER, sites_JANUARY, sites_JANUARY_FAST
from multiprocessing import Pool, Queue, Process
import os
import numpy as np

import argparse

# ESS from hulk

# well = 'B2'
# exclude = [None]
# Sites1 = [f'{well}-Site_{pos}' for well, pos in
#          list(zip([well for k in range(0, 9) if k not in exclude], [k for k in range(0, 9) if k not in exclude]))]
#
# well = 'B4'
# exclude = [None]
# Sites2 = [f'{well}-Site_{pos}' for well, pos in
#          list(zip([well for k in range(0, 9) if k not in exclude], [k for k in range(0, 9) if k not in exclude]))]
#
# well = 'B5'
# exclude = [None]
# Sites3 = [f'{well}-Site_{pos}' for well, pos in
#          list(zip([well for k in range(0, 9) if k not in exclude], [k for k in range(0, 9) if k not in exclude]))]
#
# well = 'C4'
# exclude = [None]
# Sites4 = [f'{well}-Site_{pos}' for well, pos in
#          list(zip([well for k in range(0, 9) if k not in exclude], [k for k in range(0, 9) if k not in exclude]))]
#
# well = 'C5'
# exclude = [None]
# Sites5 = [f'{well}-Site_{pos}' for well, pos in
#          list(zip([well for k in range(0, 9) if k not in exclude], [k for k in range(0, 9) if k not in exclude]))]
#
# Sites = []
# Sites.extend(Sites1)
# Sites.extend(Sites2)
# Sites.extend(Sites3)
# Sites.extend(Sites4)
# Sites.extend(Sites5)

# SITES_ctrl = ['C5-Site_0', 'C5-Site_4']
# SITES_GBM = ['B2-Site_0', 'B2-Site_4']
# SITES_IL17 = ['B4-Site_0', 'B4-Site_4']
# SITES_IFbeta=['B5-Site_0', 'B5-Site_4']
# SITES_fast = ['C5-Site_0', 'C5-Site_4']

RAW_NOVEMBER = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/NOVEMBER/raw'
RAW_JANUARY = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/JANUARY/raw'
RAW_JANUARY_FAST = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/JANUARY_FAST/raw'

INTERMEDIATE_NOVEMBER = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/NOVEMBER/supp'
INTERMEDIATE_JANUARY = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/JANUARY/supp'
INTERMEDIATE_JANUARY_FAST = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/JANUARY_FAST/supp'

'''======= TARGET is the output directory for "segmentation_validation" runs ONLY ======='''
#TARGET = '/gpfs/CompMicro/Projects/learningCellState/microglia/segmentation_experiments/expt_009'
# TARGET = '/data_sm/home/michaelwu/VALIDATION'


class Worker(Process):
    def __init__(self, inputs, gpuid=0, method='segmentation'):
        super().__init__()
        self.gpuid = gpuid
        self.inputs = inputs
        self.method = method

    def run(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpuid)

        if self.method == 'segmentation':
            segmentation(self.inputs)
        elif self.method == 'instance_segmentation':
            instance_segmentation(self.inputs)
        elif self.method == 'segmentation_validation':
            segmentation_validation_michael(self.inputs, self.gpuid, 'unfiltered')


def main(arguments_):

    if not arguments_.input or not arguments_.output:
        print('no input or output supplied, using hard coded paths')
        n_gpu = 4
        TARGET = '/data_sm/home/michaelwu/VALIDATION'

        for sites, inputs, outputs in zip([sites_NOVEMBER, sites_JANUARY, sites_JANUARY_FAST],
                                [RAW_NOVEMBER, RAW_JANUARY, RAW_JANUARY_FAST],
                                [INTERMEDIATE_NOVEMBER, INTERMEDIATE_JANUARY, INTERMEDIATE_JANUARY_FAST]):
            segment_sites = [site for site in sites if os.path.exists(os.path.join(inputs, "%s.npy" % site)) and \
                                         os.path.exists(os.path.join(inputs, "%s_NNProbabilities.npy" % site))]
            sep = np.linspace(0, len(segment_sites), n_gpu+1).astype(int)
            for i in range(n_gpu):
                _sites = segment_sites[sep[i]:sep[i+1]]
                args = (inputs, outputs, TARGET, _sites)
                process = Worker(args, gpuid=i)
                process.start()
            for i in range(n_gpu):
                process.join()
    else:
        print("CLI arguments provided")
        inputs = arguments_.input
        outputs = arguments_.output

        # probabilities are written to subfolder "supp"
        outputs = os.path.join(outputs, "supp")
        if not os.path.isdir(outputs):
            os.mkdir(outputs)

        n_gpu = arguments_.gpus
        method = arguments_.method

        if method == 'segmentation_validation':
            TARGET = outputs
        else:
            TARGET = ''

        if arguments_.sites:
            sites = arguments_.sites
        else:
            sites = [site for site in os.listdir(inputs) if os.path.isdir(site)]

        segment_sites = [site for site in sites if os.path.exists(os.path.join(inputs, "%s.npy" % site)) and \
                         os.path.exists(os.path.join(inputs, "%s_NNProbabilities.npy" % site))]
        sep = np.linspace(0, len(segment_sites), n_gpu + 1).astype(int)

        processes = []
        for i in range(n_gpu):
            _sites = segment_sites[sep[i]:sep[i + 1]]
            args = (inputs, outputs, TARGET, _sites)
            process = Worker(args, gpuid=i, method=method)
            process.start()
            processes.append(process)
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
        choices=['segmentation', 'instance_segmentation', 'segmentation_validation'],
        help="Method: one of 'segmentation', 'instance_segmentation', or 'segmentation_validation'",
    )
    parser.add_argument(
        '-g', '--gpus',
        type=int,
        required=False,
        help="Number of GPS to use",
    )
    parser.add_argument(
        '-s', '--sites',
        type=list,
        required=False,
        help="list of field-of-views to process (subfolders in raw data directory)",
    )
    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_args()
    main(arguments)
