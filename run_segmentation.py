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

# RAW_NOVEMBER = '/gpfs/CompMicro/projects/dynamorph/microglia/raw_for_segmentation/NOVEMBER/raw'
# RAW_JANUARY = '/gpfs/CompMicro/projects/dynamorph/microglia/raw_for_segmentation/JANUARY/raw'
# RAW_JANUARY_FAST = '/gpfs/CompMicro/projects/dynamorph/microglia/raw_for_segmentation/JANUARY_FAST/raw'
#
# INTERMEDIATE_NOVEMBER = '/gpfs/CompMicro/projects/dynamorph/microglia/raw_for_segmentation/NOVEMBER/supp'
# INTERMEDIATE_JANUARY = '/gpfs/CompMicro/projects/dynamorph/microglia/raw_for_segmentation/JANUARY/supp'
# INTERMEDIATE_JANUARY_FAST = '/gpfs/CompMicro/projects/dynamorph/microglia/raw_for_segmentation/JANUARY_FAST/supp'

# '''======= TARGET is the output directory for "segmentation_validation" runs ONLY ======='''
#TARGET = '/gpfs/CompMicro/projects/dynamorph/microglia/segmentation_experiments/expt_009'
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

    # if not arguments_.input or not arguments_.output:
    #     print('no input or output supplied, using hard coded paths')
    #     n_gpu = 4
    #     TARGET = '/data_sm/home/michaelwu/VALIDATION'
    #     #TARGET = '/gpfs/CompMicro/projects/dynamorph/CellVAE/save_0005_bkp4.pt'
    #
    #     for sites, inputs, outputs in zip([sites_NOVEMBER, sites_JANUARY, sites_JANUARY_FAST],
    #                             [RAW_NOVEMBER, RAW_JANUARY, RAW_JANUARY_FAST],
    #                             [INTERMEDIATE_NOVEMBER, INTERMEDIATE_JANUARY, INTERMEDIATE_JANUARY_FAST]):
    #         segment_sites = [site for site in sites if os.path.exists(os.path.join(inputs, "%s.npy" % site)) and \
    #                                      os.path.exists(os.path.join(inputs, "%s_NNProbabilities.npy" % site))]
    #         sep = np.linspace(0, len(segment_sites), n_gpu+1).astype(int)
    #         for i in range(n_gpu):
    #             _sites = segment_sites[sep[i]:sep[i+1]]
    #             args = (inputs, outputs, TARGET, _sites)
    #             process = Worker(args, gpuid=i)
    #             process.start()
    #         for i in range(n_gpu):
    #             process.join()
    # else:
        print("CLI arguments provided")
        inputs = arguments_.raw
        outputs = arguments_.supplementary

        n_gpu = arguments_.gpus
        method = arguments_.method

        # segmentation validation requires raw, supp, and validation definitions
        if method == 'segmentation_validation':
            if arguments_.validation:
                TARGET = arguments_.validation
            else:
                raise AttributeError("validation directory must be specified when method=segmentation_validation")
            if not arguments_.supplementary:
                raise AttributeError("supplemntary directory must be specifie dwhen method=segmentation_validation")

        # segmentation requires raw (NNProb), and weights to be defined
        elif method == 'segmentation':
            if arguments_.weights is None:
                raise AttributeError("Weights path must be specified when method=segmentation")
            else:
                TARGET = arguments_.weights

        # instance segmentation requires raw (stack, NNprob), supp (to write outputs) to be defined
        elif method == 'instance_segmentation':
            TARGET = ''
        else:
            raise AttributeError(f"method flag {arguments_.method} not implemented")

        # all methods all require
        if arguments_.sites:
            sites = arguments_.sites
        else:
            # get all "XX-SITE_#" identifiers in raw data directory
            sites = [os.path.splitext(site)[0][0:9].split('_NN')[0] for site in os.listdir(inputs) if site.endswith(".npy")]
            sites = list(set(sites))

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
        '-r', '--raw',
        type=str,
        required=True,
        help="Path to multipage-tiff file of format [t, x, y]",
    )
    parser.add_argument(
        '-s', '--supplementary',
        type=str,
        required=False,
        help="Path to write results",
    )
    parser.add_argument(
        '-v', '--validation',
        type=str,
        required=False,
        help="Path to write validation images",
    )
    parser.add_argument(
        '-w', '--weights',
        type=str,
        required=False,
        default=None,
        help="Path to pytorch weights from trained segmentaton model",
    )
    parser.add_argument(
        '-m', '--method',
        type=str,
        required=True,
        choices=['segmentation', 'instance_segmentation', 'segmentation_validation'],
        default='segmentation',
        help="Method: one of 'segmentation', 'instance_segmentation', or 'segmentation_validation'",
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
        help="comma-delimited list of FOVs (subfolders in raw data directory)",
    )
    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_args()
    main(arguments)
