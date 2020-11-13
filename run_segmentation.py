# bchhun, {2020-02-21}

from pipeline.segmentation import segmentation, instance_segmentation
from pipeline.segmentation_validation import segmentation_validation_michael
from multiprocessing import Process
import os
import numpy as np

import argparse


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
            raise AttributeError("Weights supp_dir must be specified when method=segmentation")
        else:
            TARGET = arguments_.weights

    # instance segmentation requires raw (stack, NNprob), supp (to write outputs) to be defined
    elif method == 'instance_segmentation':
        TARGET = ''

    else:
        raise AttributeError(f"method flag {arguments_.method} not implemented")

    # all methods all require
    if arguments_.fov:
        sites = arguments_.fov
    else:
        # get all "XX-SITE_#" identifiers in raw data directory
        img_names = [file for file in os.listdir(inputs) if (file.endswith(".npy")) & ('_NN' not in file)]
        sites = [os.path.splitext(img_name)[0] for img_name in img_names]
        sites = list(set(sites))

    segment_sites = [site for site in sites if os.path.exists(os.path.join(inputs, "%s.npy" % site))]
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
        help="Path to the folder for raw inputs (multipage-tiff file of format [t, x, y]) and summary results",
    )
    parser.add_argument(
        '-s', '--supplementary',
        type=str,
        required=False,
        help="Path to the folder for supplementary results",
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
        help="Path to keras weights for trained UNet segmentaton model",
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
        '-f', '--fov',
        type=lambda s: [str(item.strip(' ').strip("'")) for item in s.split(',')],
        required=False,
        help="comma-delimited list of FOVs (subfolders in raw data directory)",
    )
    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_args()
    main(arguments)
