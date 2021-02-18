# bchhun, {2020-02-21}

from pipeline.segmentation import segmentation, instance_segmentation
from pipeline.segmentation_validation import segmentation_validation_michael
from multiprocessing import Process
import os
import numpy as np

import argparse
from configs.config_reader import YamlReader


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
            segmentation(*self.inputs)
        elif self.method == 'instance_segmentation':
            instance_segmentation(*self.inputs)
        elif self.method == 'segmentation_validation':
            segmentation_validation_michael(self.inputs, self.gpuid, 'unfiltered')


def main(method_, raw_dir_, supp_dir_, val_dir_, config_):
    method = method_

    inputs = raw_dir_
    outputs = supp_dir_
    gpus = config.inference.gpus

    assert len(config_.inference.channels) > 0, "At least one channel must be specified"

    # segmentation validation requires raw, supp, and validation definitions
    if method == 'segmentation_validation':
        if not val_dir_:
            raise AttributeError("validation directory must be specified when method=segmentation_validation")
        if not outputs:
            raise AttributeError("supplemntary directory must be specifie dwhen method=segmentation_validation")

    # segmentation requires raw (NNProb), and weights to be defined
    elif method == 'segmentation':
        if config_.inference.weights is None:
            raise AttributeError("Weights supp_dir must be specified when method=segmentation")

    # instance segmentation requires raw (stack, NNprob), supp (to write outputs) to be defined
    elif method == 'instance_segmentation':
        TARGET = ''
    else:
        raise AttributeError(f"method flag {method} not implemented")

    # all methods all require
    if config_.inference.fov:
        sites = config.inference.fov
    else:
        # get all "XX-SITE_#" identifiers in raw data directory
        img_names = [file for file in os.listdir(inputs) if (file.endswith(".npy")) & ('_NN' not in file)]
        sites = [os.path.splitext(img_name)[0] for img_name in img_names]
        sites = list(set(sites))

    segment_sites = [site for site in sites if os.path.exists(os.path.join(inputs, "%s.npy" % site))]
    sep = np.linspace(0, len(segment_sites), gpus + 1).astype(int)

    processes = []
    for i in range(gpus):
        _sites = segment_sites[sep[i]:sep[i + 1]]
        args = (inputs, outputs, val_dir_, _sites, config_)
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
        '-m', '--method',
        type=str,
        required=True,
        choices=['segmentation', 'instance_segmentation', 'segmentation_validation'],
        default='segmentation',
        help="Method: one of 'segmentation', 'instance_segmentation', or 'segmentation_validation'",
    )

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

    # batch run
    for (raw_dir, supp_dir, val_dir) in list(zip(config.files.raw_dirs, config.files.supp_dirs, config.files.val_dirs)):
        main(arguments.method, raw_dir, supp_dir, val_dir, config)
