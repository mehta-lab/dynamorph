from pipeline.patch_VAE import assemble_VAE, process_VAE, trajectory_matching
from torch.multiprocessing import Pool, Queue, Process
import torch.multiprocessing as mp
import os, sys
import argparse

from configs.config_reader import YamlReader


class Worker(Process):
    def __init__(self, inputs, gpuid=0, method='assemble'):
        super().__init__()
        self.gpuid = gpuid
        self.inputs = inputs
        self.method = method

    def run(self):
        if self.method == 'assemble':
            assemble_VAE(*self.inputs)
        elif self.method == 'process':
            process_VAE(*self.inputs)
        elif self.method == 'trajectory_matching':
            trajectory_matching(*self.inputs)


def main(method_, raw_dir_, supp_dir_, config_):
    method = method_

    inputs = raw_dir_
    outputs = supp_dir_
    weights = config_.latent_encoding.weights
    # channels = config_.inference.channels
    # network = config_.inference.model
    gpu_id = config_.latent_encoding.gpu_ids

    # assert len(channels) > 0, "At least one channel must be specified"

    # todo file path checks can be done earlier
    # assemble needs raw (write file_paths/static_patches/adjusted_patches), and supp (read site-supps)
    if method == 'assemble':
        if not inputs:
            raise AttributeError("raw directory must be specified when method = assemble")
        if not outputs:
            raise AttributeError("supplementary directory must be specified when method = assemble")

    # process needs raw (load _file_paths), and target (torch weights)
    elif method == 'process':
        if not inputs:
            raise AttributeError("raw directory must be specified when method = process")
        # if type(weights) is not list:
        #     weights = [weights]
        if not weights:
            raise AttributeError("pytorch VQ-VAE weights path must be specified when method = process")

    # trajectory matching needs raw (load file_paths, write trajectories), supp (load cell_traj)
    elif method == 'trajectory_matching':
        if not inputs:
            raise AttributeError("raw directory must be specified when method = trajectory_matching")
        if not outputs:
            raise AttributeError("supplementary directory must be specified when method = trajectory_matching")

    if config_.latent_encoding.fov:
        sites = config_.latent_encoding.fov
    else:
        # get all "XX-SITE_#" identifiers in raw data directory
        img_names = [file for file in os.listdir(inputs) if (file.endswith(".npy")) & ('_NN' not in file)]
        sites = [os.path.splitext(img_name)[0] for img_name in img_names]
        sites = list(set(sites))

    wells = set(s[:2] for s in sites)
    mp.set_start_method('spawn', force=True)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print("CUDA_VISIBLE_DEVICES=" + os.environ["CUDA_VISIBLE_DEVICES"])
    for i, well in enumerate(wells):
        # don't batch weights here
        # for weight in weights:
        # print('Encoding using model {}'.format(weight))
        well_sites = [s for s in sites if s[:2] == well]
        # args = (inputs, outputs, channels, weights, well_sites, network)
        args = (inputs, outputs, well_sites, config_)
        p = Worker(args, gpuid=gpu_id, method=method)
        p.start()
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
        choices=['assemble', 'process', 'trajectory_matching'],
        default='assemble',
        help="Method: one of 'assemble', 'process' or 'trajectory_matching'",
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
    for (raw_dir, supp_dir) in list(zip(config.latent_encoding.raw_dirs, config.latent_encoding.supp_dirs)):
        main(arguments.method, raw_dir, supp_dir, config)
