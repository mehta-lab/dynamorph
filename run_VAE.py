from pipeline.patch_VAE import assemble_VAE, process_VAE, trajectory_matching
from run_dim_reduction import process_PCA_hcs
from torch.multiprocessing import Pool, Queue, Process
import torch.multiprocessing as mp
import os
import argparse


class Worker(Process):
    def __init__(self, inputs, gpuid=0, method='assemble'):
        super().__init__()
        self.gpuid = gpuid
        self.inputs = inputs
        self.method = method

    def run(self):
        if self.method == 'assemble':
            assemble_VAE(self.inputs)
        elif self.method == 'process':
            process_VAE(self.inputs, save_ouput=True)
        elif self.method == 'pca':
            process_PCA_hcs(self.inputs)
        elif self.method == 'trajectory_matching':
            trajectory_matching(self.inputs)


def main(arguments_):

    inputs = arguments_.raw
    outputs = arguments_.supplementary
    weights = arguments_.weights
    method = arguments_.method
    gpu = arguments_.gpu

    # assemble needs raw (write file_paths/static_patches/adjusted_patches), and supp (read site-supps)
    if arguments_.method == 'assemble':
        if not arguments_.raw:
            raise AttributeError("raw directory must be specified when method = assemble")
        if not arguments_.supplementary:
            raise AttributeError("supplementary directory must be specified when method = assemble")

    # process and pca needs raw (load _file_paths), and target (torch weights)
    elif arguments_.method == 'process' or arguments_.method == 'pca':
        if not arguments_.raw:
            raise AttributeError("raw directory must be specified when method = process / pca")
        if type(weights) is not list:
            weights = [weights]
        if not weights:
            raise AttributeError("pytorch VQ-VAE weights path must be specified when method = process / pca")

    # trajectory matching needs raw (load file_paths, write trajectories), supp (load cell_traj)
    elif arguments_.method == 'trajectory_matching':
        if not arguments_.raw:
            raise AttributeError("raw directory must be specified when method = trajectory_matching")
        if not arguments_.supplementary:
            raise AttributeError("supplementary directory must be specified when method = trajectory_matching")

    if arguments_.fov:
        sites = arguments_.fov
    else:
        # get all "XX-SITE_#" identifiers in raw data directory
        img_names = [file for file in os.listdir(inputs) if (file.endswith(".npy")) & ('_NN' not in file)]
        sites = [os.path.splitext(img_name)[0] for img_name in img_names]
        sites = list(set(sites))

    wells = set(s[:2] for s in sites)
    mp.set_start_method('spawn', force=True)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    print("CUDA_VISIBLE_DEVICES=" + os.environ["CUDA_VISIBLE_DEVICES"])
    for i, well in enumerate(wells):
        for weight in weights:
            print('Encoding using model {}'.format(weight))
            well_sites = [s for s in sites if s[:2] == well]
            # print(well_sites)
            args = (inputs, outputs, weight, well_sites)
            p = Worker(args, gpuid=gpu, method=method)
            p.start()
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
        required=False,
        help="Path to the folder for raw inputs (multipage-tiff file of format [t, x, y]) and summary results",
    )
    parser.add_argument(
        '-s', '--supplementary',
        type=str,
        required=False,
        help="Path to the folder for supplementary results",
    )
    parser.add_argument(
        '-m', '--method',
        type=str,
        required=True,
        choices=['assemble', 'process', 'pca', 'trajectory_matching'],
        default='assemble',
        help="Method: one of 'assemble', 'process', 'pca' or 'trajectory_matching'",
    )
    parser.add_argument(
        '-f', '--fov',
        type=lambda s: [str(item.strip(' ').strip("'")) for item in s.split(',')],
        required=False,
        help="list of field-of-views to process (subfolders in raw data directory)",
    )
    parser.add_argument(
        '-w', '--weights',
        nargs='+',
        default=[],
        type=str,
        required=False,
        help="Path to directories containing VQ-VAE model weights or PCA weights",
    )
    parser.add_argument(
        '-g', '--gpu',
        type=int,
        required=False,
        default=0,
        help="ID of the GPU to use",
    )
    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_args()
    main(arguments)
