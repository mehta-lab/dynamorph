# bchhun, {2020-02-21}

from pipeline.patch_VAE import extract_patches, build_trajectories
from SingleCellPatch.extract_patches import get_im_sites
from multiprocessing import Pool, Queue, Process
import os
import numpy as np
import argparse
from configs.config_reader import YamlReader


class Worker(Process):
    def __init__(self, inputs, cpu_id=0, method='extract_patches'):
        super().__init__()
        self.cpu_id = cpu_id
        self.inputs = inputs
        self.method = method

    def run(self):
        if self.method == 'extract_patches':
            extract_patches(*self.inputs)
        elif self.method == 'build_trajectories':
            build_trajectories(*self.inputs)


def main(method_, raw_dir_, supp_dir_, config_):

    print("CLI arguments provided")
    raw = raw_dir_
    supp = supp_dir_
    method = method_
    fov = config.patch.fov

    n_cpus = config.patch.num_cpus

    # extract patches needs raw (NN probs, stack), supp (cell_positions, cell_pixel_assignments)
    if method == 'extract_patches':
        if not raw:
            raise AttributeError("raw directory must be specified when method = extract_patches")
        if not supp:
            raise AttributeError("supplementary directory must be specified when method = extract_patches")

    # extract patches needs supp (cell_positions, cell_pixel_assignments)
    elif method == 'build_trajectories':
        if not supp:
            raise AttributeError("supplementary directory must be specified when method = extract_patches")

    if fov:
        sites = fov
    else:
        # get all "XX-SITE_#" identifiers in raw data directory
        sites = get_im_sites(raw)
    # if probabilities and formatted stack exist
    segment_sites = [site for site in sites if os.path.exists(os.path.join(raw, "%s.npy" % site)) and \
                     os.path.exists(os.path.join(raw, "%s_NNProbabilities.npy" % site))]
    if len(segment_sites) == 0:
        raise AttributeError("no sites found in raw directory with preprocessed data and matching NNProbabilities")

    # process each site on a different GPU if using multi-gpu
    sep = np.linspace(0, len(segment_sites), n_cpus + 1).astype(int)

    # TARGET is never used in either extract_patches or build_trajectory
    processes = []
    for i in range(n_cpus):
        _sites = segment_sites[sep[i]:sep[i + 1]]
        args = (raw, supp, _sites, config_)
        p = Worker(args, cpu_id=i, method=method)
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
        '-m', '--method',
        type=str,
        required=False,
        choices=['extract_patches', 'build_trajectories'],
        default='extract_patches',
        help="Method: one of 'extract_patches', 'build_trajectories'",
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
    for (raw_dir, supp_dir) in list(zip(config.patch.raw_dirs, config.patch.supp_dirs)):
        main(arguments.method, raw_dir, supp_dir, config)
