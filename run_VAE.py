from pipeline.patch_VAE import assemble_VAE, process_VAE, process_PCA, trajectory_matching
from multiprocessing import Pool, Queue, Process
import os
import argparse


class Worker(Process):
    def __init__(self, inputs, gpuid=0, method='assemble'):
        super().__init__()
        self.gpuid = gpuid
        self.inputs = inputs
        self.method = method

    def run(self):
        #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        #os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpuid)

        if self.method == 'assemble':
            assemble_VAE(*self.inputs)
        elif self.method == 'process':
            process_VAE(*self.inputs)
        elif self.method == 'pca':
            pass
            #process_PCA(*self.inputs)
        elif self.method == 'trajectory_matching':
            trajectory_matching(*self.inputs)


def main(arguments_):

    inputs = arguments_.raw
    outputs = arguments_.supplementary
    weights = arguments_.weights
    method = arguments_.method
    channels = arguments_.channels
    assert len(channels) > 0, "At least one channel must be specified"

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
        if not arguments_.weights:
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
        sites = [os.path.splitext(site)[0][0:9].split('_NN')[0] for site in os.listdir(inputs) if
                 site.endswith(".npy")]
        sites = list(set(sites))

    wells = set(s[:2] for s in sites)
    for i, well in enumerate(wells):
        well_sites = [s for s in sites if s[:2] == well]
        print(well_sites)        
        args = (inputs, outputs, channels, weights, well_sites)
        p = Worker(args, gpuid=i, method=method)
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
        '-c', '--channels',
        type=lambda s: [int(item.strip(' ').strip("'")) for item in s.split(',')],
        required=False,
        default=[0, 1], # Assuming two channels by default
        help="comma-delimited list of channel indices (e.g. 1,2,3)",
    )
    parser.add_argument(
        '-w', '--weights',
        type=str,
        required=False,
        help="Path to pytorch model weights for VQ-VAE or PCA weights",
    )
    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_args()
    main(arguments)
