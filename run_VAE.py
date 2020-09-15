from pipeline.patch_VAE import assemble_VAE, process_VAE, trajectory_matching
from multiprocessing import Pool, Queue, Process
import os
import argparse


RAW_NOVEMBER = '/gpfs/CompMicro/projects/dynamorph/microglia/raw_for_segmentation/NOVEMBER/raw'
RAW_JANUARY = '/gpfs/CompMicro/projects/dynamorph/microglia/raw_for_segmentation/JANUARY/raw'
RAW_JANUARY_FAST = '/gpfs/CompMicro/projects/dynamorph/microglia/raw_for_segmentation/JANUARY_FAST/raw'

INTERMEDIATE_NOVEMBER = '/gpfs/CompMicro/projects/dynamorph/microglia/raw_for_segmentation/NOVEMBER/supp'
INTERMEDIATE_JANUARY = '/gpfs/CompMicro/projects/dynamorph/microglia/raw_for_segmentation/JANUARY/supp'
INTERMEDIATE_JANUARY_FAST = '/gpfs/CompMicro/projects/dynamorph/microglia/raw_for_segmentation/JANUARY_FAST/supp'

sites_NOVEMBER = [
    'B2-Site_0', 'B2-Site_1', 'B2-Site_2', 'B2-Site_5', 
    'B4-Site_0', 'B4-Site_2', 'B4-Site_3', 'B4-Site_6', 
    'B5-Site_1', 'B5-Site_2', 'B5-Site_6', 'B5-Site_7', 
    'C4-Site_0', 'C4-Site_2', 'C4-Site_3', 'C4-Site_5', 
    'C5-Site_1', 'C5-Site_2', 'C5-Site_3', 'C5-Site_5', 
]

sites_JANUARY = [
    'B4-Site_0', 'B4-Site_1', 'B4-Site_2', 'B4-Site_3', 
    'B5-Site_0', 'B5-Site_1', 'B5-Site_2', 'B5-Site_4', 
    'C3-Site_1', 'C3-Site_5', 'C3-Site_6', 'C3-Site_8', 
    'C4-Site_5', 'C4-Site_6', 'C4-Site_7', 'C4-Site_8', 
    'C5-Site_1', 'C5-Site_3', 'C5-Site_6', 'C5-Site_7', 
]

TARGET = None


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
            assemble_VAE(self.inputs)
        elif self.method == 'process':
            process_VAE(self.inputs)
        elif self.method == 'trajectory_matching':
            trajectory_matching(self.inputs)


def main(arguments_):

    if not arguments_.input or not arguments_.output:
        print('no input or output supplied, using hard coded paths')

        sites = sites_NOVEMBER
        inputs = RAW_NOVEMBER
        outputs = INTERMEDIATE_NOVEMBER

        wells = set(s[:2] for s in sites)
        for i, well in enumerate(wells):
            well_sites = [s for s in sites if s[:2] == well]
            print(well_sites)
            args = (inputs, outputs, TARGET, well_sites)
            p = Worker(args, gpuid=i)
            p.start()
            p.join()
    else:
        print("CLI arguments provided")
        inputs = arguments_.input
        outputs = arguments_.output

        # results are written to subfolder "supp"
        outputs = os.path.join(outputs, "supp")
        if not os.path.isdir(outputs):
            os.makedirs(outputs, exist_ok=True)

        if arguments_.sites:
            sites = arguments_.sites
        else:
            sites = [site for site in os.listdir(inputs) if os.path.isdir(os.path.join(inputs, site))]

        method = arguments_.method
        wells = set(s[:2] for s in sites)
        for i, well in enumerate(wells):
            well_sites = [s for s in sites if s[:2] == well]
            print(well_sites)
            args = (inputs, outputs, TARGET, well_sites)
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
        choices=['assemble', 'process', 'trajectory_matching'],
        default='assemble',
        help="Method: one of 'assemble', 'process', or 'trajectory_matching'",
    )
    parser.add_argument(
        '-s', '--sites',
        type=lambda s: [str(item.strip(' ').strip("'")) for item in s.split(',')],
        required=False,
        help="list of field-of-views to process (subfolders in raw data directory)",
    )
    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_args()
    main(arguments)
