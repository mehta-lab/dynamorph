# bchhun, {2020-02-21}

from pipeline.patch_preVAE import extract_patches, assemble_VAE, build_trajectories


# ESS from hulk

# SITES = ['B4-Site_0', 'B4-Site_1',  'B4-Site_2',  'B4-Site_3',  'B4-Site_4', 'B4-Site_5', 'B4-Site_6', 'B4-Site_7', 'B4-Site_8',
#          'B5-Site_0', 'B5-Site_1',  'B5-Site_2',  'B5-Site_3',  'B5-Site_4', 'B5-Site_5', 'B5-Site_6', 'B5-Site_7', 'B5-Site_8',
#          'C3-Site_0', 'C3-Site_1',  'C3-Site_2',  'C3-Site_3',  'C3-Site_4', 'C3-Site_5', 'C3-Site_6', 'C3-Site_7', 'C3-Site_8',
#          'C4-Site_0', 'C4-Site_1',  'C4-Site_2',  'C4-Site_3',  'C4-Site_4', 'C4-Site_5', 'C4-Site_6', 'C4-Site_7', 'C4-Site_8',
#          'C5-Site_0', 'C5-Site_1',  'C5-Site_2',  'C5-Site_3',  'C5-Site_4', 'C5-Site_5', 'C5-Site_6', 'C5-Site_7', 'C5-Site_8']

SITES = ['C5-Site_0']

DATA_PREP = '/gpfs/CompMicro/Hummingbird/Processed/Galina/VAE/data_temp'

from multiprocessing import Pool, Queue, Process
import os

# ESS from hulk

# SITES = ['B4-Site_0', 'B4-Site_1',  'B4-Site_2',  'B4-Site_3',  'B4-Site_4', 'B4-Site_5', 'B4-Site_6', 'B4-Site_7', 'B4-Site_8',
#          'B5-Site_0', 'B5-Site_1',  'B5-Site_2',  'B5-Site_3',  'B5-Site_4', 'B5-Site_5', 'B5-Site_6', 'B5-Site_7', 'B5-Site_8',
#          'C3-Site_0', 'C3-Site_1',  'C3-Site_2',  'C3-Site_3',  'C3-Site_4', 'C3-Site_5', 'C3-Site_6', 'C3-Site_7', 'C3-Site_8',
#          'C4-Site_0', 'C4-Site_1',  'C4-Site_2',  'C4-Site_3',  'C4-Site_4', 'C4-Site_5', 'C4-Site_6', 'C4-Site_7', 'C4-Site_8',
#          'C5-Site_0', 'C5-Site_1',  'C5-Site_2',  'C5-Site_3',  'C5-Site_4', 'C5-Site_5', 'C5-Site_6', 'C5-Site_7', 'C5-Site_8']

SITES_ctrl = ['C5-Site_0', 'C5-Site_4']
SITES_GBM = ['B2-Site_0', 'B2-Site_4']
# SITES_IL17 = ['B4-Site_0', 'B4-Site_4']
SITES_IFbeta=['B5-Site_0', 'B5-Site_4']
SITES_fast = ['C5-Site_0', 'C5-Site_4']

RAW = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/NOVEMBER/raw'
# RAW_JAN = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/JANUARY/raw'
RAW_FAST = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/JANUARY_FAST/raw'

INTERMEDIATE_NOV = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/NOVEMBER/supp'
# INTERMEDIATE_JAN = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/JANUARY/supp'
INTERMEDIATE_JAN_FAST = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/JANUARY_FAST/supp'

TARGET = '/gpfs/CompMicro/Projects/learningCellState/microglia/segmentation_experiments/expt_001'


class Worker(Process):
    def __init__(self, inputs, gpuid=0):
        super().__init__()
        self.gpuid=gpuid
        self.inputs=inputs

    def run(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpuid)

        extract_patches(self.inputs)
        build_trajectories(self.inputs)
        assemble_VAE(self.inputs)


def main():

    # loads 'Site.npy',
    #       '_NNProbabilities.npy',
    #       '/Site-supps/Site/cell_positions.pkl',
    #       '/Site-supps/site/cell_pixel_assignments.pkl',

    # generates 'stacks_%d.pkl' % timepoint

    # prints: "writing time %d"

    inputs1 = (RAW, INTERMEDIATE_NOV, 0, SITES_ctrl)
    w = Worker(inputs1, 0)
    w.start()

    # *** NOT USED WITH VAE ***
    # *** USED IN POST-PCA TRAJ MATCHING ***
    # loads 'cell_positions.pkl', 'cell_pixel_assignments.pkl'
    # generates 'cell_traj.pkl'
    inputs2 = (RAW, INTERMEDIATE_NOV, 0, SITES_ctrl)
    w = Worker(inputs2, 1)
    w.start()




if __name__ == '__main__':
    main()
