# bchhun, {2020-02-21}

from pipeline.segmentation import segmentation, instance_segmentation
from pipeline.segmentation_validation import segmentation_validation_bryant

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

        segmentation(self.inputs)
        instance_segmentation(self.inputs)
        segmentation_validation_bryant(self.inputs, self.gpuid)


def main():

    inputs_1 = (RAW, INTERMEDIATE_NOV, TARGET, SITES_ctrl)
    process_1 = Worker(inputs_1, gpuid=0)
    process_1.start()

    inputs_2 = (RAW, INTERMEDIATE_NOV, TARGET, SITES_GBM)
    process_2 = Worker(inputs_2, gpuid=1)
    process_2.start()

    # inputs_3 = (RAW, INTERMEDIATE_NOV, TARGET, SITES_IL17)
    # process_3 = Worker(inputs_3, gpuid=2)
    # process_3.start()

    inputs_4 = (RAW, INTERMEDIATE_NOV, TARGET, SITES_IFbeta)
    process_4 = Worker(inputs_4, gpuid=2)
    process_4.start()

    inputs_4 = (RAW_FAST, INTERMEDIATE_JAN_FAST, TARGET, SITES_fast)
    process_4 = Worker(inputs_4, gpuid=3)
    process_4.start()


if __name__ == '__main__':
    main()
