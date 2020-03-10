# bchhun, {2020-02-21}

from pipeline.segmentation import segmentation, instance_segmentation
from pipeline.segmentation_validation import segmentation_validation_bryant, segmentation_validation_michael

from multiprocessing import Pool, Queue, Process
import os

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

SITES_ctrl = ['C5-Site_0', 'C5-Site_4']
# SITES_GBM = ['B2-Site_0', 'B2-Site_4']
# SITES_IL17 = ['B4-Site_0', 'B4-Site_4']
# SITES_IFbeta=['B5-Site_0', 'B5-Site_4']
# SITES_fast = ['C5-Site_0', 'C5-Site_4']

RAW = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/NOVEMBER/raw'
# RAW_JAN = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/JANUARY/raw'
RAW_FAST = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/JANUARY_FAST/raw'

INTERMEDIATE_NOV = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/NOVEMBER/supp'
# INTERMEDIATE_JAN = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/JANUARY/supp'
INTERMEDIATE_JAN_FAST = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/JANUARY_FAST/supp'

TARGET = '/gpfs/CompMicro/Projects/learningCellState/microglia/segmentation_experiments/expt_009'


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
        segmentation_validation_michael(self.inputs, self.gpuid, 'mg')


def main():

    inputs_1 = (RAW, INTERMEDIATE_NOV, TARGET, SITES_ctrl)
    process_1 = Worker(inputs_1, gpuid=0)
    process_1.start()

    #inputs_2 = (RAW, INTERMEDIATE_NOV, TARGET, SITES_GBM)
    #process_2 = Worker(inputs_2, gpuid=1)
    #process_2.start()

    # inputs_3 = (RAW, INTERMEDIATE_NOV, TARGET, SITES_IL17)
    # process_3 = Worker(inputs_3, gpuid=2)
    # process_3.start()

    #inputs_4 = (RAW, INTERMEDIATE_NOV, TARGET, SITES_IFbeta)
    #process_4 = Worker(inputs_4, gpuid=2)
    #process_4.start()

    #inputs_4 = (RAW_FAST, INTERMEDIATE_JAN_FAST, TARGET, SITES_fast)
    #process_4 = Worker(inputs_4, gpuid=3)
    #process_4.start()


if __name__ == '__main__':
    main()
