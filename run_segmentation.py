# bchhun, {2020-02-21}

from pipeline.segmentation import segmentation, instance_segmentation
from pipeline.segmentation_validation import segmentation_validation_bryant, segmentation_validation_michael
from run_preproc import sites_NOVEMBER, sites_JANUARY, sites_JANUARY_FAST
from multiprocessing import Pool, Queue, Process
import os
import numpy as np

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

# SITES_ctrl = ['C5-Site_0', 'C5-Site_4']
# SITES_GBM = ['B2-Site_0', 'B2-Site_4']
# SITES_IL17 = ['B4-Site_0', 'B4-Site_4']
# SITES_IFbeta=['B5-Site_0', 'B5-Site_4']
# SITES_fast = ['C5-Site_0', 'C5-Site_4']

RAW_NOVEMBER = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/NOVEMBER/raw'
RAW_JANUARY = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/JANUARY/raw'
RAW_JANUARY_FAST = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/JANUARY_FAST/raw'

INTERMEDIATE_NOVEMBER = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/NOVEMBER/supp'
INTERMEDIATE_JANUARY = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/JANUARY/supp'
INTERMEDIATE_JANUARY_FAST = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/JANUARY_FAST/supp'

#TARGET = '/gpfs/CompMicro/Projects/learningCellState/microglia/segmentation_experiments/expt_009'
TARGET = '/data_sm/home/michaelwu/VALIDATION'

class Worker(Process):
    def __init__(self, inputs, gpuid=0):
        super().__init__()
        self.gpuid=gpuid
        self.inputs=inputs

    def run(self):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpuid)

        #segmentation(self.inputs)
        instance_segmentation(self.inputs)
        #segmentation_validation_michael(self.inputs, self.gpuid, 'unfiltered')


def main():
    n_gpu = 4
    for sites, inputs, outputs in zip([sites_NOVEMBER, sites_JANUARY, sites_JANUARY_FAST],
                            [RAW_NOVEMBER, RAW_JANUARY, RAW_JANUARY_FAST],
                            [INTERMEDIATE_NOVEMBER, INTERMEDIATE_JANUARY, INTERMEDIATE_JANUARY_FAST]):
        segment_sites = [site for site in sites if os.path.exists(os.path.join(inputs, "%s.npy" % site)) and \
                                     os.path.exists(os.path.join(inputs, "%s_NNProbabilities.npy" % site))]
        sep = np.linspace(0, len(segment_sites), n_gpu+1).astype(int)
        for i in range(n_gpu):
            _sites = segment_sites[sep[i]:sep[i+1]]
            args = (inputs, outputs, TARGET, _sites)
            process = Worker(args, gpuid=i)
            process.start()
        for i in range(n_gpu):
            process.join()


if __name__ == '__main__':
    main()
