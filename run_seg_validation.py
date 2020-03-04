# bchhun, {2020-03-03}

from pipeline.segmentation_validation import segmentation_validation_bryant, segmentation_validation_michael

from multiprocessing import Pool, Queue, Process

SITES_ctrl = ['C5-Site_0', 'C5-Site_4']
SITES_GBM = ['B2-Site_0', 'B2-Site_4']
SITES_IL17 = ['B4-Site_0', 'B4-Site_4']
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
        segmentation_validation_michael(self.inputs, self.gpuid)


def main():
    # inputs = []
    # for site in SITES:
    #     # if not os.path.isdir(INTERMEDIATE+os.sep+site):
    #     #     os.mkdir(INTERMEDIATE+os.sep+site)
    #     inputs.append((RAW, INTERMEDIATE, site, ''))
    #
    #     # segmentation((RAW, INTERMEDIATE, SITES[0]))
    #     # instance_segmentation((RAW, INTERMEDIATE, SITES[0]))
    #     segmentation(inputs)
    #     instance_segmentation(inputs)

    # p.map(segmentation, inputs)
    # p.map(instance_segmentation, inputs)

    inputs_1 = (RAW, INTERMEDIATE_NOV, TARGET, SITES_ctrl)
    process_1 = Worker(inputs_1, gpuid=0)
    process_1.start()

    inputs_2 = (RAW, INTERMEDIATE_NOV, TARGET, SITES_GBM)
    process_2 = Worker(inputs_2, gpuid=1)
    process_2.start()

    inputs_3 = (RAW, INTERMEDIATE_NOV, TARGET, SITES_IL17)
    process_3 = Worker(inputs_3, gpuid=2)
    process_3.start()

    inputs_4 = (RAW, INTERMEDIATE_NOV, TARGET, SITES_IFbeta)
    process_4 = Worker(inputs_4, gpuid=3)
    process_4.start()

    # inputs_5 = (RAW_FAST, INTERMEDIATE_JAN_FAST, TARGET, SITES_fast)
    # process_5 = Worker(inputs_5, gpuid=3)
    # process_5.start()



if __name__ == '__main__':
    main()