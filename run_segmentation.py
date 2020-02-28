# bchhun, {2020-02-21}

from pipeline.segmentation import segmentation, instance_segmentation

from multiprocessing import Pool, Queue
import os

# ESS from hulk

# SITES = ['B4-Site_0', 'B4-Site_1',  'B4-Site_2',  'B4-Site_3',  'B4-Site_4', 'B4-Site_5', 'B4-Site_6', 'B4-Site_7', 'B4-Site_8',
#          'B5-Site_0', 'B5-Site_1',  'B5-Site_2',  'B5-Site_3',  'B5-Site_4', 'B5-Site_5', 'B5-Site_6', 'B5-Site_7', 'B5-Site_8',
#          'C3-Site_0', 'C3-Site_1',  'C3-Site_2',  'C3-Site_3',  'C3-Site_4', 'C3-Site_5', 'C3-Site_6', 'C3-Site_7', 'C3-Site_8',
#          'C4-Site_0', 'C4-Site_1',  'C4-Site_2',  'C4-Site_3',  'C4-Site_4', 'C4-Site_5', 'C4-Site_6', 'C4-Site_7', 'C4-Site_8',
#          'C5-Site_0', 'C5-Site_1',  'C5-Site_2',  'C5-Site_3',  'C5-Site_4', 'C5-Site_5', 'C5-Site_6', 'C5-Site_7', 'C5-Site_8']

SITES = ['C5-Site_0', 'C5-Site_1', 'C5-Site_2', 'C5-Site_3', 'C5-Site_4', 'C5-Site_5', 'C5-Site_6', 'C5-Site_7', 'C5-Site_8']

RAW = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/NOVEMBER/raw'
INTERMEDIATE = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/NOVEMBER/supp'


def main():

    # p = Pool(4)
    # queue is shared and represents GPU ID
    # q = Queue()
    # q.put([0, 1, 2, 3])

    inputs = []
    for site in SITES:
        # if not os.path.isdir(INTERMEDIATE+os.sep+site):
        #     os.mkdir(INTERMEDIATE+os.sep+site)
        inputs.append((RAW, INTERMEDIATE, site, ''))

        # segmentation((RAW, INTERMEDIATE, SITES[0]))
        # instance_segmentation((RAW, INTERMEDIATE, SITES[0]))
        segmentation(inputs)
        instance_segmentation(inputs)

    # p.map(segmentation, inputs)
    # p.map(instance_segmentation, inputs)

    # p.close()
    # p.join()


if __name__ == '__main__':
    main()