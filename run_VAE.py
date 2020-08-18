from pipeline.patch_VAE import assemble_VAE, process_VAE, trajectory_matching
from multiprocessing import Pool, Queue, Process
import os
import numpy as np

RAW_NOVEMBER = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/NOVEMBER/raw'
RAW_JANUARY = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/JANUARY/raw'
RAW_JANUARY_FAST = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/JANUARY_FAST/raw'

INTERMEDIATE_NOVEMBER = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/NOVEMBER/supp'
INTERMEDIATE_JANUARY = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/JANUARY/supp'
INTERMEDIATE_JANUARY_FAST = '/gpfs/CompMicro/Projects/learningCellState/microglia/raw_for_segmentation/JANUARY_FAST/supp'

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
    def __init__(self, inputs, gpuid=0):
        super().__init__()
        self.gpuid=gpuid
        self.inputs=inputs

    def run(self):
        #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        #os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpuid)
        #assemble_VAE(self.inputs)
        process_VAE(self.inputs)
        #trajectory_matching(self.inputs)


def main():
    sites = sites_NOVEMBER
    inputs = RAW_NOVEMBER
    outputs = INTERMEDIATE_NOVEMBER
    
    wells = set(s[:2] for s in sites)
    process = []
    for i, well in enumerate(wells):
        well_sites = [s for s in sites if s[:2] == well]
        print(well_sites)
        args = (inputs, outputs, TARGET, well_sites)
        p = Worker(args, gpuid=i)
        p.start()
        p.join()

if __name__ == '__main__':
    main()
