# bchhun, {2020-02-24}

import numpy as np
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import pickle
import torch
from torch.utils.data import TensorDataset

from SingleCellPatch.generate_trajectories import process_site_build_trajectory
from SingleCellPatch.extract_patches import process_site_extract_patches

from HiddenStateExtractor.vq_vae import VQ_VAE, prepare_dataset_v2, rescale


# 7
def build_trajectories(temp_folder, site):
    site_path = os.path.join(temp_folder + '/' + site + '.npy')

    site_segmentation_path = os.path.join(temp_folder, '%s_NNProbabilities.npy' % site)
    site_supp_files_folder = os.path.join(temp_folder, '%s-supps' % site[:2], '%s' % site)

    process_site_build_trajectory(site_path, site_segmentation_path, site_supp_files_folder)