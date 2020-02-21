# bchhun, {2020-02-21}

import numpy as np
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import pickle
import torch
from torch.utils.data import TensorDataset

from SingleCellPatch.generate_trajectories import process_site_build_trajectory
from SingleCellPatch.extract_patches import process_site_extract_patches

from HiddenStateExtractor.vq_vae import VQ_VAE, prepare_dataset_v2, rescale


# 6
def extract_patches(temp_folder, site):
    site_path = os.path.join(temp_folder + '/' + site + '.npy')

    site_segmentation_path = os.path.join(temp_folder, '%s_NNProbabilities.npy' % site)
    site_supp_files_folder = os.path.join(temp_folder, '%s-supps' % site[:2], '%s' % site)

    process_site_extract_patches(site_path, site_segmentation_path, site_supp_files_folder, window_size=256)


# 7
def build_trajectories(temp_folder, site):
    site_path = os.path.join(temp_folder + '/' + site + '.npy')

    site_segmentation_path = os.path.join(temp_folder, '%s_NNProbabilities.npy' % site)
    site_supp_files_folder = os.path.join(temp_folder, '%s-supps' % site[:2], '%s' % site)

    process_site_build_trajectory(site_path, site_segmentation_path, site_supp_files_folder)


#8
def assemble_VAE(temp_folder, site):

    # Prepare dataset for VAE
    dat_fs = []
    # for s in ['C5-Site_%d' % i for i in [0, 1, 2, 3, 5, 6, 7]]:
    supp_files_folder = os.path.join(temp_folder, '%s-supps' % site[:2], '%s' % site)
    dat_fs.extend([os.path.join(supp_files_folder, f) for f in os.listdir(supp_files_folder) if f.startswith('stacks')])

    dataset, fs = prepare_dataset_v2(dat_fs, cs=[0, 1])

    with open(os.path.join(temp_folder, '%s_file_paths.pkl' % site[:2]), 'wb') as f:
        pickle.dump(fs, f)

    torch.save(dataset, os.path.join(temp_folder, '%s_all_static_patches.pt' % site[:2]))

    # Adjust channel mean/std
    # phase: 0.4980 plus/minus 0.0257
    # retardance: 0.0285 plus/minus 0.0261, only adjust for mean
    phase_slice = dataset.tensors[0][:, 0]
    phase_slice = ((phase_slice - phase_slice.mean()) / phase_slice.std()) * 0.0257 + 0.4980
    retard_slice = dataset.tensors[0][:, 1]
    retard_slice = retard_slice / retard_slice.mean() * 0.0285
    adjusted_dataset = TensorDataset(torch.stack([phase_slice, retard_slice], 1))
    torch.save(adjusted_dataset, os.path.join(temp_folder, '%s_all_adjusted_static_patches.pt' % site[:2]))
