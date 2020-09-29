# bchhun, {2020-02-21}

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import pickle
import torch
import numpy as np
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset

from SingleCellPatch.generate_trajectories import process_site_build_trajectory
from SingleCellPatch.extract_patches import process_site_extract_patches

from HiddenStateExtractor.vq_vae import VQ_VAE, prepare_dataset_v2, rescale


def extract_patches(paths):
    """ Helper function for patch extraction

    Wrapper method `process_site_extract_patches` will be called, which 
    extracts individual cells from static frames for each site.

    Results will be saved in supplementary data folder, including:
        "stacks_*.pkl": single cell patches for each time slice

    Args:
        paths (list): list of paths, containing:
            0 - folder for raw data and segmentation results (in .npy)
            1 - folder for supplementary data
            2 - deprecated
            3 - list of site names

    """

    temp_folder, supp_folder, target, sites = paths[0], paths[1], paths[2], paths[3]

    for site in sites:
        site_path = os.path.join(temp_folder + '/' + site + '.npy')

        site_segmentation_path = os.path.join(temp_folder, '%s_NNProbabilities.npy' % site)
        site_supp_files_folder = os.path.join(supp_folder, '%s-supps' % site[:2], '%s' % site)
        if not os.path.exists(site_path) or \
            not os.path.exists(site_segmentation_path) or \
            not os.path.exists(site_supp_files_folder):
                print("Site not found %s" % site_path, flush=True)
        else:
            print("Building patches %s" % site_path, flush=True)

        process_site_extract_patches(site_path, site_segmentation_path, site_supp_files_folder,
                                     window_size=256)
    return


def build_trajectories(paths):
    """ Helper function for trajectory building

    Wrapper method `process_site_build_trajectory` will be called, which 
    connects and generates trajectories from individual cell identifications.

    Results will be saved in supplementary data folder, including:
        "cell_traj.pkl": list of trajectories and list of trajectory positions
            trajectories are dict of t_point: cell ID
            trajectory positions are dict of t_point: cell center position

    Args:
        paths (list): list of paths, containing:
            0 - folder for raw data and segmentation results (in .npy)
            1 - folder for supplementary data
            2 - deprecated
            3 - list of site names

    """

    temp_folder, supp_folder, target, sites = paths[0], paths[1], paths[2], paths[3]

    for site in sites:
        site_path = os.path.join(temp_folder + '/' + site + '.npy')

        site_segmentation_path = os.path.join(temp_folder, '%s_NNProbabilities.npy' % site)
        site_supp_files_folder = os.path.join(supp_folder, '%s-supps' % site[:2], '%s' % site)
        if not os.path.exists(site_path) or \
            not os.path.exists(site_segmentation_path) or \
            not os.path.exists(site_supp_files_folder):
                print("Site not found %s" % site_path, flush=True)
        else:
            print("Building trajectories %s" % site_path, flush=True)

        process_site_build_trajectory(site_path, site_segmentation_path, site_supp_files_folder)
    return


def assemble_VAE(paths):
    """ Wrapper method for prepare dataset for VAE encoding

    This function loads data from multiple sites, adjusts intensities to correct
    for batch effect, and assembles into dataset for model prediction

    Resulting dataset will be saved in segmentation result folder, including:
        "*_file_paths.pkl": list of cell identification strings
        "*_static_patches.pt": all static patches from a given well
        "*_adjusted_static_patches.pt": all static patches from a given well 
            after adjusting phase/retardance intensities (avoid batch effect)

    Args:
        paths (list): list of paths, containing:
            0 - folder for raw data and segmentation results (in .npy)
            1 - folder for supplementary data
            2 - deprecated
            3 - list of site names, sites should be from a single well

    """

    # these sites should be from a single condition (C5, C4, B-wells, etc..)
    temp_folder, supp_folder, target, sites = paths[0], paths[1], paths[2], paths[3]

    dat_fs = []

    # Prepare dataset for VAE
    dat_fs = []
    for site in sites:
        supp_files_folder = os.path.join(supp_folder, '%s-supps' % site[:2], '%s' % site)
        dat_fs.extend([os.path.join(supp_files_folder, f) for f in os.listdir(supp_files_folder) if f.startswith('stacks')])

    dataset, fs = prepare_dataset_v2(dat_fs, cs=[0, 1])
    assert fs == sorted(fs)
    
    with open(os.path.join(temp_folder, '%s_file_paths.pkl' % sites[0][:2]), 'wb') as f:
        print(f"\tsaving {os.path.join(temp_folder, '%s_file_paths.pkl' % sites[0][:2])}")
        pickle.dump(fs, f)

    print(f"\tsaving {os.path.join(temp_folder, '%s_static_patches.pt' % sites[0][:2])}")
    torch.save(dataset, os.path.join(temp_folder, '%s_static_patches.pt' % sites[0][:2]))

    # Adjust channel mean/std
    # phase: 0.4980 plus/minus 0.0257
    # retardance: 0.0285 plus/minus 0.0261, only adjust for mean
    phase_slice = dataset.tensors[0][:, 0]
    phase_slice = ((phase_slice - phase_slice.mean()) / phase_slice.std()) * 0.0257 + 0.4980
    retard_slice = dataset.tensors[0][:, 1]
    retard_slice = retard_slice / retard_slice.mean() * 0.0285
    adjusted_dataset = TensorDataset(torch.stack([phase_slice, retard_slice], 1))
    print(f"\tsaving {os.path.join(temp_folder, '%s_adjusted_static_patches.pt' % sites[0][:2])}")
    torch.save(adjusted_dataset, os.path.join(temp_folder, '%s_adjusted_static_patches.pt' % sites[0][:2]))
    return


def process_VAE(paths):
    """ Wrapper method for VAE encoding

    This function loads prepared dataset and applies trained VAE and principal
    component extractors to encode static patches for each well, trained models
    are loaded from:
        VQ-VAE: 'HiddenStateExtractor/save_0005_bkp4.pt'
        pca: 'HiddenStateExtractor/pca_save.pkl'

    Resulting morphology descriptors will be saved in segmentation result 
    folder, including:
        "*_latent_space.pkl": array of latent vectors (before quantization)
            of individual static cells
        "*_latent_space_PCAed.pkl": array of top PCs of latent vectors (before 
            quantization)
        "*_latent_space_after.pkl": array of latent vectors (after quantization)
        "*_latent_space_after_PCAed.pkl": array of top PCs of latent vectors 
            (after quantization)

    Args:
        paths (list): list of paths, containing:
            0 - folder for raw data and segmentation results (in .npy)
            1 - folder for supplementary data
            2 - deprecated
            3 - list of site names, sites should be from a single well

    """
    # these sites should be from a single condition (C5, C4, B-wells, etc..)
    temp_folder, supp_folder, target, sites = paths[0], paths[1], paths[2], paths[3]

    assert len(set(s[:2] for s in sites)) == 1
    well = sites[0][:2]
    print(f"\tloading file paths {os.path.join(temp_folder, '%s_file_paths.pkl' % well)}")
    fs = pickle.load(open(os.path.join(temp_folder, '%s_file_paths.pkl' % well), 'rb'))

    print(f"\tloading static patches {os.path.join(temp_folder, '%s_adjusted_static_patches.pt')}")
    dataset = torch.load(os.path.join(temp_folder, '%s_adjusted_static_patches.pt' % well))
    dataset = rescale(dataset)
    
    model = VQ_VAE(alpha=0.0005, gpu=True)
    model = model.cuda()
    if target:
        model.load_state_dict(torch.load(target))
    else:
        model.load_state_dict(torch.load('HiddenStateExtractor/save_0005_bkp4.pt'))

    z_bs = {}
    z_as = {}
    for i in range(len(dataset)):
        sample = dataset[i:(i+1)][0].cuda()
        z_b = model.enc(sample)
        z_a, _, _ = model.vq(z_b)
        f_n = fs[i]
        z_bs[f_n] = z_b.cpu().data.numpy()
        z_as[f_n] = z_a.cpu().data.numpy()      

    # it's not clear where pca_save is created.  Will leave hard-coded path here for now
    try:
        pca = pickle.load(open('HiddenStateExtractor/pca_save.pkl', 'rb'))
    except Exception as ex:
        pca = None
        print("no saved PCA found at HiddenStateExtractor/pca_save.pkl'")

    dats = np.stack([z_bs[f] for f in fs], 0).reshape((len(dataset), -1))
    with open(os.path.join(temp_folder, '%s_latent_space.pkl' % well), 'wb') as f:
        print(f"\tsaving {os.path.join(temp_folder, '%s_latent_space.pkl' % well)}")
        pickle.dump(dats, f)
    if pca:
        dats_ = pca.transform(dats)
        with open(os.path.join(temp_folder, '%s_latent_space_PCAed.pkl' % well), 'wb') as f:
            print(f"\tsaving {os.path.join(temp_folder, '%s_latent_space_PCAed.pkl' % well)}")
            pickle.dump(dats_, f)
    
    dats = np.stack([z_as[f] for f in fs], 0).reshape((len(dataset), -1))
    with open(os.path.join(temp_folder, '%s_latent_space_after.pkl' % well), 'wb') as f:
        print(f"\tsaving {os.path.join(temp_folder, '%s_latent_space_after.pkl' % well)}")
        pickle.dump(dats, f)
    if pca:
        dats_ = pca.transform(dats)
        with open(os.path.join(temp_folder, '%s_latent_space_after_PCAed.pkl' % well), 'wb') as f:
            print(f"\tsaving {os.path.join(temp_folder, '%s_latent_space_after_PCAed.pkl' % well)}")
            pickle.dump(dats_, f)
    
    return


def trajectory_matching(paths):
    """ Helper function for assembling frame IDs to trajectories

    This function loads saved static frame identifiers ("*_file_paths.pkl") and 
    cell trajectories ("cell_traj.pkl" in supplementary data folder) and assembles
    list of frame IDs for each trajectory

    Results will be saved in segmentation result folder, including:
        "*_trajectories.pkl": list of frame IDs

    Args:
        paths (list): list of paths, containing:
            0 - folder for raw data and segmentation results (in .npy)
            1 - folder for supplementary data
            2 - deprecated
            3 - list of site names, sites should be from a single well

    """

    temp_folder, supp_folder, target, sites = paths[0], paths[1], paths[2], paths[3]
    
    wells = set(site[:2] for site in sites)
    assert len(wells) == 1
    well = list(wells)[0]

    print(f"\tloading file_paths {os.path.join(temp_folder, '%s_file_paths.pkl' % well)}")
    fs = pickle.load(open(os.path.join(temp_folder, '%s_file_paths.pkl' % well), 'rb'))

    site_trajs = {}
    for site in sites:
        site_supp_files_folder = os.path.join(supp_folder, '%s-supps' % well, '%s' % site)
        print(f"\tloading cell_traj {os.path.join(site_supp_files_folder, 'cell_traj.pkl')}")
        trajs = pickle.load(open(os.path.join(site_supp_files_folder, 'cell_traj.pkl'), 'rb'))
        for i, t in enumerate(trajs[0]):
            name = site + '/' + str(i)
            traj = []
            for t_point in sorted(t.keys()):
                frame_name = os.path.join(site_supp_files_folder, '%d_%d.h5' % (t_point, t[t_point]))
                if frame_name in fs:
                    traj.append(fs.index(frame_name))
            if len(traj) > 0.95 * len(t):
                site_trajs[name] = traj
          
    with open(os.path.join(temp_folder, '%s_trajectories.pkl' % well), 'wb') as f:
        print(f"\twriting trajectories {os.path.join(temp_folder, '%s_trajectories.pkl' % well)}")
        pickle.dump(site_trajs, f)