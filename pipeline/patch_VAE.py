# bchhun, {2020-02-21}

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import pickle
import torch
import numpy as np
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset

from SingleCellPatch.extract_patches import process_site_extract_patches
from SingleCellPatch.generate_trajectories import process_site_build_trajectory, process_well_generate_trajectory_relations

from HiddenStateExtractor.vq_vae import VQ_VAE
from HiddenStateExtractor.vq_vae_supp import prepare_dataset_v2, vae_preprocess


def extract_patches(summary_folder: str,
                    supp_folder: str,
                    channels: list,
                    model_path: str,
                    sites: list,
                    window_size: int = 256,
                    save_fig: bool = False,
                    reload: bool = True,
                    skip_boundary: bool = False,
                    **kwargs):
    """ Helper function for patch extraction

    Wrapper method `process_site_extract_patches` will be called, which 
    extracts individual cells from static frames for each site.

    Results will be saved in the supplementary data folder, including:
        "stacks_*.pkl": single cell patches for each time slice

    Args:
        summary_folder (str): folder for raw data, segmentation and 
            summarized results
        supp_folder (str): folder for supplementary data
        channels (list of int): indices of channels used for segmentation 
            (not used, by default all channels should be saved)
        model_path (str, optional): path to model weight (not used)
        sites (list of str): list of site names
        window_size (int, optional): default=256, x, y size of the patch
        save_fig (bool, optional): if to save extracted patches (with 
            segmentation mask)
        reload (bool, optional): if to load existing stack dat files
        skip_boundary (bool, optional): if to skip patches whose edges exceed
            the image size (do not pad)

    """

    for site in sites:
        site_path = os.path.join(summary_folder + '/' + site + '.npy')
        site_segmentation_path = os.path.join(summary_folder, '%s_NNProbabilities.npy' % site)
        site_supp_files_folder = os.path.join(supp_folder, '%s-supps' % site[:2], '%s' % site)
        if not os.path.exists(site_path) or \
            not os.path.exists(site_segmentation_path) or \
            not os.path.exists(site_supp_files_folder):
                print("Site data not found %s" % site_path, flush=True)
        else:
            print("Building patches %s" % site_path, flush=True)

            process_site_extract_patches(site_path, 
                                         site_segmentation_path, 
                                         site_supp_files_folder,
                                         window_size=window_size,
                                         save_fig=save_fig,
                                         reload=reload,
                                         skip_boundary=skip_boundary,
                                         **kwargs)
    return


def build_trajectories(summary_folder: str,
                       supp_folder: str,
                       channels: list,
                       model_path: str,
                       sites: list,
                       **kwargs):
    """ Helper function for trajectory building

    Wrapper method `process_site_build_trajectory` will be called, which 
    connects and generates trajectories from individual cell identifications.

    Results will be saved in the supplementary data folder, including:
        "cell_traj.pkl": list of trajectories and list of trajectory positions
            trajectories are dict of t_point: cell ID
            trajectory positions are dict of t_point: cell center position

    Args:
        summary_folder (str): folder for raw data, segmentation and 
            summarized results
        supp_folder (str): folder for supplementary data
        channels (list of int): indices of channels used for segmentation 
            (not used)
        model_path (str, optional): path to model weight (not used)
        sites (list of str): list of site names

    """

    for site in sites:
        site_path = os.path.join(summary_folder + '/' + site + '.npy')
        site_supp_files_folder = os.path.join(supp_folder, '%s-supps' % site[:2], '%s' % site)
        if not os.path.exists(site_path) or not os.path.exists(site_supp_files_folder):
            print("Site data not found %s" % site_path, flush=True)
        else:
            print("Building trajectories %s" % site_path, flush=True)
            process_site_build_trajectory(site_supp_files_folder, **kwargs)
    return


def assemble_VAE(summary_folder: str,
                 supp_folder: str,
                 channels: list,
                 model_path: str,
                 sites: list,
                 **kwargs):
    """ Wrapper method for prepare dataset for VAE encoding

    This function loads data from multiple sites, adjusts intensities to correct
    for batch effect, and assembles into dataset for model prediction

    Resulting dataset will be saved in the summary folder, including:
        "*_file_paths.pkl": list of cell identification strings
        "*_static_patches.pt": all static patches from a given well
        "*_adjusted_static_patches.pt": all static patches from a given well 
            after adjusting phase/retardance intensities (avoid batch effect)

    Args:
        summary_folder (str): folder for raw data, segmentation and 
            summarized results
        supp_folder (str): folder for supplementary data
        channels (list of int): indices of channels used for segmentation 
            (not used)
        model_path (str, optional): path to model weight (not used)
        sites (list of str): list of site names

    """

    # sites should be from a single condition (C5, C4, B-wells, etc..)
    assert len(set(site[:2] for site in sites)) == 1, \
        "Sites should be from a single well/condition"
    well = sites[0][:2]

    dat_fs = []

    # Prepare dataset for VAE
    dat_fs = []
    for site in sites:
        supp_files_folder = os.path.join(supp_folder, '%s-supps' % site[:2], '%s' % site)
        dat_fs.extend([os.path.join(supp_files_folder, f) \
            for f in os.listdir(supp_files_folder) if f.startswith('stacks')])

    dataset, fs = prepare_dataset_v2(dat_fs, cs=channels)
    assert fs == sorted(fs)
    
    print(f"\tsaving {os.path.join(summary_folder, '%s_file_paths.pkl' % well)}")
    with open(os.path.join(summary_folder, '%s_file_paths.pkl' % well), 'wb') as f:
        pickle.dump(fs, f)

    print(f"\tsaving {os.path.join(summary_folder, '%s_static_patches.pt' % well)}")
    torch.save(dataset, os.path.join(summary_folder, '%s_static_patches.pt' % well))
    
    well_supp_files_folder = os.path.join(supp_folder, '%s-supps' % well)
    relations = process_well_generate_trajectory_relations(fs, sites, well_supp_files_folder)
    with open(os.path.join(summary_folder, "%s_static_patches_relations.pkl" % well), 'wb') as f:
        pickle.dump(relations, f)
    
    return


def trajectory_matching(summary_folder: str,
                        supp_folder: str,
                        channels: list,
                        model_path: str,
                        sites: list,
                        **kwargs):
    """ Helper function for assembling frame IDs to trajectories

    This function loads saved static frame identifiers ("*_file_paths.pkl") and 
    cell trajectories ("cell_traj.pkl" in supplementary data folder) and assembles
    list of frame IDs for each trajectory

    Results will be saved in the summary folder, including:
        "*_trajectories.pkl": list of frame IDs

    Args:
        summary_folder (str): folder for raw data, segmentation and 
            summarized results
        supp_folder (str): folder for supplementary data
        channels (list of int): indices of channels used for segmentation 
            (not used)
        model_path (str, optional): path to model weight (not used)
        sites (list of str): list of site names

    """
    
    assert len(set(site[:2] for site in sites)) == 1, \
        "Sites should be from a single well/condition"
    well = sites[0][:2]

    print(f"\tloading file_paths {os.path.join(summary_folder, '%s_file_paths.pkl' % well)}")
    fs = pickle.load(open(os.path.join(summary_folder, '%s_file_paths.pkl' % well), 'rb'))

    def patch_name_to_tuple(f):
        f = [seg for seg in f.split('/') if len(seg) > 0]
        site_name = f[-2]
        assert site_name in sites
        t_point = int(f[-1].split('_')[0])
        cell_id = int(f[-1].split('_')[1].split('.')[0])
        return (site_name, t_point, cell_id)
    patch_id_mapping = {patch_name_to_tuple(f): i for i, f in enumerate(fs)}
    
    site_trajs = {}
    for site in sites:
        site_supp_files_folder = os.path.join(supp_folder, '%s-supps' % well, '%s' % site)
        print(f"\tloading cell_traj {os.path.join(site_supp_files_folder, 'cell_traj.pkl')}")
        trajs = pickle.load(open(os.path.join(site_supp_files_folder, 'cell_traj.pkl'), 'rb'))
        for i, t in enumerate(trajs[0]):
            name = site + '/' + str(i)
            traj = []
            for t_point in sorted(t.keys()):
                frame_id = patch_id_mapping[(site, t_point, t[t_point])]
                if not frame_id is None:
                    traj.append(frame_id)
            if len(traj) > 0.95 * len(t):
                site_trajs[name] = traj
          
    with open(os.path.join(summary_folder, '%s_trajectories.pkl' % well), 'wb') as f:
        print(f"\twriting trajectories {os.path.join(summary_folder, '%s_trajectories.pkl' % well)}")
        pickle.dump(site_trajs, f)
    return


def process_VAE(summary_folder: str,
                supp_folder: str,
                channels: list,
                model_path: str,
                sites: list,
                input_clamp: list = [0., 1.],
                save_output: bool = True,
                **kwargs):
    """ Wrapper method for VAE encoding

    This function loads prepared dataset and applies trained VAE to encode 
    static patches for each well. 

    Model weight path should be provided, if not a default path will be used:
        VQ-VAE: 'HiddenStateExtractor/save_0005_bkp4.pt'

    Resulting latent vectors will be saved in the summary folder, including:
        "*_latent_space.pkl": array of latent vectors (before quantization)
            of individual static cells
        "*_latent_space_after.pkl": array of latent vectors (after quantization)

    Args:
        summary_folder (str): folder for raw data, segmentation and 
            summarized results
        supp_folder (str): folder for supplementary data
        channels (list of int): indices of channels used for VAE encoding
        model_path (str): path to model weight
        sites (list of str): list of site names
        input_clamp (list of float or None): if given, the lower/upper limit 
            of input patches

    """

    assert len(set(site[:2] for site in sites)) == 1, \
        "Sites should be from a single well/condition"
    well = sites[0][:2]
    
    preprocess_setting={
        0: ("normalize", 0.4, 0.05), # Phase
        1: ("scale", 0.05), # Retardance
        2: ("normalize", 0.5, 0.05), # Brightfield
    }

    print(f"\tloading file paths {os.path.join(summary_folder, '%s_file_paths.pkl' % well)}")
    fs = pickle.load(open(os.path.join(summary_folder, '%s_file_paths.pkl' % well), 'rb'))

    print(f"\tloading static patches {os.path.join(summary_folder, '%s_static_patches.pt' % well)}")
    dataset = torch.load(os.path.join(summary_folder, '%s_static_patches.pt' % well))
    dataset = vae_preprocess(dataset, 
                             use_channels=channels, 
                             preprocess_setting=preprocess_setting,
                             clamp=input_clamp)
    
    model = VQ_VAE(alpha=0.0005, gpu=True)
    model = model.cuda()
    try:
        if not model_path is None:
            model.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(torch.load('HiddenStateExtractor/save_0005_bkp4.pt'))
    except Exception as ex:
        print(ex)
        raise ValueError("Error in loading model weights for VQ-VAE")

    _, n_channels, n_z, x_size, y_size = dataset.tensors[0].shape
    z_bs = {}
    z_as = {}
    for i in range(len(dataset)):
        sample = dataset[i:(i+1)][0]
        sample = sample.reshape([-1, n_channels, x_size, y_size]).cuda()
        z_b = model.enc(sample)
        z_a, _, _ = model.vq(z_b)
        f_n = fs[i]
        z_bs[f_n] = z_b.cpu().data.numpy()
        z_as[f_n] = z_a.cpu().data.numpy()

    dats = np.stack([z_bs[f] for f in fs], 0).reshape((len(dataset), -1))
    print(f"\tsaving {os.path.join(summary_folder, '%s_latent_space.pkl' % well)}")
    with open(os.path.join(summary_folder, '%s_latent_space.pkl' % well), 'wb') as f:
        pickle.dump(dats, f)
    
    dats = np.stack([z_as[f] for f in fs], 0).reshape((len(dataset), -1))
    print(f"\tsaving {os.path.join(summary_folder, '%s_latent_space_after.pkl' % well)}")
    with open(os.path.join(summary_folder, '%s_latent_space_after.pkl' % well), 'wb') as f:
        pickle.dump(dats, f)

    return


# def process_PCA(summary_folder: str,
#                 supp_folder: str,
#                 channels: list,
#                 model_path: str,
#                 sites: list,
#                 **kwargs):
#     """ Wrapper method for PCA dimension reduction

#     This function loads latent vectors generated by VQ-VAE and applies trained
#     PCA to extract top PCs as morphology descriptors.

#     PCA weight path should be provided, if not a default path will be used:
#         pca: 'HiddenStateExtractor/pca_save.pkl'

#     Resulting morphology descriptors will be saved in the summary folder, 
#     including:
#         "*_latent_space_PCAed.pkl": array of top PCs of latent vectors (before 
#             quantization)
#         "*_latent_space_after_PCAed.pkl": array of top PCs of latent vectors 
#             (after quantization)

#     Args:
#         paths (list): list of paths, containing:
#             0 - folder for raw data, segmentation and summarized results
#             1 - folder for supplementary data
#             2 - path to PCA weight
#             3 - list of site names

#     """
#     # these sites should be from a single condition (C5, C4, B-wells, etc..)
#     summary_folder, supp_folder, model_path, sites = paths[0], paths[1], paths[2], paths[3]
#     assert len(set(site[:2] for site in sites)) == 1, \
#         "Sites should be from a single well/condition"
#     well = sites[0][:2]

#     try:
#         if not model_path is None:
#             pca = pickle.load(open(model_path, 'rb'))
#         else:
#             pca = pickle.load(open('HiddenStateExtractor/pca_save.pkl', 'rb'))
#     except Exception as ex:
#         print(ex)
#         raise ValueError("Error in loading pre-saved PCA weights")

#     dats = pickle.load(open(os.path.join(summary_folder, '%s_latent_space.pkl' % well), 'rb'))
#     dats_ = pca.transform(dats)
#     print(f"\tsaving {os.path.join(summary_folder, '%s_latent_space_PCAed.pkl' % well)}")
#     with open(os.path.join(summary_folder, '%s_latent_space_PCAed.pkl' % well), 'wb') as f:
#         pickle.dump(dats_, f)

#     dats = pickle.load(open(os.path.join(summary_folder, '%s_latent_space_after.pkl' % well), 'rb'))
#     dats_ = pca.transform(dats)
#     print(f"\tsaving {os.path.join(summary_folder, '%s_latent_space_after_PCAed.pkl' % well)}")
#     with open(os.path.join(summary_folder, '%s_latent_space_after_PCAed.pkl' % well), 'wb') as f:
#         pickle.dump(dats_, f)
#     return