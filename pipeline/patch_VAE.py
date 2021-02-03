# bchhun, {2020-02-21}

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import pickle
import torch
import re
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset

from SingleCellPatch.generate_trajectories import process_site_build_trajectory
from SingleCellPatch.extract_patches import process_site_extract_patches, im_adjust

from run_training import VQ_VAE_z32, prepare_dataset_v2, zscore


def extract_patches(paths):
    """ Helper function for patch extraction

    Wrapper method `process_site_extract_patches` will be called, which 
    extracts individual cells from static frames for each site.

    Results will be saved in the supplementary data folder, including:
        "stacks_*.pkl": single cell patches for each time slice

    Args:
        paths (list): list of paths, containing:
            0 - folder for raw data, segmentation and summarized results
            1 - folder for supplementary data
            2 - path to model weight (not used in this method)
            3 - list of site names

    """

    summary_folder, supp_folder, model_path, sites = paths[0], paths[1], paths[2], paths[3]

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
                                         window_size=256,
                                         save_fig=True)
    return


def build_trajectories(paths):
    """ Helper function for trajectory building

    Wrapper method `process_site_build_trajectory` will be called, which 
    connects and generates trajectories from individual cell identifications.

    Results will be saved in the supplementary data folder, including:
        "cell_traj.pkl": list of trajectories and list of trajectory positions
            trajectories are dict of t_point: cell ID
            trajectory positions are dict of t_point: cell center position

    Args:
        paths (list): list of paths, containing:
            0 - folder for raw data, segmentation and summarized results
            1 - folder for supplementary data
            2 - path to model weight (not used in this method)
            3 - list of site names

    """

    summary_folder, supp_folder, model_path, sites = paths[0], paths[1], paths[2], paths[3]

    for site in sites:
        site_path = os.path.join(summary_folder + '/' + site + '.npy')
        site_supp_files_folder = os.path.join(supp_folder, '%s-supps' % site[:2], '%s' % site)
        if not os.path.exists(site_path) or not os.path.exists(site_supp_files_folder):
            print("Site data not found %s" % site_path, flush=True)
        else:
            print("Building trajectories %s" % site_path, flush=True)
            process_site_build_trajectory(site_supp_files_folder)

    return


def generate_trajectory_relations(paths):
    """ Find pair relations (adjacent frame, same trajectory) in static patches

    Results will be saved under `raw_folder`

    Args:
        sites (list of str): sites from the same well
        raw_folder (str): path to save image stacks, segmentation stacks, etc.
        supp_folder (str): path to save supplementary data

    """

    raw_folder, supp_folder, model_path, sites = paths[0], paths[1], paths[2], paths[3]

    assert len(set(s[:2] for s in sites)) == 1
    well = sites[0][:2]
    fs = pickle.load(open(os.path.join(raw_folder, "%s_file_paths.pkl" % well), 'rb'))
    relations = {}

    def get_patch_id(fs, key):
        """Return the index of the patch given its key"""
        inds = []
        for i, f in enumerate(fs):
            if key in f:
                inds.append(i)
        return inds[0] if len(inds) == 1 else None

    for site in sites:
        print('site:', site)
        trajectories = pickle.load(open(
            os.path.join(supp_folder, "%s-supps" % well, site, "cell_traj.pkl"), 'rb'))[0]
        print('trajectories:', trajectories)
        # print('fs:', fs)
        for trajectory in trajectories:
            t_ids = sorted(trajectory.keys())
            patch_ids = []
            for t_idx in t_ids:
                # get reference patch ID
                ref_patch_id = get_patch_id(fs, '/%s/%d_%d.' % (site, t_idx, trajectory[t_idx]))
                if ref_patch_id is None:
                    print('/%s/%d_%d.' % (site, t_idx, trajectory[t_idx]))
                assert not ref_patch_id is None
                patch_ids.append(ref_patch_id)
                # Adjacent frames
                if t_idx - 1 in t_ids:
                    adj_patch_id = get_patch_id(fs, '/%s/%d_%d.' % (site, t_idx - 1, trajectory[t_idx - 1]))
                    relations[(ref_patch_id, adj_patch_id)] = 2
                if t_idx + 1 in t_ids:
                    adj_patch_id = get_patch_id(fs, '/%s/%d_%d.' % (site, t_idx + 1, trajectory[t_idx + 1]))
                    relations[(ref_patch_id, adj_patch_id)] = 2

            # Same trajectory
            for i in patch_ids:
                for j in patch_ids:
                    if not (i, j) in relations:
                        relations[(i, j)] = 1

    print('relations:', relations)
    with open(os.path.join(raw_folder, "%s_static_patches_relations.pkl" % well), 'wb') as f:
        pickle.dump(relations, f)
    return


def assemble_VAE(paths):
    """ Wrapper method for prepare dataset for VAE encoding

    This function loads data from multiple sites, adjusts intensities to correct
    for batch effect, and assembles into dataset for model prediction

    Resulting dataset will be saved in the summary folder, including:
        "*_file_paths.pkl": list of cell identification strings
        "*_static_patches.pt": all static patches from a given well
        "*_adjusted_static_patches.pt": all static patches from a given well 
            after adjusting phase/retardance intensities (avoid batch effect)

    Args:
        paths (list): list of paths, containing:
            0 - folder for raw data, segmentation and summarized results
            1 - folder for supplementary data
            2 - path to model weight (not used in this method)
            3 - list of site names

    """

    # these sites should be from a single condition (C5, C4, B-wells, etc..)
    summary_folder, supp_folder, model_path, sites = paths[0], paths[1], paths[2], paths[3]
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

    dataset, fs = prepare_dataset_v2(dat_fs, cs=[0, 1])
    assert fs == sorted(fs)
    
    print(f"\tsaving {os.path.join(summary_folder, '%s_file_paths.pkl' % well)}")
    with open(os.path.join(summary_folder, '%s_file_paths.pkl' % well), 'wb') as f:
        pickle.dump(fs, f)

    # print(f"\tsaving {os.input_path.join(summary_folder, '%s_static_patches.pt' % well)}")
    # torch.save(dataset, os.input_path.join(summary_folder, '%s_static_patches.pt' % well))

    print(f"\tsaving {os.path.join(summary_folder, '%s_static_patches.pkl' % well)}")
    with open(os.path.join(summary_folder, '%s_static_patches.pkl' % well), 'wb') as f:
        pickle.dump(dataset, f, protocol=4)

    # Adjust channel mean/std
    # phase: 0.4980 plus/minus 0.0257
    # retardance: 0.0285 plus/minus 0.0261, only adjust for mean
    # phase_slice = dataset.tensors[0][:, 0]
    # phase_slice = ((phase_slice - phase_slice.mean()) / phase_slice.std()) * 0.0257 + 0.4980
    # retard_slice = dataset.tensors[0][:, 1]
    # retard_slice = retard_slice / retard_slice.mean() * 0.0285
    # adjusted_dataset = TensorDataset(torch.stack([phase_slice, retard_slice], 1))
    # print(f"\tsaving {os.input_path.join(summary_folder, '%s_adjusted_static_patches.pt' % well)}")
    # torch.save(adjusted_dataset, os.input_path.join(summary_folder, '%s_adjusted_static_patches.pt' % well))
    return


def process_VAE(paths, save_ouput=True):
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
        paths (list): list of paths, containing:
            0 - folder for raw data, segmentation and summarized results
            1 - folder for supplementary data
            2 - path to VQ-VAE model weight
            3 - list of site names

    """
    #TODO: add pooling datasets features and remove hardcoded normalization constants
    channel_mean = [0.49998672, 0.007081]
    channel_std = [0.00074311, 0.00906428]
    # these sites should be from a single condition (C5, C4, B-wells, etc..)
    summary_folder, supp_folder, model_dir, sites = paths[0], paths[1], paths[2], paths[3]
    model_path = os.path.join(model_dir, 'model.pt')
    model_name = os.path.basename(model_dir)
    output_dir = os.path.join(summary_folder, model_name)
    os.makedirs(output_dir, exist_ok=True)

    assert len(set(site[:2] for site in sites)) == 1, \
        "Sites should be from a single well/condition"
    well = sites[0][:2]

    print(f"\tloading file paths {os.path.join(summary_folder, '%s_file_paths.pkl' % well)}")
    fs = pickle.load(open(os.path.join(summary_folder, '%s_file_paths.pkl' % well), 'rb'))

    # print(f"\tloading static patches {os.supp_dir.join(raw_dir, '%s_adjusted_static_patches.pt' % well)}")
    # dataset = torch.load(os.supp_dir.join(raw_dir, '%s_adjusted_static_patches.pt' % well))
    print(f"\tloading static patches {os.path.join(summary_folder, '%s_static_patches.pkl' % well)}")
    dataset = pickle.load(open(os.path.join(summary_folder, '%s_static_patches.pkl' % well), 'rb'))
    dataset = zscore(dataset, channel_mean=channel_mean, channel_std=channel_std)
    dataset = TensorDataset(torch.from_numpy(dataset).float())
    search_obj = re.search(r'nh(\d+)_nrh(\d+)_ne(\d+).*', model_name)
    num_hiddens = int(search_obj.group(1))
    num_residual_hiddens = int(search_obj.group(2))
    num_embeddings = int(search_obj.group(3))
    # commitment_cost = float(search_obj.group(4))
    model = VQ_VAE_z32(num_inputs=2,
                       num_hiddens=num_hiddens,
                       num_residual_hiddens=num_residual_hiddens,
                       num_residual_layers=2,
                       num_embeddings=num_embeddings,
                       gpu=True)
    model = model.cuda()
    try:
        if not model_path is None:
            model.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(torch.load('HiddenStateExtractor/save_0005_bkp4.pt'))
    except Exception as ex:
        print(ex)
        raise ValueError("Error in loading model weights for VQ-VAE")

    z_bs = {}
    z_as = {}
    for i in range(len(dataset)):
        sample = dataset[i:(i+1)][0].cuda()
        z_b = model.enc(sample)
        z_a, _, _ = model.vq(z_b)
        f_n = fs[i]
        z_bs[f_n] = z_b.cpu().data.numpy()
        z_as[f_n] = z_a.cpu().data.numpy()      

    dats = np.stack([z_bs[f] for f in fs], 0).reshape((len(dataset), -1))
    print(f"\tsaving {os.path.join(output_dir, '%s_latent_space.pkl' % well)}")
    with open(os.path.join(output_dir, '%s_latent_space.pkl' % well), 'wb') as f:
        pickle.dump(dats, f, protocol=4)
    
    dats = np.stack([z_as[f] for f in fs], 0).reshape((len(dataset), -1))
    print(f"\tsaving {os.path.join(output_dir, '%s_latent_space_after.pkl' % well)}")
    with open(os.path.join(output_dir, '%s_latent_space_after.pkl' % well), 'wb') as f:
        pickle.dump(dats, f, protocol=4)

    if save_ouput:
        np.random.seed(0)
        random_inds = np.random.randint(0, len(dataset), (10,))
        for i in random_inds:
            sample = dataset[i:(i + 1)][0].cuda()
            output = model(sample)[0]
            im_phase = im_adjust(sample[0, 0].cpu().data.numpy())
            im_phase_recon = im_adjust(output[0, 0].cpu().data.numpy())
            im_retard = im_adjust(sample[0, 1].cpu().data.numpy())
            im_retard_recon = im_adjust(output[0, 1].cpu().data.numpy())
            n_rows = 2
            n_cols = 2
            fig, ax = plt.subplots(n_rows, n_cols, squeeze=False)
            ax = ax.flatten()
            fig.set_size_inches((15, 5 * n_rows))
            axis_count = 0
            for im, name in zip([im_phase, im_phase_recon, im_retard, im_retard_recon],
                                ['phase', 'phase_recon', 'im_retard', 'retard_recon']):
                ax[axis_count].imshow(np.squeeze(im), cmap='gray')
                ax[axis_count].axis('off')
                ax[axis_count].set_title(name, fontsize=12)
                axis_count += 1
            fig.savefig(os.path.join(output_dir, 'recon_%d.jpg' % i),
                        dpi=300, bbox_inches='tight')
            plt.close(fig)


def trajectory_matching(paths):
    """ Helper function for assembling frame IDs to trajectories

    This function loads saved static frame identifiers ("*_file_paths.pkl") and 
    cell trajectories ("cell_traj.pkl" in supplementary data folder) and assembles
    list of frame IDs for each trajectory

    Results will be saved in the summary folder, including:
        "*_trajectories.pkl": list of frame IDs

    Args:
        paths (list): list of paths, containing:
            0 - folder for raw data, segmentation and summarized results
            1 - folder for supplementary data
            2 - path to model weight (not used in this method)
            3 - list of site names

    """

    summary_folder, supp_folder, model_path, sites = paths[0], paths[1], paths[2], paths[3]
    assert len(set(site[:2] for site in sites)) == 1, \
        "Sites should be from a single well/condition"
    well = sites[0][:2]

    print(f"\tloading file_paths {os.path.join(summary_folder, '%s_file_paths.pkl' % well)}")
    fs = pickle.load(open(os.path.join(summary_folder, '%s_file_paths.pkl' % well), 'rb'))

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
          
    with open(os.path.join(summary_folder, '%s_trajectories.pkl' % well), 'wb') as f:
        print(f"\twriting trajectories {os.path.join(summary_folder, '%s_trajectories.pkl' % well)}")
        pickle.dump(site_trajs, f)