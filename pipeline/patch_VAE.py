# bchhun, {2020-02-21}

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import pickle
import torch
import re
import numpy as np
import matplotlib.pyplot as plt
import importlib
import inspect
from torch.utils.data import TensorDataset

from SingleCellPatch.extract_patches import process_site_extract_patches, im_adjust
from SingleCellPatch.generate_trajectories import process_site_build_trajectory, process_well_generate_trajectory_relations

from run_training import VQ_VAE_z32, zscore
from HiddenStateExtractor.vq_vae import VQ_VAE
from HiddenStateExtractor.vq_vae_supp import prepare_dataset_v2, vae_preprocess

NETWORK_MODULE = 'run_training'

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
                 save_mask: bool=False,
                 mask_channels: list=[-2, -1],
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

    print(f"\tsaving {os.path.join(summary_folder, '%s_static_patches.pkl' % well)}")
    with open(os.path.join(summary_folder, '%s_static_patches.pkl' % well), 'wb') as f:
        pickle.dump(dataset, f, protocol=4)

    if save_mask:
        dataset_mask, fs_mask = prepare_dataset_v2(dat_fs, cs=mask_channels)
        assert fs_mask == fs
        print(f"\tsaving {os.path.join(summary_folder, '%s_static_patches_mask.pkl' % well)}")
        with open(os.path.join(summary_folder, '%s_static_patches_mask.pkl' % well), 'wb') as f:
            pickle.dump(dataset_mask, f, protocol=4)

    well_supp_files_folder = os.path.join(supp_folder, '%s-supps' % well)
    relations = process_well_generate_trajectory_relations(fs, sites, well_supp_files_folder)
    with open(os.path.join(summary_folder, "%s_static_patches_relations.pkl" % well), 'wb') as f:
        pickle.dump(relations, f)

    return


def combine_dataset(input_dataset_names, output_dataset_name, save_mask=True):
    """ Combine multiple datasets

    Args:
        input_dataset_names (list): list of input datasets
            named as $DIRECTORY/$DATASETNAME, this method reads files below:
                $DIRECTORY/$DATASETNAME_file_paths.pkl
                $DIRECTORY/$DATASETNAME_static_patches.pkl
                $DIRECTORY/$DATASETNAME_static_patches_relations.pkl
                $DIRECTORY/$DATASETNAME_static_patches_mask.pkl (if `save_mask`)
        output_dataset_name (str): path to the output save
            the combined dataset will be saved to the specified path with 
            corresponding suffix
        save_mask (bool, optional): if to read & save dataset mask

    """

    separate_fs = []
    separate_dataset = []
    separate_dataset_mask = []
    separate_relations = []

    for n in input_dataset_names:
        assert os.path.exists(n + '_file_paths.pkl')
        assert os.path.exists(n + '_static_patches.pkl')
        assert os.path.exists(n + '_static_patches_relations.pkl')
        separate_fs.append(pickle.load(open(n + '_file_paths.pkl', 'rb')))
        separate_dataset.append(pickle.load(open(n + '_static_patches.pkl', 'rb')))
        separate_relations.append(pickle.load(open(n + '_static_patches_relations.pkl', 'rb')))
        if save_mask:
            assert os.path.exists(n + '_static_patches_mask.pkl')
            separate_dataset_mask.append(pickle.load(open(n + '_static_patches_mask.pkl', 'rb')))
        else:
            separate_dataset_mask.append([None] * len(separate_fs[-1]))

    all_fs = sorted(sum(separate_fs, []))
    assert len(all_fs) == len(set(all_fs)), "Found patches with identical name"
    with open(output_dataset_name + '_file_paths.pkl', 'wb') as f:
        pickle.dump(all_fs, f)

    separate_name_to_idx = {}
    for i, dataset_f in enumerate(separate_fs):
        for j, n in enumerate(dataset_f):
            separate_name_to_idx[n] = (i, j)

    combined_name_to_idx = {}
    for i, n in enumerate(all_fs):
        combined_name_to_idx[n] = i

    all_dataset = []
    all_dataset_mask = []
    for n in all_fs:
        i, j = separate_name_to_idx[n]
        all_dataset.append(separate_dataset[i][j])
        all_dataset_mask.append(separate_dataset_mask[i][j])

    all_dataset = np.stack(all_dataset, 0)
    with open(output_dataset_name + '_static_patches.pkl', 'wb') as f:
        pickle.dump(all_dataset, f, protocol=4)

    if save_mask:
        all_dataset_mask = np.stack(all_dataset_mask, 0)
        with open(output_dataset_name + '_static_patches_mask.pkl', 'wb') as f:
            pickle.dump(all_dataset_mask, f, protocol=4)

    all_relations = {}
    for fs, relation in zip(separate_fs, separate_relations):
        for k in relation:
            name1 = fs[k[0]]
            name2 = fs[k[1]]
            all_relations[(combined_name_to_idx[name1],
                           combined_name_to_idx[name2])] = relation[k]

    with open(output_dataset_name + '_static_patches_relations.pkl', 'wb') as f:
        pickle.dump(all_relations, f)

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
        "*_trajectories.pkl": list of frame IDs, list of cell centroid positions, list of cell identification

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
    traj_profiles = {}

    for site in sites:
        print(site)
        path = os.path.join(supp_folder, '%s-supps' % well, '%s' % site, 'cell_traj.pkl')
        cell_trajectories_inds, cell_trajectories_positions = pickle.load(open(path, 'rb'))

        path = os.path.join(supp_folder, '%s-supps' % well, '%s' % site, 'cell_pixel_assignments.pkl')
        cell_pixel_assignments = pickle.load(open(path, 'rb'))

        path = os.path.join(summary_folder, '%s_NNProbabilities.npy' % site)
        segmentation_stack = np.load(path)
        for i, (t, t_pos) in enumerate(zip(cell_trajectories_inds, cell_trajectories_positions)):
            t_name = site + '/' + str(i)
            traj_ind = []
            traj_mg_ratio = []

            for t_point in sorted(t.keys()):
                cell_id = t[t_point]
                frame_id = patch_id_mapping[(site, t_point, cell_id)]
                traj_ind.append(frame_id)

                inds = np.where(cell_pixel_assignments[t_point][1] == cell_id)
                cell_pixels = cell_pixel_assignments[t_point][0][inds]
                cell_segmentation = segmentation_stack[t_point, :, :, cell_pixels[:, 0], cell_pixels[:, 1]]
                mg_ratio = (cell_segmentation[:, 1] > 0.5).sum() / cell_segmentation.shape[0]
                traj_mg_ratio.append(mg_ratio)

            traj_profiles[t_name] = (traj_ind, traj_mg_ratio, t, t_pos)

    with open(os.path.join(summary_folder, '%s_trajectory_profiles.pkl' % well), 'wb') as f:
        print(f"\twriting trajectory profiles {os.path.join(summary_folder, '%s_trajectory_profiles.pkl' % well)}")
        pickle.dump(traj_profiles, f)
    return


def import_object(module_name, obj_name, obj_type='class'):
    """Imports a class or function dynamically

    :param str module_name: modules such as input, utils, train etc
    :param str obj_name: Object to find
    :param str obj_type: Object type (class or function)
    """

    # full_module_name = ".".join(('dynamorph', module_name))
    full_module_name = module_name
    try:
        module = importlib.import_module(full_module_name)
        obj = getattr(module, obj_name)
        if obj_type == 'class':
            assert inspect.isclass(obj),\
                "Expected {} to be class".format(obj_name)
        elif obj_type == 'function':
            assert inspect.isfunction(obj),\
                "Expected {} to be function".format(obj_name)
        return obj
    except ImportError:
        raise

def process_VAE(summary_folder: str,
                supp_folder: str,
                channels: list,
                model_dir: str,
                sites: list,
                network: str= 'VQ_VAE_z16',
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
        model_dir (str): directory for model weight
        sites (list of str): list of site names
        input_clamp (list of float or None): if given, the lower/upper limit
            of input patches

    """
    #TODO: add pooling datasets features and remove hardcoded normalization constants
    # ideally normalization parameters should be determined from pooled training data,
    # For inference same normalization parameters can be used or determined from the inference data,
    # depending on if the inference data has the same distribution as training data

    # these sites should be from a single condition (C5, C4, B-wells, etc..)
    model_path = os.path.join(model_dir, 'model.pt')
    model_name = os.path.basename(model_dir)
    output_dir = os.path.join(summary_folder, model_name)
    os.makedirs(output_dir, exist_ok=True)

    assert len(set(site[:2] for site in sites)) == 1, \
        "Sites should be from a single well/condition"
    well = sites[0][:2]
    # TODO: expose normalization parameters in train config
    #### cardiomyocyte data###
    # channel_mean = [0.49998672, 0.007081]
    # channel_std = [0.00074311, 0.00906428]

    ### microglia data####
    # channel_mean = [0.4, 0, 0.5]
    # channel_std = [0.05, 0.05, 0.05]

    ### estimate mean and std from the data ###
    channel_mean = None
    channel_std = None

    print(f"\tloading file paths {os.path.join(summary_folder, '%s_file_paths.pkl' % well)}")
    fs = pickle.load(open(os.path.join(summary_folder, '%s_file_paths.pkl' % well), 'rb'))

    # dataset = torch.load(os.path.join(summary_folder, '%s_static_patches.pt' % well))
    # dataset = vae_preprocess(dataset,
    #                          use_channels=channels,
    #                          preprocess_setting=preprocess_setting,
    #                          clamp=input_clamp)
    #
    print(f"\tloading static patches {os.path.join(summary_folder, '%s_static_patches.pkl' % well)}")
    dataset = pickle.load(open(os.path.join(summary_folder, '%s_static_patches.pkl' % well), 'rb'))
    dataset = zscore(dataset, channel_mean=channel_mean, channel_std=channel_std)
    dataset = TensorDataset(torch.from_numpy(dataset).float())
    search_obj = re.search(r'nh(\d+)_nrh(\d+)_ne(\d+).*', model_name)
    num_hiddens = int(search_obj.group(1))
    num_residual_hiddens = int(search_obj.group(2))
    num_embeddings = int(search_obj.group(3))
    # commitment_cost = float(search_obj.group(4))
    network_cls = import_object(NETWORK_MODULE, network)
    model = network_cls(num_inputs=2,
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
    print(f"\tsaving {os.path.join(output_dir, '%s_latent_space.pkl' % well)}")
    with open(os.path.join(output_dir, '%s_latent_space.pkl' % well), 'wb') as f:
        pickle.dump(dats, f, protocol=4)
    
    dats = np.stack([z_as[f] for f in fs], 0).reshape((len(dataset), -1))
    print(f"\tsaving {os.path.join(output_dir, '%s_latent_space_after.pkl' % well)}")
    with open(os.path.join(output_dir, '%s_latent_space_after.pkl' % well), 'wb') as f:
        pickle.dump(dats, f, protocol=4)

    if save_output:
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

