import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import pickle
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib
import inspect
from configs.config_reader import YamlReader
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from SingleCellPatch.extract_patches import process_site_extract_patches, im_adjust
from SingleCellPatch.generate_trajectories import process_site_build_trajectory, process_well_generate_trajectory_relations

from pipeline.train_utils import zscore, zscore_patch, ImageDataset
import HiddenStateExtractor.vae as vae
import HiddenStateExtractor.resnet as resnet
from HiddenStateExtractor.vq_vae_supp import assemble_patches, prepare_dataset_v2, vae_preprocess

NETWORK_MODULE = 'run_training'

def extract_patches(raw_folder: str,
                    supp_folder: str,
                    # channels: list,
                    sites: list,
                    config: YamlReader,
                    **kwargs):
    """ Helper function for patch extraction

    Wrapper method `process_site_extract_patches` will be called, which
    extracts individual cells from static frames for each site.

    Results will be saved in the supplementary data folder, including:
        "stacks_*.pkl": single cell patches for each time slice

    Args:
        raw_folder (str): folder for raw data, segmentation and
            summarized results
        supp_folder (str): folder for supplementary data
        sites (list of str): list of site names
        config (YamlReader): config file supplied at CLI
    """
    channels = config.patch.channels

    assert len(channels) > 0, "At least one channel must be specified"

    window_size = config.patch.window_size
    save_fig = config.patch.save_fig
    reload = config.patch.reload
    skip_boundary = config.patch.skip_boundary

    for site in sites:
        site_path = os.path.join(raw_folder + '/' + site + '.npy')
        site_segmentation_path = os.path.join(raw_folder, '%s_NNProbabilities.npy' % site)
        site_supp_files_folder = os.path.join(supp_folder, '%s-supps' % site[:2], '%s' % site)
        if not os.path.exists(site_path):
            print("Site data not found %s" % site_path, flush=True)
        if not os.path.exists(site_segmentation_path):
            print("Site data not found %s" % site_segmentation_path, flush=True)
        if not os.path.exists(site_supp_files_folder):
            print("Site supp folder not found %s" % site_supp_files_folder, flush=True)
        else:
            print("Building patches %s" % site_path, flush=True)

            process_site_extract_patches(site_path, 
                                         site_segmentation_path, 
                                         site_supp_files_folder,
                                         window_size=window_size,
                                         channels=channels,
                                         save_fig=save_fig,
                                         reload=reload,
                                         skip_boundary=skip_boundary,
                                         **kwargs)
    return


def build_trajectories(summary_folder: str,
                       supp_folder: str,
                       # channels: list,
                       sites: list,
                       config: YamlReader,
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
            process_site_build_trajectory(site_supp_files_folder, min_len=config.patch.min_length)
    return


def assemble_VAE(raw_folder: str,
                 supp_folder: str,
                 sites: list,
                 config: YamlReader,
                 patch_type: str='masked_mat',
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
        raw_folder (str): folder for raw data, segmentation and
            summarized results
        supp_folder (str): folder for supplementary data
        sites (list of str): list of site names
        config (YamlReader): config file supplied at CLI

    """

    channels = config.inference.channels

    assert len(channels) > 0, "At least one channel must be specified"

    # sites should be from a single condition (C5, C4, B-wells, etc..)
    assert len(set(site[:2] for site in sites)) == 1, \
        "Sites should be from a single well/condition"
    well = sites[0][:2]

    # Prepare dataset for VAE

    df_meta = pd.DataFrame()
    traj_id_offsets = {'time trajectory ID': 0, 'slice trajectory ID': 0}
    for site in sites:
        supp_files_folder = os.path.join(supp_folder, '%s-supps' % site[:2], '%s' % site)
        meta_path = os.path.join(supp_files_folder, 'patch_meta.csv')
        df_meta_site = pd.read_csv(meta_path, index_col=0, converters={
            'cell position': lambda x: np.fromstring(x.strip("[]"), sep=' ', dtype=np.int32)})
        # offset trajectory ids to make it unique
        for col in df_meta_site.columns:
            if col in traj_id_offsets:
                df_meta_site[col] += traj_id_offsets[col]
                traj_id_offsets[col] = df_meta_site[col].max() + 1
        df_meta = df_meta.append(df_meta_site, ignore_index=True)
    df_meta.reset_index(drop=True, inplace=True)
    meta_path = os.path.join(supp_folder, '%s-supps' % well, 'patch_meta.csv')
    df_meta.to_csv(meta_path, sep=',')
    dataset = assemble_patches(df_meta, supp_folder, channels=channels, key=patch_type)
    assert len(dataset) == len(df_meta), 'Number of patches and rows in metadata are not consistent.'
    print(f"\tsaving {os.path.join(raw_folder, '%s_static_patches.pkl' % well)}")
    with open(os.path.join(raw_folder, '%s_static_patches.pkl' % well), 'wb') as f:
        pickle.dump(dataset, f, protocol=4)
    relations, labels = process_well_generate_trajectory_relations(df_meta, track_dim='slice')
    print('len(labels):', len(labels))
    print('len(dataset):', len(dataset))
    with open(os.path.join(raw_folder, "%s_static_patches_relations.pkl" % well), 'wb') as f:
        pickle.dump(relations, f)
    with open(os.path.join(raw_folder, "%s_static_patches_labels.pkl" % well), 'wb') as f:
        pickle.dump(labels, f)
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
                        # channels: list,
                        # model_path: str,
                        sites: list,
                        config_: YamlReader,
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

def process_VAE(raw_folder: str,
                supp_folder: str,
                sites: list,
                config_: YamlReader,
                gpu: int=0,
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
        raw_folder (str): folder for raw data, segmentation and
            summarized results
        supp_folder (str): folder for supplementary data
        sites (list): list of FOVs to process
        config_ (YamlReader): Reads fields from the "INFERENCE" category

    """
    #TODO: add pooling datasets features and remove hardcoded normalization constants
    # ideally normalization parameters should be determined from pooled training data,
    # For inference same normalization parameters can be used or determined from the inference data,
    # depending on if the inference data has the same distribution as training data

    model_path = config_.inference.weights
    # weights_dir = config_.files.weights_dir
    channels = config_.inference.channels
    num_hiddens = config_.training.num_hiddens
    num_residual_hiddens = config_.training.num_residual_hiddens
    num_embeddings = config_.training.num_embeddings
    commitment_cost = config_.training.commitment_cost
    network = config_.inference.model
    save_output = config_.inference.save_output
    batch_size = config_.inference.batch_size
    num_workers = config_.inference.num_workers

    assert len(channels) > 0, "At least one channel must be specified"

    # these sites should be from a single condition (C5, C4, B-wells, etc..)
    model_dir = os.path.dirname(model_path)
    #TODO: add model_name to the config. Set the default to be the same as model folder name
    model_name = os.path.basename(model_dir)
    # output_dir = os.path.join(summary_folder, model_name + '_pool_norm')
    output_dir = os.path.join(raw_folder, model_name)
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

    ###
    # channel_mean = [32778.97446252,   681.61666079]
    # channel_std = [1314.90374187,  688.80291129]

    ### estimate mean and std from the data ###
    channel_mean = config_.inference.channel_mean
    channel_std = config_.inference.channel_std
    # channel_mean = None
    # channel_std = None

    print(f"\tloading static patches {os.path.join(raw_folder, '%s_static_patches.pkl' % well)}")
    dataset = pickle.load(open(os.path.join(raw_folder, '%s_static_patches.pkl' % well), 'rb'))
    dataset = dataset[:, channels, ...]
    # dataset = zscore(np.squeeze(dataset), channel_mean=channel_mean, channel_std=channel_std)
    dataset = zscore_patch(np.squeeze(dataset))
    dataset = TensorDataset(torch.from_numpy(dataset).float())
    assert len(dataset.tensors[0].shape) == 4, "dataset tensor dimension can only be 4, not {}".format(len(dataset.tensors[0].shape))
    _, n_channels, x_size, y_size = dataset.tensors[0].shape
    device = torch.device('cuda:%d' % gpu)
    print('Encoding images using gpu {}...'.format(gpu))
    if 'VAE' in network:
        network_cls = getattr(vae, network)
        model = network_cls(num_inputs=2,
                            num_hiddens=num_hiddens,
                            num_residual_hiddens=num_residual_hiddens,
                            num_residual_layers=2,
                            num_embeddings=num_embeddings,
                            gpu=True)

        model = model.to(device)
        try:
            if not model_path is None:
                model.load_state_dict(torch.load(model_path))
            else:
                model.load_state_dict(torch.load('HiddenStateExtractor/save_0005_bkp4.pt'))
        except Exception as ex:
            print(ex)
            raise ValueError("Error in loading model weights for VQ-VAE")

        z_bs = []
        z_as = []
        for i in range(len(dataset)):
            sample = dataset[i:(i + 1)][0]
            sample = sample.reshape([-1, n_channels, x_size, y_size]).to(device)
            z_b = model.enc(sample)
            z_a, _, _ = model.vq(z_b)
            z_bs[i] = z_b.cpu().data.numpy()
            z_as[i] = z_a.cpu().data.numpy()

        dats = np.stack(z_bs, 0).reshape((len(dataset), -1))
        print(f"\tsaving {os.path.join(output_dir, '%s_latent_space.pkl' % well)}")
        with open(os.path.join(output_dir, '%s_latent_space.pkl' % well), 'wb') as f:
            pickle.dump(dats, f, protocol=4)

        dats = np.stack(z_as, 0).reshape((len(dataset), -1))
        print(f"\tsaving {os.path.join(output_dir, '%s_latent_space_after.pkl' % well)}")
        with open(os.path.join(output_dir, '%s_latent_space_after.pkl' % well), 'wb') as f:
            pickle.dump(dats, f, protocol=4)

        if save_output:
            np.random.seed(0)
            random_inds = np.random.randint(0, len(dataset), (20,))
            for i in random_inds:
                sample = dataset[i:(i + 1)][0].to(device)
                sample = sample.reshape([-1, n_channels, x_size, y_size]).to(device)
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
    elif 'ResNet' in network:
        network_cls = getattr(resnet, 'EncodeProject')
        model = network_cls(arch=network, num_inputs=len(channels))
        model = model.to(device)
        # print(model)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        data_loader = DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=False,
                                  )
        h_s = []
        with tqdm(data_loader, desc='inference batch') as batch_pbar:
            for b_idx, batch in enumerate(batch_pbar):
                data, = batch
                data = data.to(device)
                code = model.encode(data, out='z').cpu().data.numpy().squeeze()
                h_s.append(code)
        dats = np.concatenate(h_s, axis=0)
        print(f"\tsaving {os.path.join(output_dir, '%s_latent_space.pkl' % well)}")
        with open(os.path.join(output_dir, '%s_latent_space.pkl' % well), 'wb') as f:
            pickle.dump(dats, f, protocol=4)
    else:
        raise ValueError('Network {} is not available'.format(network))




