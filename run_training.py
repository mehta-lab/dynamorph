import os
import h5py
import cv2
import numpy as np
import queue
gpu = True
gpuid = 0
if gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpuid)
    print("CUDA_VISIBLE_DEVICES", os.environ["CUDA_VISIBLE_DEVICES"])
import torch as t
import torch.nn as nn
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
# from torchvision import transforms
from scipy.sparse import csr_matrix

from HiddenStateExtractor.vae import CHANNEL_RANGE, CHANNEL_MAX, VQ_VAE_z32
from SingleCellPatch.extract_patches import im_adjust
from pipeline.train_utils import EarlyStopping, TripletDataset
from HiddenStateExtractor.losses import AllTripletMiner, HardNegativeTripletMiner
from HiddenStateExtractor.resnet import EncodeProject


def get_relation_tensor(relation_mat, sample_ids, gpu=True):
    """
    Slice relation matrix according to sample_ids; convert to torch tensor
    Args:
        relation_mat (scipy sparse array): symmetric matrix describing the relation between samples
        sample_ids (list): row & column ids to select
        gpu (bool): send the tensor to gpu if True

    Returns:
        batch_relation_mat (torch tensor or None): sliced relation matrix

    """
    if relation_mat is None:
        return None
    batch_relation_mat = relation_mat[sample_ids, :]
    batch_relation_mat = batch_relation_mat[:, sample_ids]
    batch_relation_mat = batch_relation_mat.todense()
    batch_relation_mat = t.from_numpy(batch_relation_mat).float()
    if gpu:
        batch_relation_mat = batch_relation_mat.cuda()
    return batch_relation_mat

def get_mask(mask, sample_ids, gpu=True):
    """
    Slice cell masks according to sample_ids; convert to torch tensor
    Args:
        mask (numpy array): cell masks for dataset
        sample_ids (list): mask ids to select
        gpu (bool): send the tensor to gpu if True

    Returns:
        batch_mask (torch tensor or None): sliced relation matrix
    """
    if mask is None:
        return None
    batch_mask = mask[sample_ids][0][:, 1:2, :, :]  # Hardcoded second slice (large mask)
    batch_mask = (batch_mask + 1.) / 2.
    if gpu:
        batch_mask = batch_mask.cuda()
    return batch_mask

def run_one_batch(model, batch, train_loss, labels=None, optimizer=None, batch_relation_mat=None,
                    batch_mask=None, gpu=True, transform=None, training=True):
    """ Train on a single batch of data
    Args:
        model (nn.Module): pytorch model object
        batch (TensorDataset): batch of training or validation inputs
        train_loss (dict): batch-wise training or validation loss
        optimizer: pytorch optimizer
        batch_relation_mat (np array or None): matrix of pairwise relations
        batch_mask (TensorDataset or None): if given, dataset of training
            sample weight masks
        gpu (bool, optional): Ture if the model is run on gpu
        transform (bool): data augmentation if true
        training (bool): Set True for training and False for validation (no weights update)

    Returns:
        model (nn.Module): updated model object
        train_loss (dict): updated batch-wise training or validation loss

    """
    if transform is not None:
        for idx_in_batch in range(len(batch)):
            img = batch[idx_in_batch]
            flip_idx = np.random.choice([0, 1, 2])
            if flip_idx != 0:
                img = t.flip(img, dims=(flip_idx,))
            rot_idx = int(np.random.choice([0, 1, 2, 3]))
            batch[idx_in_batch] = t.rot90(img, k=rot_idx, dims=[1, 2])
    if gpu:
        batch = batch.cuda()
        labels = labels.cuda()
    _, train_loss_dict = model(batch, labels=labels, time_matching_mat=batch_relation_mat, batch_mask=batch_mask)
    if training:
        train_loss_dict['total_loss'].backward()
        optimizer.step()
        model.zero_grad()
    for key, loss in train_loss_dict.items():
        if key not in train_loss:
            train_loss[key] = []
        # if isinstance(loss, t.Tensor):
        loss = float(loss) # float here magically removes the history attached to tensors
        train_loss[key].append(loss)
    # print(train_loss_dict)
    del batch, train_loss_dict, labels
    return model, train_loss

def train_val_split(dataset, labels, val_split_ratio=0.15, seed=0):
    assert val_split_ratio is None or 0 < val_split_ratio < 1
    n_samples = len(dataset)
    # Declare sample indices and do an initial shuffle
    sample_ids = list(range(n_samples))
    np.random.seed(seed)
    np.random.shuffle(sample_ids)
    split = int(np.floor(val_split_ratio * n_samples))
    # randomly choose the split start
    np.random.seed(seed)
    split_start = np.random.randint(0, n_samples - split)
    val_ids = sample_ids[split_start: split_start + split]
    train_ids = sample_ids[:split_start] + sample_ids[split_start + split:]
    train_set = dataset[train_ids]
    train_labels = labels[train_ids]
    val_set = dataset[val_ids]
    val_labels = labels[val_ids]
    return train_set, train_labels, val_set, val_labels

def train(model, train_loader, val_loader, output_dir, relation_mat=None, mask=None,
          n_epochs=10, lr=0.001, gpu=True,
          transform=None,  patience=20, earlystop_metric='total_loss',
          retrain=False, log_step_offset=0):
    """ Train function for VQ-VAE, VAE, IWAE, etc.

    Args:
        model (nn.Module): autoencoder model
        dataset (TensorDataset): dataset of training inputs
        relation_mat (scipy csr matrix or None, optional): if given, sparse 
            matrix of pairwise relations
        mask (TensorDataset or None, optional): if given, dataset of training 
            sample weight masks
        n_epochs (int, optional): number of epochs
        lr (float, optional): learning rate
        batch_size (int, optional): batch size
        gpu (bool, optional): Ture if the model is run on gpu
        shuffle_data (bool): shuffle data at the end of the epoch to add randomness to mini-batch.
            Set False when using matching loss
        transform (bool): data augmentation if true
        val_split_ratio (float or None): fraction of the dataset used for validation
        patience (int or None): Number of epochs to wait before stopping training if validation loss does not improve.
        retrain (bool): Retrain the model from scratch if True. Load existing model and continue training otherwise

    Returns:
        nn.Module: trained model

    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'model.pt')
    if os.path.exists(model_path) and not retrain:
        print('Found previously saved model state {}. Continue training...'.format(model_path))
        model.load_state_dict(t.load(model_path))

    # early stopping requires validation set
    if patience is not None:
        assert val_loader is not None
    optimizer = t.optim.Adam(model.parameters(), lr=lr, betas=(.9, .999))
    model.zero_grad()
    writer = SummaryWriter(output_dir)
    model_path = os.path.join(output_dir, 'model.pt')
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=model_path)
    for epoch in tqdm(range(log_step_offset, n_epochs), desc='Epoch'):
        train_loss = {}
        val_loss = {}
        # loop through training batches
        model.train()
        with tqdm(train_loader, desc='train batch') as batch_pbar:
            for b_idx, batch in enumerate(batch_pbar):
                labels, data = batch
                labels = t.cat([label for label in labels], axis=0)
                data = t.cat([datum for datum in data], axis=0)
                # batch = dataset[train_ids_batch][0]
                # TODO: move relation matrix to dataset or generate on the fly using labels in contrastive loss class
                # Relation (adjacent frame, same trajectory)
                # batch_relation_mat = get_relation_tensor(relation_mat, train_ids_batch, gpu=gpu)
                # Reconstruction mask
                # batch_mask = get_mask(mask, train_ids_batch, gpu)
                batch_relation_mat = None
                batch_mask = None
                model, train_loss = \
                    run_one_batch(model, data, train_loss, labels=labels, optimizer=optimizer,
                                    batch_relation_mat=batch_relation_mat,
                                    batch_mask=batch_mask, gpu=gpu, transform=transform, training=True)
        # loop through validation batches
        model.eval()
        with t.no_grad():
            with tqdm(val_loader, desc='val batch') as batch_pbar:
                for b_idx, batch in enumerate(batch_pbar):
                    labels, data = batch
                    labels = t.cat([label for label in labels], axis=0)
                    data = t.cat([datum for datum in data], axis=0)
                    # # Relation (adjacent frame, same trajectory)
                    # batch_relation_mat = get_relation_tensor(relation_mat, val_ids_batch, gpu=gpu)
                    # # Reconstruction mask
                    # batch_mask = get_mask(mask, val_ids_batch, gpu)
                    batch_relation_mat = None
                    batch_mask = None
                    model, val_loss = \
                        run_one_batch(model, data, val_loss, labels=labels, optimizer=optimizer,
                                      batch_relation_mat=batch_relation_mat,
                                      batch_mask=batch_mask, gpu=gpu, transform=transform, training=False)
        for key, loss in train_loss.items():
            train_loss[key] = sum(loss) / len(loss)
            writer.add_scalar('Loss/' + key, train_loss[key], epoch)
        for key, loss in val_loss.items():
            val_loss[key] = sum(loss) / len(loss)
            writer.add_scalar('Val loss/' + key, val_loss[key], epoch)
        writer.flush()
        print('epoch %d' % epoch)
        print('train: ', ''.join(['{}:{:0.4f}  '.format(key, loss) for key, loss in train_loss.items()]))
        print('val:   ', ''.join(['{}:{:0.4f}  '.format(key, loss) for key, loss in val_loss.items()]))
        early_stopping(val_loss[earlystop_metric], model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    writer.close()
    return model


def train_adversarial(model, 
                      dataset, 
                      relation_mat=None, 
                      mask=None, 
                      n_epochs=10, 
                      lr_recon=0.001, 
                      lr_dis=0.001, 
                      lr_gen=0.001, 
                      batch_size=16, 
                      gpu=True):
    """ Train function for AAE.

    Args:
        model (nn.Module): autoencoder model (AAE)
        dataset (TensorDataset): dataset of training inputs
        relation_mat (scipy csr matrix or None, optional): if given, sparse 
            matrix of pairwise relations
        mask (TensorDataset or None, optional): if given, dataset of training 
            sample weight masks
        n_epochs (int, optional): number of epochs
        lr_recon (float, optional): learning rate for reconstruction (encoder + 
            decoder)
        lr_dis (float, optional): learning rate for discriminator
        lr_gen (float, optional): learning rate for generator
        batch_size (int, optional): batch size
        gpu (bool, optional): if the model is run on gpu
    
    Returns:
        nn.Module: trained model

    """
    optim_enc = t.optim.Adam(model.enc.parameters(), lr_recon)
    optim_dec = t.optim.Adam(model.dec.parameters(), lr_recon)
    optim_enc_g = t.optim.Adam(model.enc.parameters(), lr_gen)
    optim_enc_d = t.optim.Adam(model.enc_d.parameters(), lr_dis)
    model.zero_grad()

    n_batches = int(np.ceil(len(dataset)/batch_size))
    for epoch in range(n_epochs):
        recon_loss = []
        scores = []
        print('start epoch %d' % epoch) 
        for i in range(n_batches):
            # Input data
            batch = dataset[i*batch_size:(i+1)*batch_size][0]
            if gpu:
                batch = batch.cuda()
              
            # Relation (trajectory, adjacent)
            if not relation_mat is None:
                batch_relation_mat = relation_mat[i*batch_size:(i+1)*batch_size, i*batch_size:(i+1)*batch_size]
                batch_relation_mat = batch_relation_mat.todense()
                batch_relation_mat = t.from_numpy(batch_relation_mat).float()
                if gpu:
                    batch_relation_mat = batch_relation_mat.cuda()
            else:
                batch_relation_mat = None
            
            # Reconstruction mask
            if not mask is None:
                batch_mask = mask[i*batch_size:(i+1)*batch_size][0][:, 1:2, :, :] # Hardcoded second slice (large mask)
                batch_mask = (batch_mask + 1.)/2.
                if gpu:
                    batch_mask = batch_mask.cuda()
            else:
                batch_mask = None
              
            _, loss_dict = model(batch, time_matching_mat=batch_relation_mat, batch_mask=batch_mask)
            loss_dict['total_loss'].backward()
            optim_enc.step()
            optim_dec.step()
            loss_dict2 = model.adversarial_loss(batch)
            loss_dict2['descriminator_loss'].backward()
            optim_enc_d.step()
            loss_dict2['generator_loss'].backward()
            optim_enc_g.step()
            model.zero_grad()

            recon_loss.append(loss_dict['recon_loss'])
            scores.append(loss_dict2['score'])
        print('epoch %d recon loss: %f pred score: %f' % (epoch, sum(recon_loss).item()/len(recon_loss), sum(scores).item()/len(scores)))
    return model


def prepare_dataset(fs, cs=[0, 1], input_shape=(128, 128), channel_max=CHANNEL_MAX):
    """ Prepare input dataset for VAE

    This function reads individual h5 files

    Args:
        fs (list of str): list of file paths/single cell patch identifiers,
            images are saved as individual h5 files
        cs (list of int, optional): channels in the input
        input_shape (tuple, optional): input shape (height and width only)
        channel_max (np.array, optional): max intensities for channels

    Returns:
        TensorDataset: dataset of training inputs

    """
    tensors = []
    for i, f_n in enumerate(fs):
      if i%1000 == 0:
        print("Processed %d" % i)
      with h5py.File(f_n, 'r') as f:
        dat = f['masked_mat']
        if cs is None:
          cs = np.arange(dat.shape[2])
        stacks = []
        for c, m in zip(cs, channel_max):
          c_slice = cv2.resize(np.array(dat[:, :, c]).astype(float), input_shape)
          stacks.append(c_slice/m)
        tensors.append(t.from_numpy(np.stack(stacks, 0)).float())
    dataset = TensorDataset(t.stack(tensors, 0))
    return dataset


def prepare_dataset_from_collection(fs,
                                    cs=[0, 1],
                                    input_shape=(128, 128),
                                    channel_max=CHANNEL_MAX,
                                    file_path='./',
                                    file_suffix='_all_patches.pkl'):
    """ Prepare input dataset for VAE, deprecated

    This function reads assembled pickle files (dict)

    Args:
        fs (list of str): list of pickle file names
        cs (list of int, optional): channels in the input
        input_shape (tuple, optional): input shape (height and width only)
        channel_max (np.array, optional): max intensities for channels
        file_path (str, optional): root folder for saved pickle files
        file_suffix (str, optional): suffix of saved pickle files

    Returns:
        TensorDataset: dataset of training inputs

    """

    tensors = {}
    files = set([f.split('/')[-2] for f in fs])
    for file_name in files:
        file_dat = pickle.load(open(os.path.join(file_path, '%s%s' % (file_name, file_suffix)), 'rb')) #HARDCODED
        fs_ = [f for f in fs if f.split('/')[-2] == file_name ]
        for i, f_n in enumerate(fs_):
            dat = file_dat[f_n]['masked_mat']
            if cs is None:
                cs = np.arange(dat.shape[2])
            stacks = []
            for c, m in zip(cs, channel_max):
                c_slice = cv2.resize(np.array(dat[:, :, c]).astype(float), input_shape)
                stacks.append(c_slice/m)
            tensors[f_n] = t.from_numpy(np.stack(stacks, 0)).float()
    dataset = TensorDataset(t.stack([tensors[f_n] for f_n in fs], 0))
    return dataset

def reorder_with_trajectories(dataset, relations, seed=None):
    """ Reorder `dataset` to facilitate training with matching loss

    Args:
        dataset (TensorDataset): dataset of training inputs
        relations (dict): dict of pairwise relationship (adjacent frames, same 
            trajectory)
        seed (int or None, optional): if given, random seed
        w_a (float): weight for adjacent frames
        w_t (float): weight for non-adjecent frames in the same trajectory
    Returns:
        TensorDataset: dataset of training inputs (after reordering)
        scipy csr matrix: sparse matrix of pairwise relations
        list of int: index of samples used for reordering

    """
    if not seed is None:
        np.random.seed(seed)
    inds_pool = set(range(len(dataset)))
    inds_in_order = []
    relation_dict = {}
    for pair in relations:
        if relations[pair] == 2: # Adjacent pairs
            if pair[0] not in relation_dict:
                relation_dict[pair[0]] = []
            relation_dict[pair[0]].append(pair[1])
    while len(inds_pool) > 0:
        rand_ind = np.random.choice(list(inds_pool))
        if not rand_ind in relation_dict:
            inds_in_order.append(rand_ind)
            inds_pool.remove(rand_ind)
        else:
            traj = [rand_ind]
            q = queue.Queue()
            q.put(rand_ind)
            while True:
                try:
                    elem = q.get_nowait()
                except queue.Empty:
                    break
                new_elems = relation_dict[elem]
                for e in new_elems:
                    if not e in traj:
                        traj.append(e)
                        q.put(e)
            inds_in_order.extend(traj)
            for e in traj:
                inds_pool.remove(e)
    new_tensor = dataset.tensors[0][np.array(inds_in_order)]
    
    values = []
    new_relations = []
    for k, v in relations.items():
        # 2 - adjacent, 1 - same trajectory
        if v == 1:
            values.append(1)
        elif v == 2:
            values.append(2)
        new_relations.append(k)
    new_relations = np.array(new_relations)
    relation_mat = csr_matrix((np.array(values), (new_relations[:, 0], new_relations[:, 1])),
                              shape=(len(dataset), len(dataset)))
    relation_mat = relation_mat[np.array(inds_in_order)][:, np.array(inds_in_order)]
    return TensorDataset(new_tensor), relation_mat, inds_in_order

def zscore(input_image, channel_mean=None, channel_std=None):
    """
    Performs z-score normalization. Adds epsilon in denominator for robustness

    :param input_image: input image for intensity normalization
    :return: z score normalized image
    """
    if not channel_mean:
        channel_mean = np.mean(input_image, axis=(0, 2, 3))
    if not channel_std:
        channel_std = np.std(input_image, axis=(0, 2, 3))
    channel_slices = []
    for c in range(len(channel_mean)):
        mean = channel_mean[c]
        std = channel_std[c]
        channel_slice = (input_image[:, c, ...] - mean) / \
                        (std + np.finfo(float).eps)
        # channel_slice = t.clamp(channel_slice, -1, 1)
        channel_slices.append(channel_slice)
    norm_img = np.stack(channel_slices, 1)
    # norm_img = (input_image - mean.astype(np.float64)) /\
    #            (std + np.finfo(float).eps)
    return norm_img


def unzscore(im_norm, mean, std):
    """
    Revert z-score normalization applied during preprocessing. Necessary
    before computing SSIM

    :param input_image: input image for un-zscore
    :return: image at its original scale
    """

    im = im_norm * (std + np.finfo(float).eps) + mean

    return im

def rescale(dataset):
    """ Rescale value range of image patches in `dataset` to CHANNEL_RANGE

    Args:
        dataset (TensorDataset): dataset before rescaling

    Returns:
        TensorDataset: dataset after rescaling

    """
    tensor = dataset.tensors[0]
    channel_mean = t.mean(tensor, dim=[0, 2, 3])
    channel_std = t.mean(tensor, dim=[0, 2, 3])
    print('channel_mean:', channel_mean)
    print('channel_std:', channel_std)
    assert len(channel_mean) == tensor.shape[1]
    channel_slices = []
    for i in range(len(CHANNEL_RANGE)):
        mean = channel_mean[i]
        std = channel_std[i]
        channel_slice = (tensor[:, i] - mean) / std
        # channel_slice = t.clamp(channel_slice, -1, 1)
        channel_slices.append(channel_slice)
    new_tensor = t.stack(channel_slices, 1)
    return TensorDataset(new_tensor)


def resscale_backward(tensor):
    """ Reverse operation of `rescale`

    Args:
        dataset (TensorDataset): dataset after rescaling

    Returns:
        TensorDataset: dataset before rescaling

    """
    assert len(tensor.shape) == 4
    assert len(CHANNEL_RANGE) == tensor.shape[1]
    channel_slices = []
    for i in range(len(CHANNEL_RANGE)):
        lower_, upper_ = CHANNEL_RANGE[i]
        channel_slice = lower_ + tensor[:, i] * (upper_ - lower_)
        channel_slices.append(channel_slice)
    new_tensor = t.stack(channel_slices, 1)
    return new_tensor

def save_recon_images(val_dataloader, model, model_dir):
    # %% display recon images
    os.makedirs(model_dir, exist_ok=True)
    batch = next(iter(val_dataloader))
    labels, data = batch
    labels = t.cat([label for label in labels], axis=0)
    data = t.cat([datum for datum in data], axis=0)
    output = model(data.cuda(), labels.cuda())[0]
    for i in range(10):
        im_phase = im_adjust(data[i, 0].data.numpy())
        im_phase_recon = im_adjust(output[i, 0].cpu().data.numpy())
        im_retard = im_adjust(data[i, 1].data.numpy())
        im_retard_recon = im_adjust(output[i, 1].cpu().data.numpy())
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
        fig.savefig(os.path.join(model_dir, 'recon_%d.jpg' % i),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)

def concat_relations(relations, labels, offsets):
    """combine relation dictionaries from multiple datasets

    Args:
        relations (list): list of relation dict to combine
        labels (list): list of label array to combine
        offsets (list): offset to add to the indices

    Returns: new_relations (dict): dictionary of combined relations

    """
    new_relations = {}
    new_labels = []
    for relation, label, offset in zip(relations, labels, offsets):
        old_keys = relation.keys()
        new_keys = [(id1 + offset, id2 + offset) for id1, id2 in old_keys]
        new_label = label + offset
        # make a new dict with updated keys
        relation = dict(zip(new_keys, relation.values()))
        new_relations.update(relation)
        new_labels.append(new_label)
    new_labels = np.concatenate(new_labels, axis=0)
    return new_relations, new_labels


def augment_img(img):
    flip_idx = np.random.choice([0, 1, 2])
    if flip_idx != 0:
        img = np.flip(img, axis=flip_idx)
    rot_idx = int(np.random.choice([0, 1, 2, 3]))
    img = np.rot90(img, k=rot_idx, axes=(1, 2))
    return img


if __name__ == '__main__':
    ### Settings ###
    cs = [0, 1]
    cs_mask = [2, 3]
    input_shape = (128, 128)

    w_a = 1
    w_t = 0.5
    w_n = -0.5
    margin = 1
    val_split_ratio = 0.15
    patience = np.inf
    n_pos_samples = 8
    batch_size = 768
    # adjusted batch size for dataloaders
    batch_size_adj = int(np.floor(batch_size/n_pos_samples))
    #### cardiomyocyte data###
    # channel_mean = [0.49998672, 0.007081]
    # channel_std = [0.00074311, 0.00906428]

    ### microglia data####
    # channel_mean = [0.4, 0, 0.5]
    # channel_std = [0.05, 0.05, 0.05]

    ### estimate mean and std from the data ###
    channel_mean = None
    channel_std = None

    supp_dirs = ['/CompMicro/projects/cardiomyocytes/200721_CM_Mock_SPS_Fluor/20200721_CM_Mock_SPS/dnm_supp_tstack',
                 '/CompMicro/projects/cardiomyocytes/20200722CM_LowMOI_SPS_Fluor/20200722 CM_LowMOI_SPS/dnm_supp_tstack',
                 '/CompMicro/projects/virtualstaining/kidneyslice/2019_02_15_kidney_slice/dnm_supp']
    train_dirs = ['/CompMicro/projects/cardiomyocytes/200721_CM_Mock_SPS_Fluor/20200721_CM_Mock_SPS/dnm_train_tstack',
                  '/CompMicro/projects/cardiomyocytes/20200722CM_LowMOI_SPS_Fluor/20200722 CM_LowMOI_SPS/dnm_train_tstack',
                  '/CompMicro/projects/virtualstaining/kidneyslice/2019_02_15_kidney_slice/dnm_train']
    raw_dirs = ['/CompMicro/projects/cardiomyocytes/200721_CM_Mock_SPS_Fluor/20200721_CM_Mock_SPS/dnm_input_tstack',
                '/CompMicro/projects/cardiomyocytes/20200722CM_LowMOI_SPS_Fluor/20200722 CM_LowMOI_SPS/dnm_input_tstack',
                '/CompMicro/projects/virtualstaining/kidneyslice/2019_02_15_kidney_slice/dnm_input']
    dir_sets = list(zip(supp_dirs, train_dirs, raw_dirs))
    # dir_sets = dir_sets[0:1]
    ts_keys = []
    datasets = []
    relations = []
    labels = []
    id_offsets = [0]
    ### Load Data ###
    for supp_dir, train_dir, raw_dir in dir_sets:
        os.makedirs(train_dir, exist_ok=True)
        print(f"\tloading file paths {os.path.join(raw_dir, 'im_file_paths.pkl')}")
        ts_key = pickle.load(open(os.path.join(raw_dir, 'im_file_paths.pkl'), 'rb'))
        print(f"\tloading static patches {os.path.join(raw_dir, 'im_static_patches.pkl')}")
        dataset = pickle.load(open(os.path.join(raw_dir, 'im_static_patches.pkl'), 'rb'))
        print('dataset.shape:', dataset.shape)
        dataset = pickle.load(open(os.path.join(raw_dir, 'im_static_patches.pkl'), 'rb'))
        label = pickle.load(open(os.path.join(raw_dir, "im_static_patches_labels.pkl"), 'rb'))
        # Note that `relations` is depending on the order of fs (should not sort)
        # `relations` is generated by script "generate_trajectory_relations.py"
        relation = pickle.load(open(os.path.join(raw_dir, 'im_static_patches_relations.pkl'), 'rb'))
        # dataset_mask = TensorDataset(dataset_mask.tensors[0][np.array(inds_in_order)])
        # print('relations:', relations)
        print('len(ts_key):', len(ts_key))
        print('len(dataset):', len(dataset))
        relations.append(relation)
        ts_keys += ts_key
        # TODO: handle non-singular z-dimension case earlier in the pipeline
        datasets.append(np.squeeze(dataset))
        labels.append(label)
        id_offsets.append(len(dataset))
    id_offsets = id_offsets[:-1]
    dataset = np.concatenate(datasets, axis=0)
    dataset = zscore(dataset, channel_mean=channel_mean, channel_std=channel_std).astype(np.float32)
    # dataset = TensorDataset(t.from_numpy(dataset).float())
    relations, labels = concat_relations(relations, labels, offsets=id_offsets)
    # dataset, relation_mat, inds_in_order = reorder_with_trajectories(dataset, relations, seed=123)
    # labels = labels[inds_in_order]
    train_set, train_labels, val_set, val_labels = \
        train_val_split(dataset, labels, val_split_ratio=val_split_ratio, seed=0)
    tri_train_set = TripletDataset(train_labels, lambda index: augment_img(train_set[index]), n_pos_samples)
    tri_val_set = TripletDataset(val_labels, lambda index: augment_img(val_set[index]), n_pos_samples)
    # Data Loader
    train_loader = DataLoader(tri_train_set,
                                batch_size=batch_size_adj,
                                shuffle=True,
                                num_workers=2,
                                pin_memory=False,
                                )
    val_loader = DataLoader(tri_val_set,
                              batch_size=batch_size_adj,
                              shuffle=False,
                              num_workers=2,
                              pin_memory=False,
                              )
    tri_loss = AllTripletMiner(margin=margin).cuda()
    # tri_loss = HardNegativeTripletMiner(margin=margin).cuda()
    ## Initialize Model ###
    num_hiddens = 64
    num_residual_hiddens = num_hiddens
    num_embeddings = 512
    commitment_cost = 0.25
    alpha = 100
    model_arch = 'ResNet50'
    # model = VQ_VAE_z32(num_inputs=2,
    #                    num_hiddens=num_hiddens,
    #                    num_residual_hiddens=num_residual_hiddens,
    #                    num_residual_layers=2,
    #                    num_embeddings=num_embeddings,
    #                    commitment_cost=commitment_cost,
    #                    alpha=alpha,
    #                    w_a=w_a,
    #                    w_t=w_t,
    #                    w_n=w_n,
    #                    margin=margin,
    #                    extra_loss={'Triple loss': tri_loss})
    model = EncodeProject(arch=model_arch, loss=tri_loss)
    #TODO: Torchvision data augmentation does not work for Pytorch tensordataset. Rewrite with dataloader
    #
    # transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomVerticalFlip(),
    #     transforms.RandomRotation(180, resample=PIL.Image.BILINEAR),
    #     transforms.ToTensor(),
    # ])
    # model_dir = os.path.join(train_dir, 'CM+kidney_z32_nh{}_nrh{}_ne{}_alpha{}_mrg{}_npos{}_aug_alltriloss'.format(
    #     num_hiddens, num_residual_hiddens, num_embeddings, alpha, margin, n_pos_samples))
    model_dir = os.path.join(train_dir, 'CM+kidney_{}_mrg{}_npos{}_bh{}_alltriloss_nostop'.format(
        model_arch, margin, n_pos_samples, batch_size))
    if gpu:

        model = model.cuda()
    # save_recon_images(val_loader, model, model_dir)
    model = train(model,
                  train_loader=train_loader,
                  val_loader=val_loader,
                  output_dir=model_dir,
                  relation_mat=None,
                  mask=None,
                  n_epochs=600,
                  lr=0.0001,
                  gpu=gpu,
                  patience=patience,
                  earlystop_metric='positive_triplet',
                  retrain=False,
                  log_step_offset=0)

    # ### Check coverage of embedding vectors ###
    # used_indices = []
    # for i in range(500):
    #     sample = dataset[i:(i+1)][0].cuda()
    #     z_before = model.enc(sample)
    #     indices = model.vq.encode_inputs(z_before)
    #     used_indices.append(np.unique(indices.cpu().data.numpy()))
    # print(np.unique(np.concatenate(used_indices)))
    #
    # ### Generate latent vectors ###
    # z_bs = {}
    # z_as = {}
    # for i in range(len(dataset)):
    #     sample = dataset[i:(i+1)][0].cuda()
    #     z_b = model.enc(sample)
    #     z_a, _, _ = model.vq(z_b)
    #
    #     f_n = ts_keys[i]
    #     z_as[f_n] = z_a.cpu().data.numpy()
    #     z_bs[f_n] = z_b.cpu().data.numpy()
