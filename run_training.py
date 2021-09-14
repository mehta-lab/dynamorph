import os
import h5py
import cv2
import numpy as np
import argparse
import torch as t
import torch.nn as nn
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
# from torchvision import transforms
from scipy.sparse import csr_matrix

from HiddenStateExtractor.vae import CHANNEL_MAX
from SingleCellPatch.extract_patches import im_adjust, cv2_fn_wrapper
from pipeline.train_utils import EarlyStopping, TripletDataset, zscore, zscore_patch
from HiddenStateExtractor.losses import AllTripletMiner
from HiddenStateExtractor.resnet import EncodeProject
import HiddenStateExtractor.vae as vae

from configs.config_reader import YamlReader
import queue
import distutils


# Dataset preparation functions
def prepare_dataset(fs,
                    cs=[0, 1],
                    input_shape=(128, 128)):
    """ Prepare input dataset for VAE

    This function reads individual h5 files (deprecated)

    Args:
        fs (list of str): list of file paths/single cell patch identifiers,
            images are saved as individual h5 files
        cs (list of int, optional): channels in the input
        input_shape (tuple, optional): input shape (height and width only)

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
                cs = np.arange(dat.shape[0])
            dat = np.array(dat)[np.array(cs)].astype(float)
            resized_dat = cv2_fn_wrapper(cv2.resize, dat, input_shape)
            tensors.append(resized_dat)
    dataset = np.stack(tensors, 0)
    return dataset


def prepare_dataset_from_collection(fs,
                                    cs=[0, 1],
                                    input_shape=(128, 128),
                                    file_path='./',
                                    file_suffix='_all_patches.pkl'):
    """ Prepare input dataset for VAE, deprecated

    This function reads assembled pickle files (deprecated)

    Args:
        fs (list of str): list of pickle file names
        cs (list of int, optional): channels in the input
        input_shape (tuple, optional): input shape (height and width only)
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
            dat = file_dat[f_n]['masked_mat'] # n_channels, n_z, x_size, y_size
            if cs is None:
                cs = np.arange(dat.shape[0])
            dat = np.array(dat)[np.array(cs)].astype(float)
            resized_dat = cv2_fn_wrapper(cv2.resize, dat, input_shape)
            tensors[f_n] = resized_dat
    dataset = np.stack([tensors[key] for key in fs], 0)
    return dataset

def reorder_with_trajectories(dataset, relations, seed=None):
    """ Reorder `dataset` to facilitate training with matching loss

    Args:
        dataset (TensorDataset): dataset of training inputs
        relations (dict): dict of pairwise relationship (adjacent frames, same 
            trajectory)
        seed (int or None, optional): if given, random seed

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


def vae_preprocess(dataset,
                   use_channels=[0, 1],
                   preprocess_setting={
                       0: ("normalize", 0.4, 0.05), # Phase
                       1: ("scale", 0.05), # Retardance
                       2: ("normalize", 0.5, 0.05), # Brightfield
                       },
                   clip=[0, 1]):
    """ Preprocess `dataset` to a suitable range

    Args:
        dataset (TensorDataset): dataset of training inputs
        use_channels (list, optional): list of channel indices used for model
            prediction
        preprocess_setting (dict, optional): settings for preprocessing,
            formatted as {channel index: (preprocessing mode,
                                          target mean,
                                          target std(optional))}

    Returns:
        TensorDataset: dataset of training inputs (after preprocessing)

    """

    tensor = dataset
    output = []
    for channel in use_channels:
        channel_slice = tensor[:, channel]
        channel_slice = channel_slice / CHANNEL_MAX # Scale to [0, 1]
        if preprocess_setting[channel][0] == "scale":
            target_mean = preprocess_setting[channel][1]
            slice_mean = channel_slice.mean()
            output_slice = channel_slice / slice_mean * target_mean
        elif preprocess_setting[channel][0] == "normalize":
            target_mean = preprocess_setting[channel][1]
            target_sd = preprocess_setting[channel][2]
            slice_mean = channel_slice.mean()
            slice_sd = channel_slice.std()
            z_channel_slice = (channel_slice - slice_mean) / slice_sd
            output_slice = z_channel_slice * target_sd + target_mean
        else:
            raise ValueError("Preprocessing mode not supported")
        if clip:
            output_slice = np.clip(output_slice, clip[0], clip[1])
        output.append(output_slice)
    output = np.stack(output, 1)
    return output

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
    output = model(data.to(device), labels.to(device))[0]
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
    """Data augmentation with flipping and rotation"""
    # TODO: Rewrite with torchvision transform
    flip_idx = np.random.choice([0, 1, 2])
    if flip_idx != 0:
        img = np.flip(img, axis=flip_idx)
    rot_idx = int(np.random.choice([0, 1, 2, 3]))
    img = np.rot90(img, k=rot_idx, axes=(1, 2))
    return img


def get_relation_tensor(relation_mat, sample_ids, device='cuda:0'):
    """
    Slice relation matrix according to sample_ids; convert to torch tensor
    Args:
        relation_mat (scipy sparse array): symmetric matrix describing the relation between samples
        sample_ids (list): row & column ids to select
        device (str): device to run the model on

    Returns:
        batch_relation_mat (torch tensor or None): sliced relation matrix

    """
    if relation_mat is None:
        return None
    batch_relation_mat = relation_mat[sample_ids, :]
    batch_relation_mat = batch_relation_mat[:, sample_ids]
    batch_relation_mat = batch_relation_mat.todense()
    batch_relation_mat = t.from_numpy(batch_relation_mat).float()
    if device:
        batch_relation_mat = batch_relation_mat.to(device)
    return batch_relation_mat


def get_mask(mask, sample_ids, device='cuda:0'):
    """
    Slice cell masks according to sample_ids; convert to torch tensor
    Args:
        mask (numpy array): cell masks for dataset
        sample_ids (list): mask ids to select
        device (str): device to run the model on

    Returns:
        batch_mask (torch tensor or None): sliced relation matrix
    """
    if mask is None:
        return None
    batch_mask = mask[sample_ids][0][:, 1:2, :, :]  # Hardcoded second slice (large mask)
    batch_mask = (batch_mask + 1.) / 2.
    batch_mask = batch_mask.to(device)
    return batch_mask


def run_one_batch(model, batch, train_loss, model_kwargs = None, optimizer=None,
                transform=False, training=True):
    """ Train on a single batch of data
    Args:
        model (nn.Module): pytorch model object
        batch (TensorDataset): batch of training or validation inputs
        train_loss (dict): batch-wise training or validation loss
        optimizer: pytorch optimizer
        batch_relation_mat (np array or None): matrix of pairwise relations
        batch_mask (TensorDataset or None): if given, dataset of training
            sample weight masks
        transform (bool): data augmentation if true
        training (bool): Set True for training and False for validation (no weights update)

    Returns:
        model (nn.Module): updated model object
        train_loss (dict): updated batch-wise training or validation loss

    """
    if transform:
        for idx_in_batch in range(len(batch)):
            img = batch[idx_in_batch]
            flip_idx = np.random.choice([0, 1, 2])
            if flip_idx != 0:
                img = t.flip(img, dims=(flip_idx,))
            rot_idx = int(np.random.choice([0, 1, 2, 3]))
            batch[idx_in_batch] = t.rot90(img, k=rot_idx, dims=[1, 2])
    _, train_loss_dict = model(batch, **model_kwargs)
    if training:
        train_loss_dict['total_loss'].backward()
        optimizer.step()
        model.zero_grad()
    for key, loss in train_loss_dict.items():
        if key not in train_loss:
            train_loss[key] = []
        # if isinstance(loss, t.Tensor):
        loss = float(loss)  # float here magically removes the history attached to tensors
        train_loss[key].append(loss)
    # print(train_loss_dict)
    del batch, train_loss_dict
    return model, train_loss


def train_val_split(dataset, labels, val_split_ratio=0.15, seed=0):
    """Split the dataset into train and validation sets

    Args:
        dataset (TensorDataset): dataset of training inputs
        labels (list or np array): labels corresponding to inputs
        val_split_ratio (float or None): fraction of the dataset used for validation
        seed (int): seed controlling random split of the dataset

    Returns:
        train_set (TensorDataset): train set
        train_labels (list or np array): train labels corresponding to inputs in train set
        val_set (TensorDataset): validation set
        val_labels (list or np array): validation labels corresponding to inputs in train set

    """
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


def train(model, dataset, output_dir, relation_mat=None, mask=None,
          n_epochs=10, lr=0.001, batch_size=16, device='cuda:0', shuffle_data=False,
          transform=False, val_split_ratio=0.15, patience=20):
    """ Legacy train function for VAE models.

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
        device (str): device to run the model on
        shuffle_data (bool): shuffle data at the end of the epoch to add randomness to mini-batch.
            Set False when using matching loss
        transform (bool): data augmentation if true
        val_split_ratio (float or None): fraction of the dataset used for validation
        patience (int or None): Number of epochs to wait before stopping training if validation loss does not improve.

    Returns:
        nn.Module: trained model

    """
    assert val_split_ratio is None or 0 < val_split_ratio < 1
    # early stopping requires validation set
    if patience is not None:
        assert val_split_ratio is not None
    optimizer = t.optim.Adam(model.parameters(), lr=lr, betas=(.9, .999))
    model.zero_grad()
    n_samples = len(dataset)
    # Declare sample indices and do an initial shuffle
    sample_ids = list(range(n_samples))
    split = int(np.floor(val_split_ratio * n_samples))
    # randomly choose the split start
    split_start = np.random.randint(0, n_samples - split)
    if shuffle_data:
        np.random.shuffle(sample_ids)
    val_ids = sample_ids[split_start: split_start + split]
    train_ids = sample_ids[:split_start] + sample_ids[split_start + split:]
    n_train = len(train_ids)
    n_val = len(val_ids)
    n_batches = int(np.ceil(n_train / batch_size))
    n_val_batches = int(np.ceil(n_val / batch_size))
    writer = SummaryWriter(output_dir)
    print(f"========== TENSORBOARD LOGS WRITTEN TO ============= ")
    print(f"{output_dir}")
    model_path = os.path.join(output_dir, 'model.pt')
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=model_path)
    for epoch in range(n_epochs):
        train_loss = {}
        val_loss = {}
        print('start epoch %d' % epoch)
        # loop through training batches
        for i in range(n_batches):
            # deal with last batch might < batch size
            train_ids_batch = train_ids[i * batch_size:min((i + 1) * batch_size, n_train)]
            batch = dataset[train_ids_batch][0].to(device)
            # Relation (adjacent frame, same trajectory)
            batch_relation_mat = get_relation_tensor(relation_mat, train_ids_batch, device=device)
            # Reconstruction mask
            batch_mask = get_mask(mask, train_ids_batch, device=device)
            model, train_loss = \
                run_one_batch(model, batch, train_loss, optimizer=optimizer,
                              model_kwargs={'time_matching_mat': batch_relation_mat,
                              'batch_mask': batch_mask}, transform=transform, training=True)
        # loop through validation batches
        for i in range(n_val_batches):
            val_ids_batch = val_ids[i * batch_size:min((i + 1) * batch_size, n_val)]
            batch = dataset[val_ids_batch][0].to(device)
            # Relation (adjacent frame, same trajectory)
            batch_relation_mat = get_relation_tensor(relation_mat, val_ids_batch, device=device)
            # Reconstruction mask
            batch_mask = get_mask(mask, val_ids_batch, device)
            model, val_loss = \
                run_one_batch(model, batch, val_loss, optimizer=optimizer,
                              model_kwargs={'time_matching_mat': batch_relation_mat,
                                            'batch_mask': batch_mask}, transform=transform, training=False)
        # shuffle train ids at the end of the epoch
        if shuffle_data:
            np.random.shuffle(train_ids)
        for key, loss in train_loss.items():
            train_loss[key] = sum(loss) / len(loss)
            writer.add_scalar('Loss/' + key, train_loss[key], epoch)
        for key, loss in val_loss.items():
            val_loss[key] = sum(loss) / len(loss)
            writer.add_scalar('Val loss/' + key, val_loss[key], epoch)
        early_stopping(val_loss['total_loss'], model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        writer.flush()
        print('epoch %d' % epoch)
        print('train: ', ''.join(['{}:{:0.4f}  '.format(key, loss) for key, loss in train_loss.items()]))
        print('validation: ', ''.join(['{}:{:0.4f}  '.format(key, loss) for key, loss in val_loss.items()]))
    writer.close()
    return model


def train_with_loader(model, train_loader, val_loader, output_dir,
          n_epochs=10, lr=0.001, device='cuda:0',
        patience=20, earlystop_metric='total_loss',
          retrain=False, log_step_offset=0):
    """ Train function using dataloders.

    Args:
        model (nn.Module): model
        train_loader (data loader): dataset of training inputs
        n_epochs (int, optional): number of epochs
        lr (float, optional): learning rate
        device (str): device to run the model on
        earlystop_metric (str): metric to monitor for early stopping
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
                labels = t.cat([label for label in labels], axis=0).to(device)
                batch = t.cat([datum for datum in data], axis=0).to(device)
                model, train_loss = \
                    run_one_batch(model, batch, train_loss, model_kwargs={'labels': labels}, optimizer=optimizer,
                                  transform=False, training=True)
        # loop through validation batches
        model.eval()
        with t.no_grad():
            with tqdm(val_loader, desc='val batch') as batch_pbar:
                for b_idx, batch in enumerate(batch_pbar):
                    labels, data = batch
                    labels = t.cat([label for label in labels], axis=0).to(device)
                    data = t.cat([datum for datum in data], axis=0).to(device)
                    model, val_loss = \
                        run_one_batch(model, data, val_loss, model_kwargs={'labels': labels}, optimizer=optimizer,
                                     transform=False, training=False)
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
                      output_dir,
                      use_channels=[],
                      relation_mat=None,
                      mask=None,
                      n_epochs=10,
                      lr_recon=0.001,
                      lr_dis=0.001,
                      lr_gen=0.001,
                      batch_size=16,
                      device='cuda:0',
                      shuffle_data=False,
                      transform=True,
                      seed=None):
    """ Train function for AAE.

    Args:
        model (nn.Module): autoencoder model (AAE)
        dataset (TensorDataset): dataset of training inputs
        output_dir (str): path for writing model saves and loss curves
        use_channels (list, optional): list of channel indices used for model
            training, by default all channels will be used
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
        device (str, optional): device (cuda or cpu) where models are running
        shuffle_data (bool, optional): shuffle data at the end of the epoch to
            add randomness to mini-batch; Set False when using matching loss
        transform (bool, optional): data augmentation
        seed (int, optional): random seed

    Returns:
        nn.Module: trained model

    """
    if not seed is None:
        np.random.seed(seed)
        t.manual_seed(seed)
    total_channels, n_z, x_size, y_size = dataset[0][0].shape[-4:]
    if len(use_channels) == 0:
        use_channels = list(range(total_channels))
    n_channels = len(use_channels)
    assert n_channels == model.num_inputs

    model = model.to(device)
    optim_enc = t.optim.Adam(model.enc.parameters(), lr_recon)
    optim_dec = t.optim.Adam(model.dec.parameters(), lr_recon)
    optim_enc_g = t.optim.Adam(model.enc.parameters(), lr_gen)
    optim_enc_d = t.optim.Adam(model.enc_d.parameters(), lr_dis)
    model.zero_grad()

    n_samples = len(dataset)
    n_batches = int(np.ceil(n_samples/batch_size))
    # Declare sample indices and do an initial shuffle
    sample_ids = np.arange(n_samples)
    if shuffle_data:
        np.random.shuffle(sample_ids)
    writer = SummaryWriter(output_dir)

    for epoch in range(n_epochs):
        mean_loss = {}
        print('start epoch %d' % epoch)
        for i in range(n_batches):
            # Deal with last batch might < batch size
            sample_ids_batch = sample_ids[i * batch_size:min((i + 1) * batch_size, n_samples)]
            batch = dataset[sample_ids_batch][0]
            assert len(batch.shape) == 5, "Input should be formatted as (batch, c, z, x, y)"
            batch = batch[:, np.array(use_channels)].permute(0, 2, 1, 3, 4).reshape((-1, n_channels, x_size, y_size))
            batch = batch.to(device)

            # Data augmentation
            if transform:
                for idx_in_batch in range(len(sample_ids_batch)):
                    img = batch[idx_in_batch]
                    flip_idx = np.random.choice([0, 1, 2])
                    if flip_idx != 0:
                        img = t.flip(img, dims=(flip_idx,))
                    rot_idx = int(np.random.choice([0, 1, 2, 3]))
                    batch[idx_in_batch] = t.rot90(img, k=rot_idx, dims=[1, 2])

            # Relation (adjacent frame, same trajectory)
            if not relation_mat is None:
                batch_relation_mat = relation_mat[sample_ids_batch][:, sample_ids_batch]
                batch_relation_mat = batch_relation_mat.todense()
                batch_relation_mat = t.from_numpy(batch_relation_mat).float().to(device)
            else:
                batch_relation_mat = None

            # Reconstruction mask
            if not mask is None:
                batch_mask = mask[sample_ids_batch][0][:, 1:2] # Hardcoded second slice (large mask)
                batch_mask = (batch_mask + 1.) / 2.
                batch_mask = batch_mask.to(device)
            else:
                batch_mask = None

            _, loss_dict = model(batch,
                                 time_matching_mat=batch_relation_mat,
                                 batch_mask=batch_mask)
            loss_dict['total_loss'].backward()
            optim_enc.step()
            optim_dec.step()
            loss_dict2 = model.adversarial_loss(batch)
            loss_dict2['descriminator_loss'].backward()
            optim_enc_d.step()
            loss_dict2['generator_loss'].backward()
            optim_enc_g.step()
            model.zero_grad()

            # Record loss
            for key, loss in loss_dict.items():
                if not key in mean_loss:
                    mean_loss[key] = []
                mean_loss[key].append(loss)

            for key, loss in loss_dict2.items():
                if not key in mean_loss:
                    mean_loss[key] = []
                mean_loss[key].append(loss)

        # shuffle samples ids at the end of the epoch
        if shuffle_data:
            np.random.shuffle(sample_ids)
        for key, loss in mean_loss.items():
            mean_loss[key] = sum(loss)/len(loss) if len(loss) > 0 else -1.
            writer.add_scalar('Loss/' + key, mean_loss[key], epoch)
        writer.flush()
        print('epoch %d' % epoch)
        print(''.join(['{}:{:0.4f}  '.format(key, loss) for key, loss in mean_loss.items()]))
        t.save(model.state_dict(), os.path.join(output_dir, 'model_epoch%d.pt' % epoch))
    writer.close()
    return model

def main(config_):
    """
    Args:
        config_ (object): config file object

    Returns:

    """
    config = YamlReader()
    config.read_config(config_)

    # Settings
    # estimate mean and std from the data
    channel_mean = None if config.training.channel_mean == 'None' else config.training.channel_mean
    channel_std = None if config.training.channel_std == 'None' else config.training.channel_std
    # channel_mean = config.training.channel_mean
    # channel_std = config.training.channel_std

    transform = config.training.augment

    raw_dirs = config.training.raw_dirs
    train_dirs = config.training.weights_dirs
    supp_dirs = config.training.supp_dirs
    for train_dir in train_dirs:
        os.makedirs(train_dir, exist_ok=True)

    init_weights = config.training.reload_weights if config.training.reload_weights else None

    ### Settings ###
    network = config.training.network
    num_inputs = config.training.num_inputs
    num_hiddens = config.training.num_hiddens
    num_residual_hiddens = config.training.num_residual_hiddens
    num_residual_layers = config.training.num_residual_layers
    num_embeddings = config.training.num_embeddings
    commitment_cost = config.training.commitment_cost
    weight_matching = config.training.weight_matching
    w_a = config.training.w_a
    w_t = config.training.w_t
    w_n = config.training.w_n
    margin = config.training.margin
    val_split_ratio = config.training.val_split_ratio
    learn_rate = config.training.learn_rate
    patience = config.training.patience
    n_pos_samples = config.training.n_pos_samples
    batch_size = config.training.batch_size
    # adjusted batch size for dataloaders
    batch_size_adj = int(np.floor(batch_size/n_pos_samples))
    num_workers = config.training.num_workers
    n_epochs = config.training.n_epochs
    gpu_id = config.training.gpu_id
    # earlystop_metric = 'total_loss'
    retrain = config.training.retrain
    earlystop_metric = 'positive_triplet'
    # model_name = 'A549_{}_mrg{}_npos{}_bh{}_alltriloss_tr'.format(
    model_name = config.training.model_name
    start_model_path = config.training.start_model_path
    start_epoch = config.training.start_epoch
    use_mask = config.training.use_mask

    cs = [0, 1]
    cs_mask = [2, 3]
    input_shape = (128, 128)

    device = t.device('cuda:%d' % gpu_id)

    # use data loader for training ResNet
    use_loader = False
    if 'ResNet' in network:
        use_loader = True

    dir_sets = list(zip(supp_dirs, train_dirs, raw_dirs))
    # dir_sets = dir_sets[0:1]
    ts_keys = []
    datasets = []
    masks = []
    relations = []
    labels = []
    id_offsets = [0]
    well = config.training.well_row
    ### Load Data ###
    for supp_dir, train_dir, raw_dir in dir_sets:
        os.makedirs(train_dir, exist_ok=True)
        print(f"\tloading file paths {os.path.join(raw_dir, f'{well}_file_paths.pkl')}")
        ts_key = pickle.load(open(os.path.join(raw_dir, f'{well}_file_paths.pkl'), 'rb'))
        print(f"\tloading static patches {os.path.join(raw_dir, f'{well}_static_patches.pkl')}")
        dataset = pickle.load(open(os.path.join(raw_dir, f'{well}_static_patches.pkl'), 'rb'))
        print('dataset.shape:', dataset.shape)

        # todo: not sure if reshaping here is wise -- doing this to compat with zscoring
        # dataset = dataset.reshape((dataset.shape[0], dataset.shape[1], dataset.shape[3], dataset.shape[4]))

        label = pickle.load(open(os.path.join(raw_dir, f"{well}_static_patches_labels.pkl"), 'rb'))
        # Note that `relations` is depending on the order of fs (should not sort)
        # `relations` is generated by script "generate_trajectory_relations.py"
        relation = pickle.load(open(os.path.join(raw_dir, f'{well}_static_patches_relations.pkl'), 'rb'))
        # dataset_mask = TensorDataset(dataset_mask.tensors[0][np.array(inds_in_order)])
        # print('relations:', relations)
        print('len(ts_key):', len(ts_key))
        print('len(dataset):', len(dataset))
        relations.append(relation)
        ts_keys += ts_key
        # TODO: handle non-singular z-dimension case earlier in the pipeline
        # dataset = zscore(np.squeeze(dataset),
        #                  channel_mean=channel_mean,
        #                  channel_std=channel_std,
        #                  channel_num=config.training.num_channels).astype(np.float32)
        datasets.append(dataset)
        labels.append(label)
        id_offsets.append(len(dataset))
        if use_mask:
            mask = pickle.load(open(os.path.join(raw_dir, f'{well}_static_patches_mask.pkl'), 'rb'))
            masks.append(mask)
    id_offsets = id_offsets[:-1]
    dataset = np.concatenate(datasets, axis=0)
    if use_mask:
        masks = np.concatenate(masks, axis=0)
    else:
        masks = None

    # zscoring here is hardcoded based on dimensions
    dataset = zscore(dataset,
                     channel_mean=channel_mean,
                     channel_std=channel_std,
                     channel_num=config.training.num_channels).astype(np.float32)

    # reshape here because the z dimension is not handled elsewhere
    dataset = dataset.reshape((dataset.shape[0], dataset.shape[1], dataset.shape[3], dataset.shape[4]))

    relations, labels = concat_relations(relations, labels, offsets=id_offsets)
    # Save the model in the train directory of the last dataset
    model_dir = os.path.join(train_dir, model_name)
    #TODO: write dataset class for VAE models
    if not use_loader:
        dataset = TensorDataset(t.from_numpy(dataset).float())
        dataset, relation_mat, inds_in_order = reorder_with_trajectories(dataset, relations, seed=123)
        labels = labels[inds_in_order]
        network_cls = getattr(vae, network)
        model = network_cls(num_inputs=num_inputs,
                            num_hiddens=num_hiddens,
                            num_residual_hiddens=num_residual_hiddens,
                            num_residual_layers=num_residual_layers,
                            num_embeddings=num_embeddings,
                            commitment_cost=commitment_cost,
                            weight_matching=weight_matching,
                            w_a=w_a,
                            w_t=w_t,
                            w_n=w_n,
                            margin=margin,
                            device=device).to(device)

        if init_weights:
            model.load_state_dict(t.load(init_weights[0]))

        model = train(model,
                      dataset,
                      output_dir=model_dir,
                      relation_mat=relation_mat,
                      mask=masks,
                      n_epochs=n_epochs,
                      lr=learn_rate,
                      batch_size=batch_size,
                      device=device,
                      transform=transform,
                      val_split_ratio=val_split_ratio,
                      patience=patience,
                      )
    else:
        train_set, train_labels, val_set, val_labels = \
            train_val_split(dataset, labels, val_split_ratio=val_split_ratio, seed=0)
        tri_train_set = TripletDataset(train_labels, lambda index: augment_img(train_set[index]), n_pos_samples)
        tri_val_set = TripletDataset(val_labels, lambda index: augment_img(val_set[index]), n_pos_samples)
        # Data Loader
        train_loader = DataLoader(tri_train_set,
                                    batch_size=batch_size_adj,
                                    shuffle=True,
                                    num_workers=num_workers,
                                    pin_memory=False,
                                    )
        val_loader = DataLoader(tri_val_set,
                                  batch_size=batch_size_adj,
                                  shuffle=False,
                                  num_workers=num_workers,
                                  pin_memory=False,
                                  )
        tri_loss = AllTripletMiner(margin=margin).to(device)
        # tri_loss = HardNegativeTripletMiner(margin=margin).to(device)
        ## Initialize Model ###

        model = EncodeProject(arch=network, loss=tri_loss, num_inputs=num_inputs).to(device)

        if start_model_path:
            print('Initialize the model with state {} ...'.format(start_model_path))
            model.load_state_dict(t.load(start_model_path))
        model = train_with_loader(model,
                              train_loader=train_loader,
                              val_loader=val_loader,
                              output_dir=model_dir,
                              n_epochs=n_epochs,
                              lr=learn_rate,
                              device=device,
                              patience=patience,
                              earlystop_metric=earlystop_metric,
                              retrain=retrain,
                              log_step_offset=start_epoch)

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-c', '--config',
        type=str,
        required=True,
        help='path to yaml configuration file'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args.config)

