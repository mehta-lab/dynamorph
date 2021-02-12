# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 22:38:24 2021

@author: Zhenqin Wu
"""
import os
import h5py
import cv2
import numpy as np
import scipy
import queue
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.utils.data import TensorDataset, DataLoader
from scipy.sparse import csr_matrix
from .naive_imagenet import read_file_path

CHANNEL_RANGE = [(0.3, 0.8), (0., 0.6)] 
CHANNEL_VAR = np.array([0.0475, 0.0394]) # After normalized to CHANNEL_RANGE
CHANNEL_MAX = 65535.
eps = 1e-9

def cv2_fn_wrapper(cv2_fn, mat, *args, **kwargs):
    """" A wrapper for cv2 functions
    
    Data in channel first format are adjusted to channel last format for 
    cv2 functions
    """
    
    mat_shape = mat.shape
    x_size = mat_shape[-2]
    y_size = mat_shape[-1]
    _mat = mat.reshape((-1, x_size, y_size)).transpose((1, 2, 0))
    _output = cv2_fn(_mat, *args, **kwargs)
    _x_size = _output.shape[0]
    _y_size = _output.shape[1]
    output_shape = tuple(list(mat_shape[:-2]) + [_x_size, _y_size])
    output = _output.transpose((2, 0, 1)).reshape(output_shape)
    return output


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
            tensors.append(t.from_numpy(resized_dat).float())
    dataset = TensorDataset(t.stack(tensors, 0))
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
            tensors[f_n] = t.from_numpy(resized_dat).float()
    dataset = TensorDataset(t.stack([tensors[f_n] for f_n in fs], 0))
    return dataset

def prepare_dataset_v2(dat_fs,
                       cs=[0, 1],
                       input_shape=(128, 128),
                       key='masked_mat'):
    """ Prepare input dataset for VAE

    This function reads assembled pickle files (dict)

    Args:
        dat_fs (list of str): list of pickle file paths
        cs (list of int, optional): channels in the input
        input_shape (tuple, optional): input shape (height and width only)
        key (str): 'mat' or 'masked_mat'

    Returns:
        np array: dataset of training inputs
        list of str: identifiers of single cell image patches

    """
    tensors = {}
    for dat_f in dat_fs:
        print(f"\tloading data {dat_f}")
        file_dats = pickle.load(open(dat_f, 'rb'))
        for k in file_dats:
            dat = file_dats[k][key]
            if cs is None:
                cs = np.arange(dat.shape[0])
            dat = np.array(dat)[np.array(cs)].astype(float)
            resized_dat = cv2_fn_wrapper(cv2.resize, dat, input_shape)
            tensors[k] = resized_dat
    ts_keys = sorted(tensors.keys())
    dataset = np.stack([tensors[key] for key in ts_keys], 0)
    return dataset, ts_keys

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
            values.append(0.1)
        elif v == 2:
            values.append(1.1)
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
                   clamp=[0, 1]):
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
    
    tensor = dataset.tensors[0]
    output = []
    for channel in use_channels:
        channel_slice = tensor[:, channel].float()
        channel_slice = channel_slice / CHANNEL_MAX # Scale to [0, 1]
        if preprocess_setting[channel][0] == "scale":
            target_mean = preprocess_setting[channel][1]
            slice_mean = tensor[:, channel].mean() / CHANNEL_MAX
            output_slice = channel_slice / slice_mean * target_mean
        elif preprocess_setting[channel][0] == "normalize":
            target_mean = preprocess_setting[channel][1]
            target_sd = preprocess_setting[channel][2]
            slice_mean = tensor[:, channel].mean() / CHANNEL_MAX
            slice_sd = tensor[:, channel].std() / CHANNEL_MAX
            z_channel_slice = (channel_slice - slice_mean) / slice_sd
            output_slice = z_channel_slice * target_sd + target_mean
        else:
            raise ValueError("Preprocessing mode not supported")
        if clamp:
            output_slice = t.clamp(output_slice, clamp[0], clamp[1])
        output.append(output_slice)
    output = t.stack(output, 1)
    return TensorDataset(output)


def train(model, 
          dataset, 
          use_channels=[],
          relation_mat=None, 
          mask=None, 
          n_epochs=10, 
          lr=0.001, 
          batch_size=16, 
          device='cuda:0'):
    """ Train function for VQ-VAE, VAE, IWAE, etc.

    Args:
        model (nn.Module): autoencoder model
        dataset (TensorDataset): dataset of training inputs
        use_channels (list, optional): list of channel indices used for model
            training, by default all channels will be used
        relation_mat (scipy csr matrix or None, optional): if given, sparse 
            matrix of pairwise relations
        mask (TensorDataset or None, optional): if given, dataset of training 
            sample weight masks
        n_epochs (int, optional): number of epochs
        lr (float, optional): learning rate
        batch_size (int, optional): batch size
        device (str, optional): device (cuda or cpu) where models are running
    
    Returns:
        nn.Module: trained model

    """
    total_channels, n_z, x_size, y_size = dataset[0][0].shape[-4:]
    if len(use_channels) == 0:
        use_channels = list(range(total_channels))
    n_channels = len(use_channels)
    assert n_channels == model.num_inputs
    
    model = model.to(device)
    optimizer = t.optim.Adam(model.parameters(), lr=lr, betas=(.9, .999))
    model.zero_grad()
    
    n_batches = int(np.ceil(len(dataset)/batch_size))
    for epoch in range(n_epochs):
        recon_loss = []
        perplexities = []
        print('start epoch %d' % epoch) 
        for i in range(n_batches):
            # Input data
            batch = dataset[i*batch_size:(i+1)*batch_size][0]
            assert len(batch.shape) == 5, "Input should be formatted as (batch, c, z, x, y)"
            batch = batch[:, np.array(use_channels)].permute(0, 2, 1, 3, 4).reshape((-1, n_channels, x_size, y_size))
            batch = batch.to(device)
              
            # Relation (adjacent frame, same trajectory)
            if not relation_mat is None:
                batch_relation_mat = relation_mat[i*batch_size:(i+1)*batch_size, 
                                                  i*batch_size:(i+1)*batch_size]
                batch_relation_mat = batch_relation_mat.todense()
                batch_relation_mat = t.from_numpy(batch_relation_mat).float().to(device)
            else:
                batch_relation_mat = None
            
            # Reconstruction mask
            if not mask is None:
                batch_mask = mask[i*batch_size:(i+1)*batch_size][0][:, 1:2] # Hardcoded second slice (large mask)
                batch_mask = (batch_mask + 1.)/2. # Add a baseline weight
                batch_mask = batch_mask.permute(0, 2, 1, 3, 4).reshape((-1, 1, x_size, y_size))
                batch_mask = batch_mask.to(device)
            else:
                batch_mask = None
              
            _, loss_dict = model(batch, 
                                 time_matching_mat=batch_relation_mat, 
                                 batch_mask=batch_mask)
            loss_dict['total_loss'].backward()
            optimizer.step()
            model.zero_grad()

            recon_loss.append(loss_dict['recon_loss'])
            perplexities.append(loss_dict['perplexity'])
        print('epoch %d recon loss: %f perplexity: %f' % \
            (epoch, sum(recon_loss).item()/len(recon_loss), sum(perplexities).item()/len(perplexities)))
    return model


def train_adversarial(model, 
                      dataset,
                      use_channels=[],
                      relation_mat=None, 
                      mask=None, 
                      n_epochs=10, 
                      lr_recon=0.001, 
                      lr_dis=0.001, 
                      lr_gen=0.001, 
                      batch_size=16, 
                      device='cuda:0'):
    """ Train function for AAE.

    Args:
        model (nn.Module): autoencoder model (AAE)
        dataset (TensorDataset): dataset of training inputs
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
    
    Returns:
        nn.Module: trained model

    """
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

    n_batches = int(np.ceil(len(dataset)/batch_size))
    for epoch in range(n_epochs):
        recon_loss = []
        scores = []
        print('start epoch %d' % epoch) 
        for i in range(n_batches):
            # Input data
            batch = dataset[i*batch_size:(i+1)*batch_size][0]
            assert len(batch.shape) == 5, "Input should be formatted as (batch, c, z, x, y)"
            batch = batch[:, np.array(use_channels)].permute(0, 2, 1, 3, 4).reshape((-1, n_channels, x_size, y_size))
            batch = batch.to(device)
            
            # Relation (adjacent frame, same trajectory)
            if not relation_mat is None:
                batch_relation_mat = relation_mat[i*batch_size:(i+1)*batch_size, 
                                                  i*batch_size:(i+1)*batch_size]
                batch_relation_mat = batch_relation_mat.todense()
                batch_relation_mat = t.from_numpy(batch_relation_mat).float().to(device)
            else:
                batch_relation_mat = None
            
            # Reconstruction mask
            if not mask is None:
                batch_mask = mask[i*batch_size:(i+1)*batch_size][0][:, 1:2] # Hardcoded second slice (large mask)
                batch_mask = (batch_mask + 1.)/2. # Add a baseline weight
                batch_mask = batch_mask.permute(0, 2, 1, 3, 4).reshape((-1, 1, x_size, y_size))
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

            recon_loss.append(loss_dict['recon_loss'])
            scores.append(loss_dict2['score'])
        print('epoch %d recon loss: %f pred score: %f' % (epoch, sum(recon_loss).item()/len(recon_loss), sum(scores).item()/len(scores)))
    return model