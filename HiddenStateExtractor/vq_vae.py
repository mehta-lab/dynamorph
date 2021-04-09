# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 14:22:51 2019

@author: michael.wu
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


CHANNEL_VAR = np.array([1., 1.])
CHANNEL_MAX = 65535.
eps = 1e-9


class VectorQuantizer(nn.Module):
    """ Vector Quantizer module as introduced in 
        "Neural Discrete Representation Learning"

    This module contains a list of trainable embedding vectors, during training 
    and inference encodings of inputs will find their closest resemblance
    in this list, which will be reassembled as quantized encodings (decoder 
    input)

    """
    def __init__(self, embedding_dim=128, num_embeddings=128, commitment_cost=0.25, device='cuda:0'):
        """ Initialize the module

        Args:
            embedding_dim (int, optional): size of embedding vector
            num_embeddings (int, optional): number of embedding vectors
            commitment_cost (float, optional): balance between latent losses
            device (str, optional): device the model will be running on

        """
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.device = device
        self.w = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, inputs):
        """ Forward pass

        Args:
            inputs (torch tensor): encodings of input image

        Returns:
            torch tensor: quantized encodings (decoder input)
            torch tensor: quantization loss
            torch tensor: perplexity, measuring coverage of embedding vectors

        """
        # inputs: Batch * Num_hidden(=embedding_dim) * H * W
        distances = t.sum((inputs.unsqueeze(1) - self.w.weight.reshape((1, self.num_embeddings, self.embedding_dim, 1, 1)))**2, 2)

        # Decoder input
        encoding_indices = t.argmax(-distances, 1)
        quantized = self.w(encoding_indices).transpose(2, 3).transpose(1, 2)
        assert quantized.shape == inputs.shape
        output_quantized = inputs + (quantized - inputs).detach()

        # Commitment loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Perplexity (used to monitor)
        encoding_onehot = t.zeros(encoding_indices.flatten().shape[0], self.num_embeddings).to(self.device)
        encoding_onehot.scatter_(1, encoding_indices.flatten().unsqueeze(1), 1)
        avg_probs = t.mean(encoding_onehot, 0)
        perplexity = t.exp(-t.sum(avg_probs*t.log(avg_probs + 1e-10)))

        return output_quantized, loss, perplexity

    @property
    def embeddings(self):
        return self.w.weight

    def encode_inputs(self, inputs):
        """ Find closest embedding vector combinations of input encodings

        Args:
            inputs (torch tensor): encodings of input image

        Returns:
            torch tensor: index tensor of embedding vectors
            
        """
        # inputs: Batch * Num_hidden(=embedding_dim) * H * W
        distances = t.sum((inputs.unsqueeze(1) - self.w.weight.reshape((1, self.num_embeddings, self.embedding_dim, 1, 1)))**2, 2)
        encoding_indices = t.argmax(-distances, 1)
        return encoding_indices

    def decode_inputs(self, encoding_indices):
        """ Assemble embedding vector index to quantized encodings

        Args:
            encoding_indices (torch tensor): index tensor of embedding vectors

        Returns:
            torch tensor: quantized encodings (decoder input)
            
        """
        quantized = self.w(encoding_indices).transpose(2, 3).transpose(1, 2)
        return quantized
      

class Reparametrize(nn.Module):
    """ Reparameterization step in RegularVAE
    """
    def forward(self, z_mean, z_logstd):
        """ Forward pass
        
        Args:
            z_mean (torch tensor): latent vector mean
            z_logstd (torch tensor): latent vector std (log)

        Returns:
            torch tensor: reparameterized latent vector
            torch tensor: KL divergence

        """
        z_std = t.exp(0.5 * z_logstd)
        eps = t.randn_like(z_std)
        z = z_mean + z_std * eps    
        KLD = -0.5 * t.sum(1 + z_logstd - z_mean.pow(2) - z_logstd.exp())
        return z, KLD


class Reparametrize_IW(nn.Module):
    """ Reparameterization step in IWAE
    """
    def __init__(self, k=5, **kwargs):
        """ Initialize the module

        Args:
            k (int, optional): number of sampling trials
            **kwargs: other keyword arguments

        """
        super(Reparametrize_IW, self).__init__(**kwargs)
        self.k = k
      
    def forward(self, z_mean, z_logstd):
        """ Forward pass
        
        Args:
            z_mean (torch tensor): latent vector mean
            z_logstd (torch tensor): latent vector std (log)

        Returns:
            torch tensor: reparameterized latent vectors
            torch tensor: randomness

        """
        z_std = t.exp(0.5 * z_logstd)
        epss = [t.randn_like(z_std) for _ in range(self.k)]
        zs = [z_mean + z_std * eps for eps in epss]
        return zs, epss


class Flatten(nn.Module):
    """ Helper module for flatten tensor
    """
    def forward(self, input):
        return input.view(input.size(0), -1)


class ResidualBlock(nn.Module):
    """ Customized residual block in network
    """
    def __init__(self,
                 num_hiddens=128,
                 num_residual_hiddens=512,
                 num_residual_layers=2):
        """ Initialize the module

        Args:
            num_hiddens (int, optional): number of hidden units
            num_residual_hiddens (int, optional): number of hidden units in the
                residual layer
            num_residual_layers (int, optional): number of residual layers

        """
        super(ResidualBlock, self).__init__()
        self.num_hiddens = num_hiddens
        self.num_residual_layers = num_residual_layers
        self.num_residual_hiddens = num_residual_hiddens

        self.layers = []
        for _ in range(self.num_residual_layers):
            self.layers.append(nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(self.num_hiddens, self.num_residual_hiddens, 3, padding=1),
                nn.BatchNorm2d(self.num_residual_hiddens),
                nn.ReLU(),
                nn.Conv2d(self.num_residual_hiddens, self.num_hiddens, 1),
                nn.BatchNorm2d(self.num_hiddens)))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        """ Forward pass

        Args:
            x (torch tensor): input tensor

        Returns:
            torch tensor: output tensor

        """
        output = x
        for i in range(self.num_residual_layers):
            output = output + self.layers[i](output)
        return output


class VQ_VAE(nn.Module):
    """ Vector-Quantized VAE as introduced in 
        "Neural Discrete Representation Learning"
    """
    def __init__(self,
                 num_inputs=2,
                 num_hiddens=16,
                 num_residual_hiddens=32,
                 num_residual_layers=2,
                 num_embeddings=64,
                 commitment_cost=0.25,
                 channel_var=CHANNEL_VAR,
                 weight_recon=1.,
                 weight_commitment=1.,
                 weight_matching=0.005,
                 device="cuda:0",
                 **kwargs):
        """ Initialize the model

        Args:
            num_inputs (int, optional): number of channels in input
            num_hiddens (int, optional): number of hidden units (size of latent 
                encodings per position)
            num_residual_hiddens (int, optional): number of hidden units in the
                residual layer
            num_residual_layers (int, optional): number of residual layers
            num_embeddings (int, optional): number of VQ embedding vectors
            commitment_cost (float, optional): balance between latent losses
            channel_var (list of float, optional): each channel's SD, used for 
                balancing loss across channels
            weight_recon (float, optional): balance of reconstruction loss
            weight_commitment (float, optional): balance of commitment loss
            weight_matching (float, optional): balance of matching loss
            device (str, optional): device the model will be running on
            **kwargs: other keyword arguments

        """
        super(VQ_VAE, self).__init__(**kwargs)
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_residual_layers = num_residual_layers
        self.num_residual_hiddens = num_residual_hiddens
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.channel_var = nn.Parameter(t.from_numpy(channel_var).float().reshape((1, num_inputs, 1, 1)), requires_grad=False)
        self.weight_recon = weight_recon
        self.weight_commitment = weight_commitment
        self.weight_matching = weight_matching
        self.enc = nn.Sequential(
            nn.Conv2d(self.num_inputs, self.num_hiddens//2, 1),
            nn.Conv2d(self.num_hiddens//2, self.num_hiddens//2, 4, stride=2, padding=1),
            nn.BatchNorm2d(self.num_hiddens//2),
            nn.ReLU(),
            nn.Conv2d(self.num_hiddens//2, self.num_hiddens, 4, stride=2, padding=1),
            nn.BatchNorm2d(self.num_hiddens),
            nn.ReLU(),
            nn.Conv2d(self.num_hiddens, self.num_hiddens, 4, stride=2, padding=1),
            nn.BatchNorm2d(self.num_hiddens),
            nn.ReLU(),
            nn.Conv2d(self.num_hiddens, self.num_hiddens, 3, padding=1),
            nn.BatchNorm2d(self.num_hiddens),
            ResidualBlock(self.num_hiddens, self.num_residual_hiddens, self.num_residual_layers))
        self.vq = VectorQuantizer(self.num_hiddens, self.num_embeddings, commitment_cost=self.commitment_cost, device=device)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(self.num_hiddens, self.num_hiddens//2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.num_hiddens//2, self.num_hiddens//4, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.num_hiddens//4, self.num_hiddens//4, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_hiddens//4, self.num_inputs, 1))
      
    def forward(self, inputs, time_matching_mat=None, batch_mask=None):
        """ Forward pass

        Args:
            inputs (torch tensor): input cell image patches
            time_matching_mat (torch tensor or None, optional): if given, 
                pairwise relationship between samples in the minibatch, used 
                to calculate time matching loss
            batch_mask (torch tensor or None, optional): if given, weight mask 
                of training samples, used to concentrate loss on cell bodies

        Returns:
            torch tensor: decoded/reconstructed cell image patches
            dict: losses and perplexity of the minibatch

        """
        # inputs: Batch * num_inputs(channel) * H * W, each channel from 0 to 1
        z_before = self.enc(inputs)
        z_after, c_loss, perplexity = self.vq(z_before)
        decoded = self.dec(z_after)
        if batch_mask is None:
            batch_mask = t.ones_like(inputs)
        recon_loss = t.mean(F.mse_loss(decoded * batch_mask, inputs * batch_mask, reduction='none') / self.channel_var)
        total_loss = self.weight_recon * recon_loss + self.weight_commitment * c_loss
        time_matching_loss = 0.
        if not time_matching_mat is None:
            z_before_ = z_before.reshape((z_before.shape[0], -1))
            len_latent = z_before_.shape[1]
            sim_mat = t.pow(z_before_.reshape((1, -1, len_latent)) - \
                            z_before_.reshape((-1, 1, len_latent)), 2).mean(2)
            assert sim_mat.shape == time_matching_mat.shape
            time_matching_loss = (sim_mat * time_matching_mat).sum()
            total_loss += self.weight_matching * time_matching_loss
        return decoded, \
               {'recon_loss': recon_loss,
                'commitment_loss': c_loss,
                'time_matching_loss': time_matching_loss,
                'total_loss': total_loss,
                'perplexity': perplexity}

    def predict(self, inputs):
        """ Prediction fn, same as forward pass """
        return self.forward(inputs)


class VAE(nn.Module):
    """ Regular VAE """
    def __init__(self,
                 num_inputs=2,
                 num_hiddens=16,
                 num_residual_hiddens=32,
                 num_residual_layers=2,
                 channel_var=CHANNEL_VAR,
                 weight_recon=1.,
                 weight_kld=1.,
                 weight_matching=0.005,
                 **kwargs):
        """ Initialize the model

        Args:
            num_inputs (int, optional): number of channels in input
            num_hiddens (int, optional): number of hidden units (size of latent 
                encodings per position)
            num_residual_hiddens (int, optional): number of hidden units in the
                residual layer
            num_residual_layers (int, optional): number of residual layers
            channel_var (list of float, optional): each channel's SD, used for 
                balancing loss across channels
            weight_recon (float, optional): balance of reconstruction loss
            weight_kld (float, optional): balance of KL divergence
            weight_matching (float, optional): balance of matching loss
            **kwargs: other keyword arguments

        """
        super(VAE, self).__init__(**kwargs)
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_residual_layers = num_residual_layers
        self.num_residual_hiddens = num_residual_hiddens
        self.channel_var = nn.Parameter(t.from_numpy(channel_var).float().reshape((1, num_inputs, 1, 1)), requires_grad=False)
        self.weight_recon = weight_recon
        self.weight_kld = weight_kld
        self.weight_matching = weight_matching
        self.enc = nn.Sequential(
            nn.Conv2d(self.num_inputs, self.num_hiddens//2, 1),
            nn.Conv2d(self.num_hiddens//2, self.num_hiddens//2, 4, stride=2, padding=1),
            nn.BatchNorm2d(self.num_hiddens//2),
            nn.ReLU(),
            nn.Conv2d(self.num_hiddens//2, self.num_hiddens, 4, stride=2, padding=1),
            nn.BatchNorm2d(self.num_hiddens),
            nn.ReLU(),
            nn.Conv2d(self.num_hiddens, self.num_hiddens, 4, stride=2, padding=1),
            nn.BatchNorm2d(self.num_hiddens),
            nn.ReLU(),
            nn.Conv2d(self.num_hiddens, self.num_hiddens, 3, padding=1),
            nn.BatchNorm2d(self.num_hiddens),
            ResidualBlock(self.num_hiddens, self.num_residual_hiddens, self.num_residual_layers),
            nn.Conv2d(self.num_hiddens, 2*self.num_hiddens, 1))
        self.rp = Reparametrize()
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(self.num_hiddens, self.num_hiddens//2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.num_hiddens//2, self.num_hiddens//4, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.num_hiddens//4, self.num_hiddens//4, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_hiddens//4, self.num_inputs, 1))
      
    def forward(self, inputs, time_matching_mat=None, batch_mask=None):
        """ Forward pass

        Args:
            inputs (torch tensor): input cell image patches
            time_matching_mat (torch tensor or None, optional): if given, 
                pairwise relationship between samples in the minibatch, used 
                to calculate time matching loss
            batch_mask (torch tensor or None, optional): if given, weight mask 
                of training samples, used to concentrate loss on cell bodies

        Returns:
            torch tensor: decoded/reconstructed cell image patches
            dict: losses and perplexity of the minibatch

        """
        # inputs: Batch * num_inputs(channel) * H * W, each channel from 0 to 1
        z_before = self.enc(inputs)
        z_mean = z_before[:, :self.num_hiddens]
        z_logstd = z_before[:, self.num_hiddens:]

        # Reparameterization trick
        z_after, KLD = self.rp(z_mean, z_logstd)
        
        decoded = self.dec(z_after)
        if batch_mask is None:
            batch_mask = t.ones_like(inputs)
        recon_loss = t.sum(F.mse_loss(decoded * batch_mask, inputs * batch_mask, reduction='none')/self.channel_var)
        total_loss = self.weight_recon * recon_loss + self.weight_kld * KLD
        time_matching_loss = 0.
        if not time_matching_mat is None:
            z_before_ = z_mean.reshape((z_mean.shape[0], -1))
            len_latent = z_before_.shape[1]
            sim_mat = t.pow(z_before_.reshape((1, -1, len_latent)) - \
                            z_before_.reshape((-1, 1, len_latent)), 2).mean(2)
            assert sim_mat.shape == time_matching_mat.shape
            time_matching_loss = (sim_mat * time_matching_mat).sum()
            total_loss += self.weight_matching * time_matching_loss
        return decoded, \
               {'recon_loss': recon_loss/(inputs.shape[0] * 32768),
                'KLD': KLD,
                'time_matching_loss': time_matching_loss,
                'total_loss': total_loss,
                'perplexity': t.zeros(())}

    def predict(self, inputs):
        """ Prediction fn without reparameterization

        Args:
            inputs (torch tensor): input cell image patches

        Returns:
            torch tensor: decoded/reconstructed cell image patches
            dict: reconstruction loss

        """
        # inputs: Batch * num_inputs(channel) * H * W, each channel from 0 to 1
        z_before = self.enc(inputs)
        z_mean = z_before[:, :self.num_hiddens]
        decoded = self.dec(z_mean)
        recon_loss = t.mean(F.mse_loss(decoded, inputs, reduction='none')/self.channel_var)
        return decoded, {'recon_loss': recon_loss}


class IWAE(VAE):
    """ Importance Weighted Autoencoder as introduced in 
        "Importance Weighted Autoencoders"
    """
    def __init__(self, k=5, **kwargs):
        """ Initialize the model

        Args:
            k (int, optional): number of sampling trials
            **kwargs: other keyword arguments (including arguments for `VAE`)

        """
        super(IWAE, self).__init__(**kwargs)
        self.k = k
        self.rp = Reparametrize_IW(k=self.k)

    def forward(self, inputs, time_matching_mat=None, batch_mask=None):
        """ Forward pass

        Args:
            inputs (torch tensor): input cell image patches
            time_matching_mat (torch tensor or None, optional): if given, 
                pairwise relationship between samples in the minibatch, used 
                to calculate time matching loss
            batch_mask (torch tensor or None, optional): if given, weight mask 
                of training samples, used to concentrate loss on cell bodies

        Returns:
            None: placeholder
            dict: losses and perplexity of the minibatch

        """
        z_before = self.enc(inputs)
        z_mean = z_before[:, :self.num_hiddens]
        z_logstd = z_before[:, self.num_hiddens:]
        z_afters, epss = self.rp(z_mean, z_logstd)

        if batch_mask is None:
            batch_mask = t.ones_like(inputs)
        time_matching_loss = 0.
        if not time_matching_mat is None:
            z_before_ = z_mean.reshape((z_mean.shape[0], -1))
            len_latent = z_before_.shape[1]
            sim_mat = t.pow(z_before_.reshape((1, -1, len_latent)) - \
                            z_before_.reshape((-1, 1, len_latent)), 2).mean(2)
            assert sim_mat.shape == time_matching_mat.shape
            time_matching_loss = (sim_mat * time_matching_mat).sum()

        log_ws = []
        recon_losses = []
        for z, eps in zip(z_afters, epss):
            decoded = self.dec(z)
            log_p_x_z = - t.sum(F.mse_loss(decoded * batch_mask, inputs * batch_mask, reduction='none')/self.channel_var, dim=(1, 2, 3))
            log_p_z = - t.sum(0.5 * z ** 2, dim=(1, 2, 3)) #- 0.5 * t.numel(z[0]) * np.log(2 * np.pi)
            log_q_z_x = - t.sum(0.5 * eps ** 2 + z_logstd, dim=(1, 2, 3)) #- 0.5 * t.numel(z[0]) * np.log(2 * np.pi) 
            log_w_unnormed = log_p_x_z  + log_p_z - log_q_z_x
            log_ws.append(log_w_unnormed)
            recon_losses.append(-log_p_x_z)
        log_ws = t.stack(log_ws, 1)
        log_ws_minus_max = log_ws - t.max(log_ws, dim=1, keepdim=True)[0]
        ws = t.exp(log_ws_minus_max)
        normalized_ws = ws / t.sum(ws, dim=1, keepdim=True)
        loss = -(normalized_ws.detach() * log_ws).sum()
        total_loss = loss + self.weight_matching * time_matching_loss
        
        recon_losses = t.stack(recon_losses, 1)
        recon_loss = (normalized_ws.detach() * recon_losses).sum()
        return None, \
               {'recon_loss': recon_loss/(inputs.shape[0] * 32768),
                'time_matching_loss': time_matching_loss,
                'total_loss': total_loss,
                'perplexity': t.zeros(())}


class AAE(nn.Module):
    """ Adversarial Autoencoder as introduced in 
        "Adversarial Autoencoders"
    """
    def __init__(self,
                 num_inputs=2,
                 num_hiddens=16,
                 num_residual_hiddens=32,
                 num_residual_layers=2,
                 channel_var=CHANNEL_VAR,
                 weight_recon=1.,
                 weight_matching=0.005,
                 **kwargs):
        """ Initialize the model

        Args:
            num_inputs (int, optional): number of channels in input
            num_hiddens (int, optional): number of hidden units (size of latent 
                encodings per position)
            num_residual_hiddens (int, optional): number of hidden units in the
                residual layer
            num_residual_layers (int, optional): number of residual layers
            channel_var (list of float, optional): each channel's SD, used for 
                balancing loss across channels
            weight_recon (float, optional): balance of reconstruction loss
            weight_matching (float, optional): balance of matching loss
            **kwargs: other keyword arguments

        """
        super(AAE, self).__init__(**kwargs)
        self.num_inputs = num_inputs
        self.num_hiddens = num_hiddens
        self.num_residual_layers = num_residual_layers
        self.num_residual_hiddens = num_residual_hiddens
        self.channel_var = nn.Parameter(t.from_numpy(channel_var).float().reshape((1, num_inputs, 1, 1)), requires_grad=False)
        self.weight_recon = weight_recon
        self.weight_matching = weight_matching
        self.enc = nn.Sequential(
            nn.Conv2d(self.num_inputs, self.num_hiddens//2, 1),
            nn.Conv2d(self.num_hiddens//2, self.num_hiddens//2, 4, stride=2, padding=1),
            nn.BatchNorm2d(self.num_hiddens//2),
            nn.ReLU(),
            nn.Conv2d(self.num_hiddens//2, self.num_hiddens, 4, stride=2, padding=1),
            nn.BatchNorm2d(self.num_hiddens),
            nn.ReLU(),
            nn.Conv2d(self.num_hiddens, self.num_hiddens, 4, stride=2, padding=1),
            nn.BatchNorm2d(self.num_hiddens),
            nn.ReLU(),
            nn.Conv2d(self.num_hiddens, self.num_hiddens, 3, padding=1),
            nn.BatchNorm2d(self.num_hiddens),
            ResidualBlock(self.num_hiddens, self.num_residual_hiddens, self.num_residual_layers))
        self.enc_d = nn.Sequential(
            nn.Conv2d(self.num_hiddens, self.num_hiddens//2, 1),
            nn.Conv2d(self.num_hiddens//2, self.num_hiddens//2, 4, stride=2, padding=1),
            nn.BatchNorm2d(self.num_hiddens//2),
            nn.ReLU(),
            nn.Conv2d(self.num_hiddens//2, self.num_hiddens//2, 4, stride=2, padding=1),
            nn.BatchNorm2d(self.num_hiddens//2),
            nn.ReLU(),
            nn.Conv2d(self.num_hiddens//2, self.num_hiddens//2, 4, stride=2, padding=1),
            nn.BatchNorm2d(self.num_hiddens//2),
            nn.ReLU(),
            Flatten(),
            nn.Linear(self.num_hiddens * 2, self.num_hiddens * 8),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(self.num_hiddens * 8, self.num_hiddens),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(self.num_hiddens, 1),
            nn.Sigmoid())
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(self.num_hiddens, self.num_hiddens//2, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.num_hiddens//2, self.num_hiddens//4, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(self.num_hiddens//4, self.num_hiddens//4, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(self.num_hiddens//4, self.num_inputs, 1))
    
    def forward(self, inputs, time_matching_mat=None, batch_mask=None):
        """ Forward pass

        Args:
            inputs (torch tensor): input cell image patches
            time_matching_mat (torch tensor or None, optional): if given, 
                pairwise relationship between samples in the minibatch, used 
                to calculate time matching loss
            batch_mask (torch tensor or None, optional): if given, weight mask 
                of training samples, used to concentrate loss on cell bodies

        Returns:
            None: placeholder
            dict: losses and perplexity of the minibatch

        """
        # inputs: Batch * num_inputs(channel) * H * W, each channel from 0 to 1
        z = self.enc(inputs)
        decoded = self.dec(z)
        if batch_mask is None:
            batch_mask = t.ones_like(inputs)
        recon_loss = t.mean(F.mse_loss(decoded * batch_mask, inputs * batch_mask, reduction='none')/self.channel_var)
        total_loss = self.weight_recon * recon_loss
        time_matching_loss = 0.
        if not time_matching_mat is None:
            z_ = z.reshape((z.shape[0], -1))
            len_latent = z_.shape[1]
            sim_mat = t.pow(z_.reshape((1, -1, len_latent)) - \
                            z_.reshape((-1, 1, len_latent)), 2).mean(2)
            assert sim_mat.shape == time_matching_mat.shape
            time_matching_loss = (sim_mat * time_matching_mat).sum()
            total_loss += self.weight_matching * time_matching_loss
        return decoded, \
               {'recon_loss': recon_loss,
                'time_matching_loss': time_matching_loss,
                'total_loss': total_loss,
                'perplexity': t.zeros(())}

    def adversarial_loss(self, inputs):
        """ Calculate adversarial loss for the batch

        Args:
            inputs (torch tensor): input cell image patches

        Returns:
            dict: generator/discriminator losses

        """
        # inputs: Batch * num_inputs(channel) * H * W, each channel from 0 to 1
        z_data = self.enc(inputs)
        z_prior = t.randn_like(z_data)
        _z_data = self.enc_d(z_data)
        _z_prior = self.enc_d(z_prior)
        g_loss = -t.mean(t.log(_z_data + eps))
        d_loss = -t.mean(t.log(_z_prior + eps) + t.log(1 - _z_data.detach() + eps))
        return {'generator_loss': g_loss,
                'descriminator_loss': d_loss,
                'score': t.mean(_z_data)}

    def predict(self, inputs):
        """ Prediction fn, same as forward pass """
        return self.forward(inputs)


if __name__ == '__main__':
    pass
    # ### Settings ###
    # cs = [0, 1]
    # cs_mask = [2, 3]
    # input_shape = (128, 128)
    # gpu = True
    # path = '/mnt/comp_micro/Projects/CellVAE'

    # ### Load Data ###
    # fs = read_file_path(path + '/Data/StaticPatches')

    # dataset = prepare_dataset(fs, cs=cs, input_shape=input_shape, channel_max=CHANNEL_MAX)
    # dataset_mask = prepare_dataset(fs, cs=cs_mask, input_shape=input_shape, channel_max=[1., 1.])
    
    # # Note that `relations` is depending on the order of fs (should not sort)
    # # `relations` is generated by script "generate_trajectory_relations.py"
    # relations = pickle.load(open(path + '/Data/StaticPatchesAllRelations.pkl', 'rb'))
    # dataset, relation_mat, inds_in_order = reorder_with_trajectories(dataset, relations, seed=123)
    # dataset_mask = TensorDataset(dataset_mask.tensors[0][np.array(inds_in_order)])
    # dataset = rescale(dataset)
    
    # ### Initialize Model ###
    # model = VQ_VAE(alpha=0.0005)
    # if gpu:
    #     model = model.cuda()
    # model = train(model, 
    #               dataset, 
    #               relation_mat=relation_mat, 
    #               mask=dataset_mask,
    #               n_epochs=500, 
    #               lr=0.0001, 
    #               batch_size=128, 
    #               gpu=gpu)
    # t.save(model.state_dict(), 'temp.pt')
    
    # ### Check coverage of embedding vectors ###
    # used_indices = []
    # for i in range(500):
    #     sample = dataset[i:(i+1)][0].cuda()
    #     z_before = model.enc(sample)
    #     indices = model.vq.encode_inputs(z_before)
    #     used_indices.append(np.unique(indices.cpu().data.numpy()))
    # print(np.unique(np.concatenate(used_indices)))

    # ### Generate latent vectors ###
    # z_bs = {}
    # z_as = {}
    # for i in range(len(dataset)):
    #     sample = dataset[i:(i+1)][0].cuda()
    #     z_b = model.enc(sample)
    #     z_a, _, _ = model.vq(z_b)

    #     f_n = fs[inds_in_order[i]]
    #     z_as[f_n] = z_a.cpu().data.numpy()
    #     z_bs[f_n] = z_b.cpu().data.numpy()
      
    
    # ### Visualize reconstruction ###
    # def enhance(mat, lower_thr, upper_thr):
    #     mat = np.clip(mat, lower_thr, upper_thr)
    #     mat = (mat - lower_thr)/(upper_thr - lower_thr)
    #     return mat

    # random_inds = np.random.randint(0, len(dataset), (10,))
    # for i in random_inds:
    #     sample = dataset[i:(i+1)][0].cuda()
    #     cv2.imwrite('sample%d_0.png' % i, 
    #         enhance(sample[0, 0].cpu().data.numpy(), 0., 1.)*255)
    #     cv2.imwrite('sample%d_1.png' % i, 
    #         enhance(sample[0, 1].cpu().data.numpy(), 0., 1.)*255)
    #     output = model(sample)[0]
    #     cv2.imwrite('sample%d_0_rebuilt.png' % i, 
    #         enhance(output[0, 0].cpu().data.numpy(), 0., 1.)*255)
    #     cv2.imwrite('sample%d_1_rebuilt.png' % i, 
    #         enhance(output[0, 1].cpu().data.numpy(), 0., 1.)*255)