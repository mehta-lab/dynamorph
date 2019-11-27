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
from .naive_imagenet import DATA_ROOT, read_file_path

CHANNEL_RANGE = [(0.3, 0.8), (0., 0.6)] 
CHANNEL_VAR = np.array([0.0475, 0.0394]) # After normalized to CHANNEL_RANGE
CHANNEL_MAX = np.array([65535., 65535.])

class VectorQuantizer(nn.Module):
  def __init__(self, embedding_dim=128, num_embeddings=128, commitment_cost=0.25, gpu=True):
    super(VectorQuantizer, self).__init__()
    self.embedding_dim = embedding_dim
    self.num_embeddings = num_embeddings
    self.commitment_cost = commitment_cost
    self.gpu = gpu
    self.w = nn.Embedding(num_embeddings, embedding_dim)

  def forward(self, inputs):

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
    # TODO: better deal with the gpu case here
    encoding_onehot = t.zeros(encoding_indices.flatten().shape[0], self.num_embeddings)
    if self.gpu:
      encoding_onehot = encoding_onehot.cuda()
    encoding_onehot.scatter_(1, encoding_indices.flatten().unsqueeze(1), 1)
    avg_probs = t.mean(encoding_onehot, 0)
    perplexity = t.exp(-t.sum(avg_probs*t.log(avg_probs + 1e-10)))

    return output_quantized, loss, perplexity

  @property
  def embeddings(self):
    return self.w.weight

  def encode_inputs(self, inputs):
    # inputs: Batch * Num_hidden(=embedding_dim) * H * W
    distances = t.sum((inputs.unsqueeze(1) - self.w.weight.reshape((1, self.num_embeddings, self.embedding_dim, 1, 1)))**2, 2)
    # Decoder input
    encoding_indices = t.argmax(-distances, 1)
    return encoding_indices

class Reparametrize(nn.Module):
  def forward(self, z_mean, z_logstd):
    z_std = t.exp(0.5 * z_logstd)
    eps = t.randn_like(z_std)
    z = z_mean + z_std * eps    
    KLD = -0.5 * t.sum(1 + z_logstd - z_mean.pow(2) - z_logstd.exp())
    return z, KLD

class ResidualBlock(nn.Module):
  def __init__(self,
               num_hiddens=128,
               num_residual_hiddens=512,
               num_residual_layers=2):
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
    output = x
    for i in range(self.num_residual_layers):
      output = output + self.layers[i](output)
    return output

# 32 * 32 * 128, strong decoder
#class VQ_VAE(nn.Module):
#  def __init__(self,
#               num_inputs=3,
#               num_hiddens=128,
#               num_residual_hiddens=64,
#               num_residual_layers=2,
#               num_embeddings=128,
#               commitment_cost=0.25,
#               channel_var=CHANNEL_VAR,
#               alpha=0.1,
#               **kwargs):
#    super(VQ_VAE, self).__init__(**kwargs)
#    self.num_inputs = num_inputs
#    self.num_hiddens = num_hiddens
#    self.num_residual_layers = num_residual_layers
#    self.num_residual_hiddens = num_residual_hiddens
#    self.num_embeddings = num_embeddings
#    self.commitment_cost = commitment_cost
#    self.channel_var = nn.Parameter(t.from_numpy(channel_var).float().reshape((1, 3, 1, 1)), requires_grad=False)
#    self.alpha = alpha
#    self.enc = nn.Sequential(
#        nn.Conv2d(self.num_inputs, self.num_hiddens//2, 1),
#        nn.Conv2d(self.num_hiddens//2, self.num_hiddens//2, 4, stride=2, padding=1),
#        nn.BatchNorm2d(self.num_hiddens//2),
#        nn.ReLU(),
#        nn.Conv2d(self.num_hiddens//2, self.num_hiddens, 4, stride=2, padding=1),
#        nn.BatchNorm2d(self.num_hiddens),
#        nn.ReLU(),
#        nn.Conv2d(self.num_hiddens, self.num_hiddens, 3, padding=1),
#        nn.BatchNorm2d(self.num_hiddens),
#        ResidualBlock(self.num_hiddens, self.num_residual_hiddens, self.num_residual_layers))
#    self.vq = VectorQuantizer(self.num_hiddens, self.num_embeddings, commitment_cost=self.commitment_cost)
#    self.dec = nn.Sequential(
#        nn.Conv2d(self.num_hiddens, self.num_hiddens, 3, padding=1),
#        ResidualBlock(self.num_hiddens, self.num_residual_hiddens, self.num_residual_layers),
#        nn.ConvTranspose2d(self.num_hiddens, self.num_hiddens//2, 4, stride=2, padding=1),
#        nn.BatchNorm2d(self.num_hiddens//2),
#        nn.ReLU(),
#        nn.ConvTranspose2d(self.num_hiddens//2, self.num_hiddens//4, 4, stride=2, padding=1),
#        nn.BatchNorm2d(self.num_hiddens//4),
#        nn.ReLU(),
#        nn.Conv2d(self.num_hiddens//4, self.num_inputs, 1))

# 16*16*16, strong decoder
#class VQ_VAE(nn.Module):
#  def __init__(self,
#               num_inputs=3,
#               num_hiddens=16,
#               num_residual_hiddens=64,
#               num_residual_layers=2,
#               num_embeddings=64,
#               commitment_cost=0.25,
#               channel_var=CHANNEL_VAR,
#               alpha=0.1,
#               **kwargs):
#    super(VQ_VAE, self).__init__(**kwargs)
#    self.num_inputs = num_inputs
#    self.num_hiddens = num_hiddens
#    self.num_residual_layers = num_residual_layers
#    self.num_residual_hiddens = num_residual_hiddens
#    self.num_embeddings = num_embeddings
#    self.commitment_cost = commitment_cost
#    self.channel_var = nn.Parameter(t.from_numpy(channel_var).float().reshape((1, 3, 1, 1)), requires_grad=False)
#    self.alpha = alpha
#    self.enc = nn.Sequential(
#        nn.Conv2d(self.num_inputs, self.num_hiddens//2, 1),
#        nn.Conv2d(self.num_hiddens//2, self.num_hiddens//2, 4, stride=2, padding=1),
#        nn.BatchNorm2d(self.num_hiddens//2),
#        nn.ReLU(),
#        nn.Conv2d(self.num_hiddens//2, self.num_hiddens, 4, stride=2, padding=1),
#        nn.BatchNorm2d(self.num_hiddens),
#        nn.ReLU(),
#        nn.Conv2d(self.num_hiddens, self.num_hiddens, 4, stride=2, padding=1),
#        nn.BatchNorm2d(self.num_hiddens),
#        nn.ReLU(),
#        nn.Conv2d(self.num_hiddens, self.num_hiddens, 3, padding=1),
#        nn.BatchNorm2d(self.num_hiddens),
#        ResidualBlock(self.num_hiddens, self.num_residual_hiddens, self.num_residual_layers))
#    self.vq = VectorQuantizer(self.num_hiddens, self.num_embeddings, commitment_cost=self.commitment_cost)
#    self.dec = nn.Sequential(
#        nn.Conv2d(self.num_hiddens, self.num_hiddens, 3, padding=1),
#        ResidualBlock(self.num_hiddens, self.num_residual_hiddens, self.num_residual_layers),
#        nn.ConvTranspose2d(self.num_hiddens, self.num_hiddens//2, 4, stride=2, padding=1),
#        nn.BatchNorm2d(self.num_hiddens//2),
#        nn.ReLU(),
#        nn.ConvTranspose2d(self.num_hiddens//2, self.num_hiddens//4, 4, stride=2, padding=1),
#        nn.BatchNorm2d(self.num_hiddens//4),
#        nn.ReLU(),
#        nn.ConvTranspose2d(self.num_hiddens//4, self.num_hiddens//4, 4, stride=2, padding=1),
#        nn.BatchNorm2d(self.num_hiddens//4),
#        nn.ReLU(),
#        nn.Conv2d(self.num_hiddens//4, self.num_inputs, 1))

# 32*32*128, weak decoder
#class VQ_VAE(nn.Module):
#  def __init__(self,
#               num_inputs=3,
#               num_hiddens=128,
#               num_residual_hiddens=64,
#               num_residual_layers=2,
#               num_embeddings=32,
#               commitment_cost=0.25,
#               channel_var=CHANNEL_VAR,
#               alpha=0.1,
#               **kwargs):
#    super(VQ_VAE, self).__init__(**kwargs)
#    self.num_inputs = num_inputs
#    self.num_hiddens = num_hiddens
#    self.num_residual_layers = num_residual_layers
#    self.num_residual_hiddens = num_residual_hiddens
#    self.num_embeddings = num_embeddings
#    self.commitment_cost = commitment_cost
#    self.channel_var = nn.Parameter(t.from_numpy(channel_var).float().reshape((1, 3, 1, 1)), requires_grad=False)
#    self.alpha = alpha
#    self.enc = nn.Sequential(
#        nn.Conv2d(self.num_inputs, self.num_hiddens//2, 1),
#        nn.Conv2d(self.num_hiddens//2, self.num_hiddens//2, 4, stride=2, padding=1),
#        nn.BatchNorm2d(self.num_hiddens//2),
#        nn.ReLU(),
#        nn.Conv2d(self.num_hiddens//2, self.num_hiddens, 4, stride=2, padding=1),
#        nn.BatchNorm2d(self.num_hiddens),
#        nn.ReLU(),
#        nn.Conv2d(self.num_hiddens, self.num_hiddens, 3, padding=1),
#        nn.BatchNorm2d(self.num_hiddens),
#        ResidualBlock(self.num_hiddens, self.num_residual_hiddens, self.num_residual_layers))
#    self.vq = VectorQuantizer(self.num_hiddens, self.num_embeddings, commitment_cost=self.commitment_cost)
#    self.dec = nn.Sequential(
#        nn.ConvTranspose2d(self.num_hiddens, self.num_hiddens//2, 4, stride=2, padding=1),
#        nn.ReLU(),
#        nn.ConvTranspose2d(self.num_hiddens//2, self.num_hiddens//4, 4, stride=2, padding=1),
#        nn.ReLU(),
#        nn.Conv2d(self.num_hiddens//4, self.num_inputs, 1))

# 16*16*16, weak decoder
class VQ_VAE(nn.Module):
  def __init__(self,
               num_inputs=2,
               num_hiddens=16,
               num_residual_hiddens=32,
               num_residual_layers=2,
               num_embeddings=64,
               commitment_cost=0.25,
               channel_var=CHANNEL_VAR,
               alpha=0.005,
               gpu=True,
               **kwargs):
    super(VQ_VAE, self).__init__(**kwargs)
    self.num_inputs = num_inputs
    self.num_hiddens = num_hiddens
    self.num_residual_layers = num_residual_layers
    self.num_residual_hiddens = num_residual_hiddens
    self.num_embeddings = num_embeddings
    self.commitment_cost = commitment_cost
    self.channel_var = nn.Parameter(t.from_numpy(channel_var).float().reshape((1, num_inputs, 1, 1)), requires_grad=False)
    self.alpha = alpha
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
    self.vq = VectorQuantizer(self.num_hiddens, self.num_embeddings, commitment_cost=self.commitment_cost, gpu=gpu)
    self.dec = nn.Sequential(
        nn.ConvTranspose2d(self.num_hiddens, self.num_hiddens//2, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(self.num_hiddens//2, self.num_hiddens//4, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.ConvTranspose2d(self.num_hiddens//4, self.num_hiddens//4, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(self.num_hiddens//4, self.num_inputs, 1))
    
  def forward(self, inputs, time_matching_mat=None, batch_mask=None):
    # inputs: Batch * num_inputs(channel) * H * W, each channel from 0 to 1
    z_before = self.enc(inputs)
    z_after, c_loss, perplexity = self.vq(z_before)
    decoded = self.dec(z_after)
    if batch_mask is None:
      batch_mask = t.ones_like(inputs)
    recon_loss = t.mean(F.mse_loss(decoded * batch_mask, inputs * batch_mask, reduction='none')/self.channel_var)
    total_loss = recon_loss + c_loss
    time_matching_loss = None
    if not time_matching_mat is None:
      z_before_ = z_before.reshape((z_before.shape[0], -1))
      len_latent = z_before_.shape[1]
      sim_mat = t.pow(z_before_.reshape((1, -1, len_latent)) - \
                      z_before_.reshape((-1, 1, len_latent)), 2).mean(2)
      assert sim_mat.shape == time_matching_mat.shape
      time_matching_loss = (sim_mat * time_matching_mat).sum()
      total_loss += time_matching_loss * self.alpha
    return decoded, \
           {'recon_loss': recon_loss,
            'commitment_loss': c_loss,
            'time_matching_loss': time_matching_loss,
            'total_loss': total_loss,
            'perplexity': perplexity}

#class VAE(nn.Module):
#  def __init__(self,
#               num_inputs=3,
#               num_hiddens=16,
#               num_residual_hiddens=32,
#               num_residual_layers=2,
#               commitment_cost=0.25,
#               channel_var=CHANNEL_VAR,
#               **kwargs):
#    super(VAE, self).__init__(**kwargs)
#    self.num_inputs = num_inputs
#    self.num_hiddens = num_hiddens
#    self.num_residual_layers = num_residual_layers
#    self.num_residual_hiddens = num_residual_hiddens
#    self.commitment_cost = commitment_cost
#    self.channel_var = nn.Parameter(t.from_numpy(channel_var).float().reshape((1, 3, 1, 1)), requires_grad=False)
#    self.enc = nn.Sequential(
#        nn.Conv2d(self.num_inputs, self.num_hiddens//2, 1),
#        nn.Conv2d(self.num_hiddens//2, self.num_hiddens//2, 4, stride=2, padding=1),
#        nn.BatchNorm2d(self.num_hiddens//2),
#        nn.ReLU(),
#        nn.Conv2d(self.num_hiddens//2, self.num_hiddens, 4, stride=2, padding=1),
#        nn.BatchNorm2d(self.num_hiddens),
#        nn.ReLU(),
#        nn.Conv2d(self.num_hiddens, self.num_hiddens, 4, stride=2, padding=1),
#          nn.BatchNorm2d(self.num_hiddens),
#        nn.ReLU(),
#        nn.Conv2d(self.num_hiddens, self.num_hiddens*2, 3, padding=1),
#        nn.BatchNorm2d(self.num_hiddens * 2),
#        ResidualBlock(self.num_hiddens * 2, self.num_residual_hiddens, self.num_residual_layers)) # For mean and logstd
#    self.rp = Reparametrize()
#    self.dec = nn.Sequential(
#        nn.ConvTranspose2d(self.num_hiddens, self.num_hiddens//2, 4, stride=2, padding=1),
#        nn.ReLU(),
#        nn.ConvTranspose2d(self.num_hiddens//2, self.num_hiddens//4, 4, stride=2, padding=1),
#        nn.ReLU(),
#        nn.ConvTranspose2d(self.num_hiddens//4, self.num_hiddens//4, 4, stride=2, padding=1),
#        nn.ReLU(),
#        nn.Conv2d(self.num_hiddens//4, self.num_inputs, 1))
#    
#  def forward(self, inputs, time_matching_mat=None, batch_mask=None):
#    # inputs: Batch * num_inputs(channel) * H * W, each channel from 0 to 1
#    z_before = self.enc(inputs)
#    z_mean = z_before[:, :self.num_hiddens]
#    z_logstd = z_before[:, self.num_hiddens:]
#    z_after, KLD = self.rp(z_mean, z_logstd)
#    decoded = self.dec(z_after)
#    if batch_mask is None:
#      batch_mask = t.ones_like(inputs)
#    recon_loss = t.mean(F.mse_loss(decoded * batch_mask, inputs * batch_mask, reduction='none')/self.channel_var)
#    total_loss = recon_loss + KLD
#    time_matching_loss = None
#    if not time_matching_mat is None:
#      z_before_ = z_before.reshape((z_before.shape[0], -1))
#      len_latent = z_before_.shape[1]
#      sim_mat = t.pow(z_before_.reshape((1, -1, len_latent)) - \
#                      z_before_.reshape((-1, 1, len_latent)), 2).sum(2)
#      assert sim_mat.shape == time_matching_mat.shape
#      time_matching_loss = (sim_mat * time_matching_mat).sum()
#      total_loss += time_matching_loss
#    return decoded, \
#           {'recon_loss': recon_loss,
#            'KL_loss': KLD,
#            'time_matching_loss': time_matching_loss,
#            'total_loss': total_loss,
#            'perplexity': t.zeros((1,))}

def train(model, dataset, relation_mat=None, mask=None, n_epochs=10, lr=0.001, batch_size=16, gpu=True):
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
      optimizer.step()
      model.zero_grad()

      recon_loss.append(loss_dict['recon_loss'])
      perplexities.append(loss_dict['perplexity'])
    print('epoch %d recon loss: %f perplexity: %f' % \
        (epoch, sum(recon_loss).item()/len(recon_loss), sum(perplexities).item()/len(perplexities)))
  return model

def prepare_dataset(fs, cs=[0, 1], input_shape=(128, 128), channel_max=CHANNEL_MAX):
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
  relation_mat = scipy.sparse.csr_matrix((np.array(values), (new_relations[:, 0], new_relations[:, 1])),
                                         shape=(len(dataset), len(dataset)))
  relation_mat = relation_mat[np.array(inds_in_order)][:, np.array(inds_in_order)]
  return TensorDataset(new_tensor), relation_mat, inds_in_order

def rescale(dataset):
  tensor = dataset.tensors[0]
  assert len(CHANNEL_RANGE) == tensor.shape[1]
  channel_slices = []
  for i in range(len(CHANNEL_RANGE)):
    lower_, upper_ = CHANNEL_RANGE[i]
    channel_slice = (tensor[:, i] - lower_) / (upper_ - lower_)
    channel_slice = t.clamp(channel_slice, 0, 1)
    channel_slices.append(channel_slice)
  new_tensor = t.stack(channel_slices, 1)
  return TensorDataset(new_tensor)

if __name__ == '__main__':
  ### Settings ###
  cs = [0, 1]
  cs_mask = [2, 3]
  input_shape = (128, 128)
  gpu = True
  path = '/mnt/comp_micro/Projects/CellVAE'

  ### Load Data ###
  #fs = read_file_path(DATA_ROOT + '/Data/StaticPatches')
  fs = pickle.load(open('./HiddenStateExtractor/file_paths_bkp.pkl', 'rb'))

  #dataset = prepare_dataset(fs, cs=cs, input_shape=input_shape, channel_max=CHANNEL_MAX)
  #dataset = prepare_dataset_from_collection(fs, cs=cs, input_shape=input_shape, channel_max=CHANNEL_MAX)
  dataset = t.load('StaticPatchesAll.pt')
  #dataset_mask = prepare_dataset(fs, cs=cs_mask, input_shape=input_shape, channel_max=[1., 1.])
  #dataset_mask = prepare_dataset_from_collection(fs, cs=cs_mask, input_shape=input_shape, channel_max=[1., 1.])
  dataset_mask = t.load('StaticPatchesAllMask.pt')

  relations = pickle.load(open(path + '/Data/StaticPatchesAllRelations.pkl', 'rb'))  
  dataset, relation_mat, inds_in_order = reorder_with_trajectories(dataset, relations, seed=123)
  dataset_mask = TensorDataset(dataset_mask.tensors[0][np.array(inds_in_order)])
  dataset = rescale(dataset)
  
  ### Initialize Model ###
  model = VQ_VAE(alpha=0.0005)
  if gpu:
    model = model.cuda()
  model = train(model, 
                dataset, 
                relation_mat=relation_mat, 
                mask=dataset_mask,
                n_epochs=500, 
                lr=0.0001, 
                batch_size=128, 
                gpu=gpu)
  t.save(model.state_dict(), 'save_small.pt')
  
  ### Check used prior vectors ###=bcdmoprst
  
  used_indices = []
  for i in range(500):
    sample = dataset[i:(i+1)][0].cuda()
    z_before = model.enc(sample)
    indices = model.vq.encode_inputs(z_before)
    used_indices.append(np.unique(indices.cpu().data.numpy()))
  print(np.unique(np.concatenate(used_indices)))
  ### Generate latent vectors ###
  z_bs = {}
  z_as = {}
  for i in range(len(dataset)):
    sample = dataset[i:(i+1)][0].cuda()
    z_b = model.enc(sample)
    z_a, _, _ = model.vq(z_b)

    f_n = fs[inds_in_order[i]]
    f_n = f_n.replace('../Data/StaticPatches/', '')
    z_as[f_n] = z_a.cpu().data.numpy()
    z_bs[f_n] = z_b.cpu().data.numpy()
    
  
  ### Visualize reconstruction ###

  def enhance(mat, lower_thr, upper_thr):
    mat = np.clip(mat, lower_thr, upper_thr)
    mat = (mat - lower_thr)/(upper_thr - lower_thr)
    return mat

  random_inds = np.random.randint(0, len(dataset), (10,))
  for i in random_inds:
    sample = dataset[i:(i+1)][0].cuda()
    cv2.imwrite('/home/michaelwu/sample%d_0.png' % i, enhance(sample[0, 0].cpu().data.numpy(), 0.4, 0.7)*255)
    cv2.imwrite('/home/michaelwu/sample%d_1.png' % i, enhance(sample[0, 1].cpu().data.numpy(), 0., 0.2)*255)
    output = model(sample)[0]
    cv2.imwrite('/home/michaelwu/sample%d_0_rebuilt.png' % i, enhance(output[0, 0].cpu().data.numpy(), 0.4, 0.7)*255)
    cv2.imwrite('/home/michaelwu/sample%d_1_rebuilt.png' % i, enhance(output[0, 1].cpu().data.numpy(), 0., 0.2)*255)