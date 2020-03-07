# bchhun, {2020-02-21}

import numpy as np
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import pickle
import torch

from HiddenStateExtractor.vq_vae import VQ_VAE, prepare_dataset_v2, rescale



def vae_encoding():
  # VAE encoding
  input_dataset = rescale(adjusted_dataset)
  model = VQ_VAE(alpha=0.0005, gpu=True)
  model.load_state_dict(torch.load('HiddenStateExtractor/save_0005_bkp4.pt'))
  z_bs = {}
  for i in range(len(dataset)):
    sample = input_dataset[i:(i + 1)][0].cuda()
    z_b = model.enc(sample)
    f_n = fs[i]
    z_bs[f_n] = z_b.cpu().data.numpy()
  dats = np.stack([z_bs[f] for f in fs], 0).reshape((len(dataset), -1))
  with open(os.path.join(DATA_ROOT, '%s_latent_space.pkl' % sites), 'wb') as f:
    pickle.dump(dats, f)


def pca():
  # PCA
  pca = pickle.load(open('HiddenStateExtractor/pca_save.pkl', 'rb'))
  dats_ = pca.transform(dats)
  with open(os.path.join(DATA_ROOT, '%s_latent_space_PCAed.pkl' % sites), 'wb') as f:
    pickle.dump(dats_, f)


def trajectory_matching():
  # trajectory matching
  site_trajs = {}
  for site_i in [0, 1, 2, 3, 5, 6, 7]:
    site_name = 'C5-Site_%d' % site_i
    site_supp_files_folder = os.path.join(DATA_ROOT, '%s-supps' % sites, '%s' % site_name)
    trajs = pickle.load(open(os.path.join(site_supp_files_folder, 'cell_traj.pkl'), 'rb'))
    for i, t in enumerate(trajs[0]):
      name = site_name + '/' + str(i)
      traj = []
      for t_point in sorted(t.keys()):
        frame_name = '/data/michaelwu/data_temp/C5-supps/%s/%d_%d.h5' % (site_name, t_point, t[t_point])
        if frame_name in fs:
          traj.append(fs.index(frame_name))
      if len(traj) > 0.95 * len(t):
        site_trajs[name] = traj
  with open(os.path.join(DATA_ROOT, '%s_trajectories.pkl' % sites), 'wb') as f:
    pickle.dump(dats_, f)
