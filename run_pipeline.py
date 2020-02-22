import tensorflow as tf
import numpy as np
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import pickle
import torch
from torch.utils.data import TensorDataset
from NNsegmentation.data import predict_whole_map
from NNsegmentation.models import Segment
from SingleCellPatch.extract_patches import process_site_instance_segmentation, process_site_extract_patches
from SingleCellPatch.generate_trajectories import process_site_build_trajectory
from HiddenStateExtractor.vq_vae import VQ_VAE, prepare_dataset_v2, rescale
import cv2

# pipeline:
# 1. check input: (n_frames * 2048 * 2048 * 2) channel 0 - phase, channel 1 - retardance
# 2. adjust channel range
#     a. phase: 32767 plus/minus 1600~2000
#     b. retardance: 1400~1600 plus/minus 1500~1800
# 3. save as '$SITE_NAME.npy' numpy array, dtype=uint16
# 4. run segmentation using saved model: `/data/michaelwu/CellVAE/NNSegmentation/temp_save_unsaturated/final.h5`
# 5. run instance segmentation
# 6. save individual cell patches
# 7. connect individual cells into trajectories
# 8. collect patches and assemble for VAE encoding
# 9. PCA of VAE encoded latent vectors

sites = 'C5'
DATA_ROOT = '/data/michaelwu/data_temp'
for i in [0, 1, 2, 3, 5, 6, 7]:
  site_name = 'C5-Site_%d' % i
  site_path = os.path.join(DATA_ROOT, '%s.npy' % site_name)
  site_segmentation_path = os.path.join(DATA_ROOT, '%s_NNProbabilities.npy' % site_name)
  site_supp_files_folder = os.path.join(DATA_ROOT, '%s-supps' % sites, '%s' % site_name)
  if not os.path.exists(site_supp_files_folder):
    os.mkdir(site_supp_files_folder)

  # Generate semantic segmentation  
  model = Segment(input_shape=(256, 256, 2), 
                  unet_feat=32,
                  fc_layers=[64, 32],
                  n_classes=3)
  model.load('NNsegmentation/temp_save_unsaturated/final.h5')
  predict_whole_map(site_path, model, n_classes=3, batch_size=8, n_supp=5)

  # Extract single cell patches
  process_site_instance_segmentation(site_path, site_segmentation_path, site_supp_files_folder)
  process_site_extract_patches(site_path, site_segmentation_path, site_supp_files_folder, window_size=256)

  # Build trajectories
  process_site_build_trajectory(site_path, site_segmentation_path, site_supp_files_folder)

# Prepare dataset for VAE
dat_fs = []
for s in ['C5-Site_%d' % i for i in [0, 1, 2, 3, 5, 6, 7]]:
  supp_files_folder = os.path.join(DATA_ROOT, '%s-supps' % sites, '%s' % s)
  dat_fs.extend([os.path.join(supp_files_folder, f) for f in os.listdir(supp_files_folder) if f.startswith('stacks')])
dataset, fs = prepare_dataset_v2(dat_fs, cs=[0, 1])
with open(os.path.join(data_ROOT, '%s_file_paths.pkl' % sites), 'wb') as f:
  pickle.dump(fs, f)
torch.save(dataset, os.path.join(DATA_ROOT, '%s_all_static_patches.pt' % sites))

# Adjust channel mean/std
# phase: 0.4980 plus/minus 0.0257
# retardance: 0.0285 plus/minus 0.0261, only adjust for mean
phase_slice = dataset.tensors[0][:, 0]
phase_slice = ((phase_slice - phase_slice.mean())/phase_slice.std()) * 0.0257 + 0.4980
retard_slice = dataset.tensors[0][:, 1]
retard_slice = retard_slice/retard_slice.mean() * 0.0285
adjusted_dataset = TensorDataset(torch.stack([phase_slice, retard_slice], 1))
torch.save(adjusted_dataset, os.path.join(DATA_ROOT, '%s_all_adjusted_static_patches.pt' % sites))

# VAE encoding
input_dataset = rescale(adjusted_dataset)
model = VQ_VAE(alpha=0.0005, gpu=True)
model.load_state_dict(torch.load('HiddenStateExtractor/save_0005_bkp4.pt'))
z_bs = {}
for i in range(len(dataset)):
  sample = input_dataset[i:(i+1)][0].cuda()
  z_b = model.enc(sample)
  f_n = fs[i]
  z_bs[f_n] = z_b.cpu().data.numpy()
dats = np.stack([z_bs[f] for f in fs], 0).reshape((len(dataset), -1))
with open(os.path.join(DATA_ROOT, '%s_latent_space.pkl' % sites), 'wb') as f:
  pickle.dump(dats, f)

# PCA
pca = pickle.load(open('HiddenStateExtractor/pca_save.pkl', 'rb'))
dats_ = pca.transform(dats)
with open(os.path.join(DATA_ROOT, '%s_latent_space_PCAed.pkl' % sites), 'wb') as f:
  pickle.dump(dats_, f)

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
    if len(traj) > 0.95*len(t):
      site_trajs[name] = traj
with open(os.path.join(DATA_ROOT, '%s_trajectories.pkl' % sites), 'wb') as f:
  pickle.dump(dats_, f)
