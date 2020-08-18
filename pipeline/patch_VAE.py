# bchhun, {2020-02-21}

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import pickle
import torch
import numpy as np
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset

from SingleCellPatch.generate_trajectories import process_site_build_trajectory
from SingleCellPatch.extract_patches import process_site_extract_patches

from HiddenStateExtractor.vq_vae import VQ_VAE, prepare_dataset_v2, rescale


# 6
def extract_patches(paths):

    temp_folder, supp_folder, target, sites = paths[0], paths[1], paths[2], paths[3]

    for site in sites:
        site_path = os.path.join(temp_folder + '/' + site + '.npy')

        site_segmentation_path = os.path.join(temp_folder, '%s_NNProbabilities.npy' % site)
        site_supp_files_folder = os.path.join(supp_folder, '%s-supps' % site[:2], '%s' % site)
        if not os.path.exists(site_path) or \
          not os.path.exists(site_segmentation_path) or \
          not os.path.exists(site_supp_files_folder):
            print("Site not found %s" % site_path, flush=True)
        else:
            print("Building patches %s" % site_path, flush=True)

        process_site_extract_patches(site_path, site_segmentation_path, site_supp_files_folder,
                                     window_size=256)
    return

# 7
def build_trajectories(paths):

    temp_folder, supp_folder, target, sites = paths[0], paths[1], paths[2], paths[3]

    for site in sites:
        site_path = os.path.join(temp_folder + '/' + site + '.npy')

        site_segmentation_path = os.path.join(temp_folder, '%s_NNProbabilities.npy' % site)
        site_supp_files_folder = os.path.join(supp_folder, '%s-supps' % site[:2], '%s' % site)
        if not os.path.exists(site_path) or \
          not os.path.exists(site_segmentation_path) or \
          not os.path.exists(site_supp_files_folder):
            print("Site not found %s" % site_path, flush=True)
        else:
            print("Building trajectories %s" % site_path, flush=True)

        process_site_build_trajectory(site_path, site_segmentation_path, site_supp_files_folder)
    return

#8
def assemble_VAE(paths):

    # these sites should be from a single condition (C5, C4, B-wells, etc..)
    temp_folder, supp_folder, target, sites = paths[0], paths[1], paths[2], paths[3]

    dat_fs = []

    # Prepare dataset for VAE
    dat_fs = []
    for site in sites:
        supp_files_folder = os.path.join(supp_folder, '%s-supps' % site[:2], '%s' % site)
        dat_fs.extend([os.path.join(supp_files_folder, f) for f in os.listdir(supp_files_folder) if f.startswith('stacks')])

    dataset, fs = prepare_dataset_v2(dat_fs, cs=[0, 1])
    assert fs == sorted(fs)
    
    with open(os.path.join(temp_folder, '%s_file_paths.pkl' % sites[0][:2]), 'wb') as f:
        pickle.dump(fs, f)

    torch.save(dataset, os.path.join(temp_folder, '%s_static_patches.pt' % sites[0][:2]))

    # Adjust channel mean/std
    # phase: 0.4980 plus/minus 0.0257
    # retardance: 0.0285 plus/minus 0.0261, only adjust for mean
    phase_slice = dataset.tensors[0][:, 0]
    phase_slice = ((phase_slice - phase_slice.mean()) / phase_slice.std()) * 0.0257 + 0.4980
    retard_slice = dataset.tensors[0][:, 1]
    retard_slice = retard_slice / retard_slice.mean() * 0.0285
    adjusted_dataset = TensorDataset(torch.stack([phase_slice, retard_slice], 1))
    torch.save(adjusted_dataset, os.path.join(temp_folder, '%s_adjusted_static_patches.pt' % sites[0][:2]))
    return

#9
def process_VAE(paths):

    # these sites should be from a single condition (C5, C4, B-wells, etc..)
    temp_folder, supp_folder, target, sites = paths[0], paths[1], paths[2], paths[3]

    assert len(set(s[:2] for s in sites)) == 1
    well = sites[0][:2]
    fs = pickle.load(open(os.path.join(temp_folder, '%s_file_paths.pkl' % well), 'rb'))
    dataset = torch.load(os.path.join(temp_folder, '%s_adjusted_static_patches.pt' % well))
    dataset = rescale(dataset)
    
    model = VQ_VAE(alpha=0.0005, gpu=True)
    model = model.cuda()
    model.load_state_dict(torch.load('HiddenStateExtractor/save_0005_bkp4.pt'))
 
    z_bs = {}
    z_as = {}
    for i in range(len(dataset)):
      sample = dataset[i:(i+1)][0].cuda()
      z_b = model.enc(sample)
      z_a, _, _ = model.vq(z_b)
      f_n = fs[i]
      z_bs[f_n] = z_b.cpu().data.numpy()
      z_as[f_n] = z_a.cpu().data.numpy()      
    
    pca = pickle.load(open('HiddenStateExtractor/pca_save.pkl', 'rb'))

    dats = np.stack([z_bs[f] for f in fs], 0).reshape((len(dataset), -1))
    with open(os.path.join(temp_folder, '%s_latent_space.pkl' % well), 'wb') as f:
      pickle.dump(dats, f)
    dats_ = pca.transform(dats)
    with open(os.path.join(temp_folder, '%s_latent_space_PCAed.pkl' % well), 'wb') as f:
      pickle.dump(dats_, f)

    
    dats = np.stack([z_as[f] for f in fs], 0).reshape((len(dataset), -1))
    with open(os.path.join(temp_folder, '%s_latent_space_after.pkl' % well), 'wb') as f:
      pickle.dump(dats, f)
    dats_ = pca.transform(dats)
    with open(os.path.join(temp_folder, '%s_latent_space_after_PCAed.pkl' % well), 'wb') as f:
      pickle.dump(dats_, f)
    
    return

#10
def trajectory_matching(paths):

  temp_folder, supp_folder, target, sites = paths[0], paths[1], paths[2], paths[3]
  
  wells = set(site[:2] for site in sites)
  assert len(wells) == 1
  well = list(wells)[0]
    
  fs = pickle.load(open(os.path.join(temp_folder, '%s_file_paths.pkl' % well), 'rb'))

  site_trajs = {}
  for site in sites:
    site_supp_files_folder = os.path.join(supp_folder, '%s-supps' % well, '%s' % site)
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
        
  with open(os.path.join(temp_folder, '%s_trajectories.pkl' % well), 'wb') as f:
    pickle.dump(site_trajs, f)