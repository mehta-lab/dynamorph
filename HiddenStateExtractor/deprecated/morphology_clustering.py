import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from .naive_imagenet import DATA_ROOT, read_file_path
import pickle
import cv2
import h5py
from matplotlib.patches import Rectangle
from matplotlib import cm
import imageio
#import tifffile

def generate_cell_sizes(fs, out_path=None):
  sizes = {}
  for i, f_n in enumerate(fs):
    if i % 1000 == 0:
      print("Processed %d" % i)
      if not out_path is None:
        with open(out_path, 'wb') as f_w:
          pickle.dump(sizes, f_w)
    with h5py.File(f_n, 'r') as f:
      size = f['masked_mat'][:, :, 2].sum()
      sizes[f_n] = size
  if not out_path is None:
    with open(out_path, 'wb') as f_w:
      pickle.dump(sizes, f_w)
  return sizes

def generate_cell_aspect_ratios(fs, out_path=None):
  aps = {}
  for i, f_n in enumerate(fs):
    if i % 1000 == 0:
      print("Processed %d" % i)
      if not out_path is None:
        with open(out_path, 'wb') as f_w:
          pickle.dump(aps, f_w)
    with h5py.File(f_n, 'r') as f:
      mask = np.array(f['masked_mat'][:, :, 2]).astype('uint8')

      cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2]
      cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
      rbox = cv2.minAreaRect(cnt)
      aps[f_n] = rbox[1][0]/rbox[1][1]
  if not out_path is None:
    with open(out_path, 'wb') as f_w:
      pickle.dump(aps, f_w)
  return sizes

def select_clean_trajecteories(dats_, trajs):
  clean_trajs = {}
  traj_diffs_dict = {}
  for t in trajs:
    traj_dats_ = dats_[np.array(trajs[t])]
    traj_diffs = np.linalg.norm(traj_dats_[1:] - traj_dats_[:-1], ord=2, axis=1)
    traj_diffs_dict[t] = traj_diffs
  thr = np.quantile(np.concatenate(list(traj_diffs_dict.values())), 0.9)
  for t in trajs:
    if np.quantile(traj_diffs_dict[t], 0.7) < thr:
      clean_trajs[t] = trajs[t]
  return clean_trajs
  
def read_trajectories(fs, out_path=None):
  latent_space_trajs = {}
  sites = ['D%d-Site_%d' % (i, j) for j in range(9) for i in range(3, 6)]
  for site in sites:
    trajectories = pickle.load(open(DATA_ROOT + '/Data/DynamicPatches/%s/mg_traj.pkl' % site, 'rb'))[0] # Select from [trajectories, trajectories_positions]
    for i, t in enumerate(trajectories):
      names = [DATA_ROOT + '/Data/StaticPatches/%s/%d_%d.h5' % (site, k, t[k]) for k in sorted(t.keys())]
      inds = [fs.index(name) for name in names if name in fs]
      latent_space_trajs[site + '/%d' % i] = inds
  if not out_path is None:
    with open(out_path, 'wb') as f:
      pickle.dump(latent_space_trajs, f)
  return latent_space_trajs

def step_displacement_histogram(vs, trajs):
  np.random.seed(123)
  traj_step_sizes = []
  random_traj_step_sizes = []
  for traj in trajs:
    traj_ = np.stack([vs[i] for i in traj], 0)
    step_sizes = np.linalg.norm(traj_[1:] - traj_[:-1], ord=2, axis=1)
    traj_step_sizes.append(step_sizes)
    
    random_traj = np.random.randint(0, len(vs), size=(len(traj),))
    random_traj_ = np.stack([vs[i] for i in random_traj], 0)
    random_step_sizes = np.linalg.norm(random_traj_[1:] - random_traj_[:-1], ord=2, axis=1)
    random_traj_step_sizes.append(random_step_sizes)
  
  traj_step_sizes = np.concatenate(traj_step_sizes)
  random_traj_step_sizes = np.concatenate(random_traj_step_sizes)
  
  traj_step_sizes = np.array(traj_step_sizes)/np.median(random_traj_step_sizes)
  random_traj_step_sizes = np.array(random_traj_step_sizes)/np.median(random_traj_step_sizes)
  plt.clf()
  plt.hist(random_traj_step_sizes, bins=np.arange(0, 2, 0.02), color=(0, 0, 1, 0.5), label='random')
  plt.hist(traj_step_sizes, bins=np.arange(0, 2, 0.02), color=(0, 1, 0, 0.5), label='trajectory, mean: %f' % np.mean(traj_step_sizes))
  plt.legend()

def generate_short_traj_morphorlogy(vs, traj_list, length=5):
  short_trajs = []
  for t in traj_list:
    n_sub_trajs = len(t) - (length - 1)
    for i in range(n_sub_trajs):
      sub_traj = t[i:(i+length)]
      
      sub_v = vs[np.array(sub_traj)]
      short_trajs.append(sub_v)
  short_trajs = np.stack(short_trajs, 0).reshape((len(short_trajs), -1))
  return short_trajs

def Kmean_on_short_trajs(vs, trajs, length=5, n_clusters=4):
  short_trajs = generate_short_traj_morphorlogy(vs, list(trajs.values()), length=length)
  short_trajs = short_trajs.reshape((len(short_trajs), -1))
  
  clustering = KMeans(n_clusters=n_clusters)
  clustering.fit(short_trajs)
  predicted_classes = {}
  for t in trajs:
    sub_trajs = generate_short_traj_morphorlogy(vs, [trajs[t]], length=length)
    sub_trajs = sub_trajs.reshape((len(sub_trajs), -1))
    labels = clustering.predict(sub_trajs)
    predicted_classes[t] = labels
  return predicted_classes

def Kmean_on_short_traj_diffs(vs, trajs, length=5, n_clusters=4):
  short_trajs = generate_short_traj_morphorlogy(vs, list(trajs.values()), length=length)
  short_traj_diffs = (short_trajs[:, 1:] - short_trajs[:, :-1]).reshape((len(short_trajs), -1))
  
  clustering = KMeans(n_clusters=n_clusters)
  clustering.fit(short_traj_diffs)
  predicted_classes = {}
  for t in trajs:
    sub_trajs = generate_short_traj_morphorlogy(vs, [trajs[t]], length=length)
    sub_traj_diffs = (sub_trajs[:, 1:] - sub_trajs[:, :-1]).reshape((len(sub_trajs), -1))
    labels = clustering.predict(sub_traj_diffs)
    predicted_classes[t] = labels
  return predicted_classes

def save_traj(k, output_path=None):
  input_path = DATA_ROOT + '/Data/DynamicPatches/%s/mg_traj_%s.tif' % (k.split('/')[0], k.split('/')[1])
  # images = tifffile.imread(input_path)
  _, images = cv2.imreadmulti(input_path, flags=cv2.IMREAD_ANYDEPTH)
  images = np.array(images)
  if output_path is None:
    output_path = './%s.gif' % (t, k[:9] + '_' + k[10:])
  imageio.mimsave(output_path, images)
  return


if __name__ == '__main__':

  feat = 'save_0005_before'
  sites = ['D%d-Site_%d' % (i, j) for j in range(9) for i in range(3, 6)]
  fs = sorted(read_file_path(DATA_ROOT + '/Data/StaticPatches'))
  
  # TRAJECTORIES  
  #trajs = read_trajectories(fs, './trajectory_in_inds.pkl')
  trajs = pickle.load(open('./trajectory_in_inds.pkl', 'rb'))

  # IMAGE REPRESENTATIONS
  dats = pickle.load(open(DATA_ROOT + '/Data/%s.pkl' % feat, 'rb'))
  ks = sorted([k for k in dats.keys() if dats[k] is not None])
  assert ks == fs
  vs = [dats[k] for k in ks]
  vs = np.stack(vs, 0).reshape((len(ks), -1))

  # CELL SIZES
  #sizes = generate_cell_sizes(fs, path + '/Data/EncodedSizes.pkl')
  sizes = pickle.load(open(DATA_ROOT + '/Data/EncodedSizes.pkl', 'rb'))
  ss = [sizes[k] for k in ks]

  ###########################################  
  step_displacement_histogram(vs, list(trajs.values()))
  ###########################################  
#  pca = PCA(n_components=0.5)
#  dats_ = pca.fit_transform(vs)
#  with open('./%s_PCA.pkl' % feat, 'wb') as f:
#    pickle.dump(dats_, f)
  length = 5
  n_clusters = 3
  dats_ = pickle.load(open('./%s_PCA.pkl' % feat, 'rb'))
  
  clean_trajs = select_clean_trajecteories(dats_, trajs)
  
  traj_classes = Kmean_on_short_trajs(dats_, trajs, length=length, n_clusters=n_clusters)
  
  
  representative_trajs = {}
  try:
    os.mkdir('%s_clustered_traj_diffs' % feat)
  except:
    pass
  traj_names = list(traj_classes.keys())
  np.random.shuffle(traj_names)
  for t in traj_names:
    if np.unique(traj_classes[t]).shape[0] == 1:
      cl = str(traj_classes[t][0])
    elif np.unique(traj_classes[t]).shape[0] == 2:
      if np.unique(traj_classes[t][:5]).shape[0] == 1 and \
         np.unique(traj_classes[t][-5:]).shape[0] == 1 and \
         traj_classes[t][0] != traj_classes[t][-1]:
        cl = str(traj_classes[t][0]) + '_' + str(traj_classes[t][-1])
      else:
        continue
    else:
      continue
    if not cl in representative_trajs:
      try:
        os.mkdir('./%s_clustered_traj_diffs/%s' % (feat, cl))
      except:
        pass
      representative_trajs[cl] = []
    representative_trajs[cl].append(t)
    if len(representative_trajs[cl]) < 50:
      save_traj(t, output_path='./%s_clustered_traj_diffs/%s/%s.gif' % (feat, cl, t[:9] + '_' + t[10:])) 
  
  ##############################################
  color_range = [np.array((0., 0., 1., 0.5)), 
                 np.array((1., 0., 0., 0.5))]
  range_min = np.log(min(ss))
  range_max = np.log(max(ss))
  colors = [(np.log(s) - range_min)/(range_max - range_min) * color_range[0] + \
            (range_max - np.log(s))/(range_max - range_min) * color_range[1] for s in ss]
    
  
  plt.clf() 
  plt.scatter(dats_[:, 0], dats_[:, 1], c=colors, s=0.1)
  plt.legend()
  plt.xlabel("PC1")
  plt.ylabel("PC2")
  plt.savefig('/home/michaelwu/pca_%s.png' % feat, dpi=300)
  
  
  
  single_patch_classes = -np.ones((len(dats_),), dtype=int)
  for cl in range(n_clusters):
    trajs_cl = representative_trajs[str(cl)]
    for t in trajs_cl:
      for ind in trajs[t]:
        single_patch_classes[ind] = cl
  cmap = cm.get_cmap('tab10')
  colors = list(cmap.colors[:(n_clusters+1)])
  colors[-1] = (0.8, 0.8, 0.8)

  unplotted = np.where(single_patch_classes < 0)[0]
  plotted = np.where(single_patch_classes >= 0)[0]
  plt.clf()
  
  plt.scatter(dats_[np.where(single_patch_classes==0)[0][0], 1], 
              dats_[np.where(single_patch_classes==0)[0][0], 2], c=colors[0], s=1., label='Cluster 0')
  plt.scatter(dats_[np.where(single_patch_classes==1)[0][0], 1], 
              dats_[np.where(single_patch_classes==1)[0][0], 2], c=colors[1], s=1., label='Cluster 1')
  plt.scatter(dats_[np.where(single_patch_classes==2)[0][0], 1], 
              dats_[np.where(single_patch_classes==2)[0][0], 2], c=colors[2], s=1., label='Cluster 2')
  plt.scatter(dats_[unplotted][:, 0], dats_[unplotted][:, 1], c=(0.8, 0.8, 0.8), s=0.1)
  plt.scatter(dats_[plotted][:, 0], dats_[plotted][:, 1], c=[colors[i] for i in single_patch_classes[plotted]], s=0.1)

  plt.legend()
  plt.xlabel("PC1")
  plt.ylabel("PC2")
  plt.savefig('/home/michaelwu/pca_%s2.png' % feat, dpi=300)

#  range0 = [np.quantile(dats_[:, 0], 0.02), np.quantile(dats_[:, 0], 0.98)]
#  range1 = [np.quantile(dats_[:, 1], 0.02), np.quantile(dats_[:, 1], 0.98)]
#  range2 = [np.quantile(dats_[:, 2], 0.02), np.quantile(dats_[:, 2], 0.98)]
#  range0 = [range0[0] - (range0[1] - range0[0]) * 0.2, range0[1] + (range0[1] - range0[0]) * 0.2]
#  range1 = [range1[0] - (range1[1] - range1[0]) * 0.2, range1[1] + (range1[1] - range1[0]) * 0.2]
#  range2 = [range2[0] - (range2[1] - range2[0]) * 0.2, range2[1] + (range2[1] - range2[0]) * 0.2]
#  plt.xlim(range0)
#  plt.ylim(range1)



