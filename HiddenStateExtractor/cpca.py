import numpy as np
import pickle
import contrastive
from HiddenStateExtractor.naive_imagenet import read_file_path, DATA_ROOT
import matplotlib
from matplotlib import cm
matplotlib.use('AGG')
import matplotlib.pyplot as plt

dat = pickle.load(open('./save_0005_before.pkl', 'rb'))
fs = sorted(pickle.load(open('./HiddenStateExtractor/file_paths_bkp.pkl', 'rb')))
trajs = pickle.load(open('./HiddenStateExtractor/trajectory_in_inds.pkl', 'rb'))
sizes = pickle.load(open(DATA_ROOT + '/Data/EncodedSizes.pkl', 'rb'))
densities = pickle.load(open(DATA_ROOT + '/Data/EncodedDensities.pkl', 'rb'))



sample_in_traj = []
sample_in_traj_dat = []
for t in trajs:
  patches = trajs[t]
  for i in patches:
    sample_in_traj.append(fs[i])
    sample_in_traj_dat.append(dat[fs[i]])



_sample_in_traj = set(sample_in_traj)
sample_not_in_traj = []
sample_not_in_traj_dat = []
for k in dat:
  if not k in _sample_in_traj:
    sample_not_in_traj.append(k)
    sample_not_in_traj_dat.append(dat[k])


sample_in_traj_dat = np.concatenate(sample_in_traj_dat, 0).reshape((len(sample_in_traj), -1))
sample_not_in_traj_dat = np.concatenate(sample_not_in_traj_dat, 0).reshape((len(sample_not_in_traj), -1))

mdl = contrastive.CPCA()
projected_data, alphas = mdl.fit_transform(sample_in_traj_dat, sample_not_in_traj_dat, return_alphas=True)



ss = [sizes[f][0] for f in sample_in_traj]
ds = [densities[f][0][1] for f in sample_in_traj]

dats_ = projected_data[2]
#cmap = matplotlib.cm.get_cmap('BuPu')  
#range_min = np.log(min(ss))
#range_max = np.log(max(ss))
#colors = [cmap(((np.log(s) - range_min)/(range_max - range_min))**1.5) for s in ss]
plt.clf()
sns.set_style('white')
fig, ax = plt.subplots()
ax.scatter(dats_[:, 0], dats_[:, 1], s=0.5, edgecolors='none')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.savefig('/home/michaelwu/cpca2.png', dpi=300)


diffs = []
for t in trajs:
  cpca_traj = []
  for i in trajs[t]:
    cpca_traj.append(dats_[sample_in_traj.index(fs[i])])
  cpca_traj = np.stack(cpca_traj)
  traj_diff = np.linalg.norm(cpca_traj[1:] - cpca_traj[:-1], ord=2, axis=1)
  diffs.append(traj_diff)


base_diff = np.linalg.norm(dats_[1:] - dats_[0:1], ord=2, axis=1)
plt.hist(base_diff, bins=np.arange(0, 10, 0.1))
plt.hist(np.concatenate(diffs), bins=np.arange(0, 10, 0.1))
plt.savefig('/home/michaelwu/diff_hist.png', dpi=300)