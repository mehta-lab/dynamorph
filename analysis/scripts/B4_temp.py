import pickle
import numpy as np
import pandas as pd
import torch
import cv2
import matplotlib
matplotlib.use('AGG')
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns
from HiddenStateExtractor.vq_vae import VQ_VAE, CHANNEL_MAX, CHANNEL_VAR, CHANNEL_RANGE, prepare_dataset, rescale
from sklearn.decomposition import PCA

sites = ['B4-Site_%d' % i for i in [0, 2, 3, 5, 6]]
dats = pickle.load(open('./save_0005_bkp4.pkl', 'rb'))
fs = pickle.load(open('./HiddenStateExtractor/file_paths_bkp.pkl', 'rb'))
trajs = pickle.load(open('./HiddenStateExtractor/trajectory_in_inds.pkl', 'rb'))

sites = ['B4-Site_%d' % i for i in [0, 2, 3, 5, 6]]
B4_dats = pickle.load(open('./save_0005_bkp4_B4.pkl', 'rb'))
B4_fs = sorted(B4_dats.keys())
B4_dats = np.stack([B4_dats[f] for f in B4_fs], 0).reshape((len(B4_fs), -1))
# B4_trajs = {}
# B4_trajs_nonmg_ratio = {}
# for site in sites:
#   site_trajs = pickle.load(open('../data_temp/B4-supps/%s/cell_trajs.pkl' % site, 'rb'))[0]
#   site_pixel_assignments = pickle.load(open('../data_temp/B4-supps/%s/cell_pixel_assignments.pkl' % site, 'rb'))
#   site_segmentations = np.load('../data_temp/%s_NNProbabilities.npy' % site)
#   for i, t in enumerate(site_trajs):
#     name = '%s/%d' % (site, i)
#     B4_traj_ind = []
#     B4_traj_nonmg_ratio = []
#     for t_point in sorted(t.keys()):
#       a, b = site_pixel_assignments[t_point]
#       cell_id = t[t_point]
#       cell_ps = a[np.where(b == cell_id)]
#       cell_segs = np.stack([site_segmentations[t_point, l[0], l[1]] for l in cell_ps])
#       cell_nonmg_ratio = (cell_segs[:, 2] > cell_segs[:, 1]).sum()/float(len(cell_ps))
#       B4_traj_nonmg_ratio.append(cell_nonmg_ratio)
#       patch_name = '/data/michaelwu/data_temp/B4-supps/%s/%d_%d.png' % (site, t_point, t[t_point])
#       B4_traj_ind.append(B4_fs.index(patch_name))
#     B4_trajs[name] = B4_traj_ind
#     B4_trajs_nonmg_ratio[name] = B4_traj_nonmg_ratio
# valid_ts = []
# for t in B4_trajs_nonmg_ratio:
#   r = np.quantile(B4_trajs_nonmg_ratio[t], 0.9)
#   if r < 0.2:
#     valid_ts.append(t)
# B4_trajs = {t: B4_trajs[t] for t in valid_ts if len(B4_trajs[t]) > 30}
B4_trajs = pickle.load(open('./HiddenStateExtractor/B4_trajectory_in_inds.pkl', 'rb'))

pca = PCA(0.5)
dats_ = pca.fit_transform(dats)
B4_dats_ = pca.transform(B4_dats)

######################################################################
cs = [0, 1]
input_shape = (128, 128)
gpu = False
B4_dataset = torch.load('../data_temp/B4_all_adjusted_static_patches.pt')
B4_dataset = rescale(B4_dataset)
model = VQ_VAE(alpha=0.0005, gpu=gpu)
model.load_state_dict(torch.load('./HiddenStateExtractor/save_0005_bkp4.pt', map_location='cpu'))

sample_fs = ['/data/michaelwu/data_temp/B4-supps/B4-Site_5/35_13.png',
             '/data/michaelwu/data_temp/B4-supps/B4-Site_0/149_82.png',
             '/data/michaelwu/data_temp/B4-supps/B4-Site_2/118_75.png',
             '/data/michaelwu/data_temp/B4-supps/B4-Site_5/151_13.png']

for i, f in enumerate(sample_fs):
  sample_ind = B4_fs.index(f)
  sample = B4_dataset[sample_ind:(sample_ind+1)][0]
  output = model(sample)[0]
  inp = sample.cpu().data.numpy()
  out = output.cpu().data.numpy()
  input_phase = (inp[0, 0] * 65535).astype('uint16')
  output_phase = (out[0, 0] * 65535).astype('uint16')
  input_retardance = (inp[0, 1] * 65535).astype('uint16')
  output_retardance = (out[0, 1] * 65535).astype('uint16')
  cv2.imwrite('/home/michaelwu/supp_fig8_B4_VAE_pair%d_input_phase.png' % i, enhance_contrast(input_phase, 1., -10000)) # Note dataset has been rescaled
  cv2.imwrite('/home/michaelwu/supp_fig8_B4_VAE_pair%d_output_phase.png' % i, enhance_contrast(output_phase, 1., -10000))
  cv2.imwrite('/home/michaelwu/supp_fig8_B4_VAE_pair%d_input_retardance.png' % i, enhance_contrast(input_retardance, 2., 0.))
  cv2.imwrite('/home/michaelwu/supp_fig8_B4_VAE_pair%d_output_retardance.png' % i, enhance_contrast(output_retardance, 2., 0.))

######################################################################

plt.clf()
sns.kdeplot(dats_[:, 0], dats_[:, 1], shade=True, cmap="Blues", n_levels=16)
plt.xlim(-4, 4)
plt.ylim(-3, 5)
plt.savefig('/home/michaelwu/supp_fig8_PC1-2_wt.eps')
plt.savefig('/home/michaelwu/supp_fig8_PC1-2_wt.png', dpi=300)

plt.clf()
sns.kdeplot(B4_dats_[:, 0], B4_dats_[:, 1], shade=True, cmap="Reds", n_levels=16)
plt.xlim(-4, 4)
plt.ylim(-3, 5)
plt.savefig('/home/michaelwu/supp_fig8_PC1-2_sti.eps', dpi=300)
plt.savefig('/home/michaelwu/supp_fig8_PC1-2_sti.png', dpi=300)

######################################################################

# ts_of_I = []
# for t in B4_trajs:
#   t_dats_ = B4_dats_[np.array(B4_trajs[t])]
#   if np.std(t_dats_[:15, 0]) + np.std(t_dats_[:15, 1]) < 1.2 and \
#      np.std(t_dats_[-15:, 0]) + np.std(t_dats_[-15:, 1]) < 1.2 and \
#      np.square(np.mean(t_dats_[-15:, :2], 0) - np.mean(t_dats_[:15, :2], 0)).sum() > 9:
#      ts_of_I.append(t)

# # ['B4-Site_0/2',
# #  'B4-Site_0/18',
# #  'B4-Site_2/212',
# #  'B4-Site_2/258',
# #  'B4-Site_3/12',
# #  'B4-Site_6/10']

# for t in ts_of_I:
#   os.system('cp /data/michaelwu/data_temp/B4-supps/%s/traj_movies/cell_traj_%s.gif /data/michaelwu/temp_B4_sample_%s.gif' % (t.split('/')[0], t.split('/')[1], t.replace('/', '_')))

######################################################################

# Substitute for supp fig 5

traj_PC1_diffs = []
traj_PC2_diffs = []
base_PC1_diffs = []
base_PC2_diffs = []
for t in trajs:
  traj_PC1 = dats_[np.array(trajs[t])][:, 0]
  traj_PC2 = dats_[np.array(trajs[t])][:, 1]
  traj_PC1_diff = np.abs(traj_PC1[1:] - traj_PC1[:-1])
  traj_PC2_diff = np.abs(traj_PC2[1:] - traj_PC2[:-1])
  traj_PC1_diffs.append(traj_PC1_diff)
  traj_PC2_diffs.append(traj_PC2_diff)
  random_PC1 = dats_[np.random.choice(np.arange(dats_.shape[0]), (len(trajs[t]),), replace=False), 0]
  random_PC2 = dats_[np.random.choice(np.arange(dats_.shape[0]), (len(trajs[t]),), replace=False), 1]
  base_PC1_diffs.append(np.abs(random_PC1[1:] - random_PC1[:-1]))
  base_PC2_diffs.append(np.abs(random_PC2[1:] - random_PC2[:-1]))
traj_PC1_diffs = np.concatenate(traj_PC1_diffs)
traj_PC2_diffs = np.concatenate(traj_PC2_diffs)
base_PC1_diffs = np.concatenate(base_PC1_diffs)
base_PC2_diffs = np.concatenate(base_PC2_diffs)

B4_traj_PC1_diffs = []
B4_traj_PC2_diffs = []
for t in B4_trajs:
  traj_PC1 = B4_dats_[np.array(B4_trajs[t])][:, 0]
  traj_PC2 = B4_dats_[np.array(B4_trajs[t])][:, 1]
  traj_PC1_diff = np.abs(traj_PC1[1:] - traj_PC1[:-1])
  traj_PC2_diff = np.abs(traj_PC2[1:] - traj_PC2[:-1])
  B4_traj_PC1_diffs.append(traj_PC1_diff)
  B4_traj_PC2_diffs.append(traj_PC2_diff)
B4_traj_PC1_diffs = np.concatenate(B4_traj_PC1_diffs)
B4_traj_PC2_diffs = np.concatenate(B4_traj_PC2_diffs)

line_orig = np.histogram(traj_PC1_diffs, bins=np.arange(0, 8, 0.2), density=True)
line_B4 = np.histogram(B4_traj_PC1_diffs, bins=np.arange(0, 8, 0.2), density=True)
line_base = np.histogram(base_PC1_diffs, bins=np.arange(0, 8, 0.2), density=True)
plt.clf()
plt.bar(line_orig[1][:-1]+0.1-0.09, line_orig[0], width=0.06, color=cm.get_cmap('Blues')(0.6), label='Original Sites Trajectories')
plt.bar(line_orig[1][:-1]+0.1-0.03, line_B4[0], width=0.06, color=cm.get_cmap('Reds')(0.6), label='B4 Sites Trajectories')
plt.bar(line_orig[1][:-1]+0.1+0.03, line_base[0], width=0.06, color=cm.get_cmap('Greys')(0.5), label='Random Baseline')
plt.legend(fontsize=16)
plt.xlabel('PC1 diff', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.savefig('/home/michaelwu/supp_fig5_distri_PC1.eps')
plt.savefig('/home/michaelwu/supp_fig5_distri_PC1.png', dpi=300)

line_orig = np.histogram(traj_PC2_diffs, bins=np.arange(0, 8, 0.2), density=True)
line_B4 = np.histogram(B4_traj_PC2_diffs, bins=np.arange(0, 8, 0.2), density=True)
line_base = np.histogram(base_PC2_diffs, bins=np.arange(0, 8, 0.2), density=True)
plt.clf()
plt.bar(line_orig[1][:-1]+0.1-0.09, line_orig[0], width=0.06, color=cm.get_cmap('Blues')(0.6), label='Original Sites Trajectories')
plt.bar(line_orig[1][:-1]+0.1-0.03, line_B4[0], width=0.06, color=cm.get_cmap('Reds')(0.6), label='B4 Sites Trajectories')
plt.bar(line_orig[1][:-1]+0.1+0.03, line_base[0], width=0.06, color=cm.get_cmap('Greys')(0.5), label='Random Baseline')
plt.legend(fontsize=16)
plt.xlabel('PC2 diff', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
plt.savefig('/home/michaelwu/supp_fig5_distri_PC2.eps')
plt.savefig('/home/michaelwu/supp_fig5_distri_PC2.png', dpi=300)

######################################################################

B4_trajs_positions = {}
for site in sites:
  site_trajs, site_trajs_positions = pickle.load(open('../data_temp/B4-supps/%s/cell_trajs.pkl' % site, 'rb'))
  for i, t in enumerate(site_trajs):
    name = '%s/%d' % (site, i)
    if not name in B4_trajs:
      continue
    t_positions = site_trajs_positions[i]
    B4_trajs_positions[name] = t_positions

traj_average_moving_distances = {}
traj_PC1 = {}
traj_PC2 = {}
for t in B4_trajs:
  t_keys = sorted(B4_trajs_positions[t].keys())
  dists = []
  for t_point in range(len(t_keys) - 3):
    d = np.linalg.norm(B4_trajs_positions[t][t_keys[t_point+3]] - \
                       B4_trajs_positions[t][t_keys[t_point]], ord=2) #+3 to adjust for experiment settings
    dists.append(d)
  traj_average_moving_distances[t] = np.mean(dists)
  pc1s = [B4_dats_[ind, 0] for ind in B4_trajs[t]]
  pc2s = [B4_dats_[ind, 1] for ind in B4_trajs[t]]
  traj_PC1[t] = np.mean(pc1s)
  traj_PC2[t] = np.mean(pc2s)

t_arrays = sorted(B4_trajs.keys())
df = pd.DataFrame({'PC1': [traj_PC1[t] for t in t_arrays],
                   'PC2': [traj_PC2[t] for t in t_arrays],
                   'dists': [np.log(traj_average_moving_distances[t] * 0.722222) for t in t_arrays]}) #0.72um/h for 1pixel/27min

sns.set_style('white')
bins_y = np.linspace(0.1, 4.3, 20)
bins_x = np.linspace(-4, 4, 20)
plt.clf()
g = sns.JointGrid(x='PC1', y='dists', data=df, ylim=(0.1, 4.3), xlim=(-4, 4))
_ = g.ax_marg_x.hist(df['PC1'], bins=bins_x)
_ = g.ax_marg_y.hist(df['dists'], bins=bins_y, orientation='horizontal')
g.plot_joint(sns.kdeplot, cmap="Blues", shade=True)
y_ticks = np.array([1.5, 3., 6., 12., 24., 48.])
g.ax_joint.set_yticks(np.log(y_ticks))
g.ax_joint.set_yticklabels(y_ticks)
g.set_axis_labels('', '')
plt.savefig('/home/michaelwu/supp_fig8_correlation_kde.eps')
plt.savefig('/home/michaelwu/supp_fig8_correlation_kde.png', dpi=300)

######################################################################

MSD_length = 60
MSD_min_length = 20

traj_ensembles = []
for t in B4_trajs_positions:
  t_init = min(B4_trajs_positions[t].keys())
  t_end = max(B4_trajs_positions[t].keys()) + 1
  for t_start in range(t_init, t_end - MSD_min_length):
    if t_start in B4_trajs_positions[t]:
      s_traj = {(t_now - t_start): B4_trajs_positions[t][t_now] \
          for t_now in range(t_start, t_start + MSD_length) if t_now in B4_trajs_positions[t]}
      traj_ensembles.append(s_traj)

traj_MSDs = {}
traj_MSDs_trimmed = {}
for i in range(MSD_length):
  s_dists = [np.square(t[i] - t[0]).sum() for t in traj_ensembles if i in t]
  traj_MSDs[i] = s_dists
  traj_MSDs_trimmed[i] = scipy.stats.trimboth(s_dists, 0.25)

x = np.arange(3, MSD_length) # Start from 27min to keep consistent
y_bins = np.arange(0.9, 11.7, 0.6)
density_map = np.zeros((MSD_length, len(y_bins) - 1))
y = []
for i in range(3, MSD_length):
  for d in traj_MSDs[i]:
    if d == 0: 
      continue
    ind_bin = ((np.log(d) - y_bins) > 0).sum() - 1
    if ind_bin < density_map.shape[1] and ind_bin >= 0:
      density_map[i][ind_bin] += 1
  y.append((np.log(np.mean(traj_MSDs[i])) - 0.9)/(y_bins[1] - y_bins[0]))
density_map = density_map/density_map.sum(1, keepdims=True)

def forceAspect(ax,aspect=1):
  im = ax.get_images()
  extent =  im[0].get_extent()
  ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

sns.set_style('white')
plt.clf()
fig = plt.figure()
ax = fig.add_subplot(121)
ax.imshow(np.transpose(density_map), cmap='Reds', origin='lower', vmin=0.01, vmax=0.3, alpha=0.5)
ax.plot(x, np.array(y) - 0.5, '.-', c='#ba4748') # -0.5 is the adjustment for imshow
ax.set_xscale('log')
xticks = np.array([0.5, 1, 2, 4, 8])
xticks_positions = xticks / (9/60)
ax.set_xticks(xticks_positions)
ax.set_xticklabels(xticks)
ax.xaxis.set_minor_locator(NullLocator())
yticks = np.array([0.5, 2, 8, 32, 128, 512, 2048])
yticks_positions = (np.log(yticks / (0.325 * 0.325)) - 0.9)/(y_bins[1] - y_bins[0]) - 0.5 # same adjustment for imshow
ax.set_yticks(yticks_positions)
ax.set_yticklabels(yticks)
plt.savefig('/home/michaelwu/supp_fig8_B4_MSD.eps')
plt.savefig('/home/michaelwu/supp_fig8_B4_MSD.png', dpi=300)

X = np.log(np.arange(1, 60))
y_ = [np.mean(traj_MSDs[i]) for i in np.arange(1, 60)]
y_ = np.log(np.array(y_))
est = sm.OLS(y_, sm.add_constant(X)).fit()
print(est.params)
# [5.44530955 1.00921815]

