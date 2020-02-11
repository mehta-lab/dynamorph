import numpy as np
import pickle
import contrastive
from HiddenStateExtractor.naive_imagenet import read_file_path, DATA_ROOT
import matplotlib
from matplotlib import cm
matplotlib.use('AGG')
import matplotlib.pyplot as plt

dats = pickle.load(open('./save_0005_bkp4.pkl', 'rb'))
fs = pickle.load(open('./HiddenStateExtractor/file_paths_bkp.pkl', 'rb'))
trajs = pickle.load(open('./HiddenStateExtractor/trajectory_in_inds.pkl', 'rb'))
site_dat = torch.load('../data_temp/B4_all_adjusted_static_patches.pt')

B4_dats = pickle.load(open('./save_0005_bkp4_B4.pkl', 'rb'))
B4_fs = sorted(B4_dats.keys())
B4_dats = np.stack([B4_dats[f] for f in B4_fs], 0).reshape((len(B4_fs), -1))
B4_trajs = pickle.load(open('./HiddenStateExtractor/B4_trajectory_in_inds.pkl', 'rb'))

mdl = contrastive.CPCA()
projected_data, alphas = mdl.fit_transform(B4_dats, dats, return_alphas=True)


for fold in range(1, 4):
  #os.mkdir('/data/michaelwu/PC_samples/cpca_alpha%d_PC1' % fold)
  #os.mkdir('/data/michaelwu/PC_samples/cpca_alpha%d_PC2' % fold)
  dats_ = projected_data[fold]
  plt.clf()
  fig, ax = plt.subplots()
  ax.scatter(dats_[:, 0], dats_[:, 1], s=0.5, edgecolors='none')
  plt.savefig('/data/michaelwu/PC_samples/cpca_alpha%d.png' % fold, dpi=300)

  names = []
  out_paths = []
  PC1s = dats_[:, 0]
  for i in range(5):
    rang = [np.quantile(PC1s, i * 0.2), np.quantile(PC1s, (i+1) * 0.2)]
    rang_fs = [f for i, f in enumerate(B4_fs) if rang[0] <= PC1s[i] < rang[1]]
    ct = 0
    base = np.zeros((128, 128), dtype=float)
    for j, f in enumerate(rang_fs):
      ind = B4_fs.index(f)
      slic = site_dat[ind][0][0].cpu().numpy().astype('float')
      base = base + slic
      ct += 1
    aver = base/ct
    aver = (aver * 65535).astype('uint16')
    cv2.imwrite('/data/michaelwu/PC_samples/cpca_alpha%d_PC1_fold%d_aver.png' % (fold, i), enhance_contrast(aver, a=2, b=-50000))
    for j, f in enumerate(np.random.choice(rang_fs, (20,), replace=False)):
      names.append(f)
      out_paths.append('/data/michaelwu/PC_samples/cpca_alpha%d_PC1/PC1_%d_%d_sample%d.png' % (fold, i, i+1, j))

  PC2s = dats_[:, 1]
  for i in range(5):
    rang = [np.quantile(PC2s, i * 0.2), np.quantile(PC2s, (i+1) * 0.2)]
    rang_fs = [f for i, f in enumerate(B4_fs) if rang[0] <= PC2s[i] < rang[1]]
    ct = 0
    base = np.zeros((128, 128), dtype=float)
    for j, f in enumerate(rang_fs):
      ind = B4_fs.index(f)
      slic = site_dat[ind][0][0].cpu().numpy().astype('float')
      base = base + slic
      ct += 1
    aver = base/ct
    aver = (aver * 65535).astype('uint16')
    cv2.imwrite('/data/michaelwu/PC_samples/cpca_alpha%d_PC2_fold%d_aver.png' % (fold, i), enhance_contrast(aver, a=2, b=-50000))
    for j, f in enumerate(np.random.choice(rang_fs, (20,), replace=False)):
      names.append(f)
      out_paths.append('/data/michaelwu/PC_samples/cpca_alpha%d_PC2/PC2_%d_%d_sample%d.png' % (fold, i, i+1, j))

  for name, out_path in zip(names, out_paths):
    ind = B4_fs.index(name)
    slic = (site_dat[ind][0][0].cpu().numpy() * 65535).astype('uint16')
    cv2.imwrite(out_path, enhance_contrast(slic, a=2, b=-50000))



# names = []
# out_paths = []
# np.random.seed(122)

# PC1s = dats_[:, 0]
# lower_ = np.quantile(PC1s, 0.2)
# lower_fs = [f for i, f in enumerate(B4_fs) if PC1s[i] < lower_]
# upper_ = np.quantile(PC1s, 0.8)
# upper_fs = [f for i, f in enumerate(B4_fs) if PC1s[i] > upper_]
# for i, f in enumerate(np.random.choice(lower_fs, (50,), replace=False)):
#   names.append(f)
#   out_paths.append('/home/michaelwu/cpca_PC1_lower_sample%d.png' % i)
# for i, f in enumerate(np.random.choice(upper_fs, (50,), replace=False)):
#   names.append(f)
#   out_paths.append('/home/michaelwu/cpca_PC1_upper_sample%d.png' % i)


# PC1_range = (np.quantile(PC1s, 0.4), np.quantile(PC1s, 0.6))
# PC2s = dats_[:, 1]
# lower_ = np.quantile(PC2s, 0.2)
# lower_fs = [f for i, f in enumerate(B4_fs) if PC2s[i] < lower_ and PC1_range[0] < PC1s[i] < PC1_range[1]]
# upper_ = np.quantile(PC2s, 0.8)
# upper_fs = [f for i, f in enumerate(B4_fs) if PC2s[i] > upper_ and PC1_range[0] < PC1s[i] < PC1_range[1]]
# for i, f in enumerate(np.random.choice(lower_fs, (50,), replace=False)):
#   names.append(f)
#   out_paths.append('/home/michaelwu/cpca_PC2_lower_sample%d.png' % i)
# for i, f in enumerate(np.random.choice(upper_fs, (50,), replace=False)):
#   names.append(f)
#   out_paths.append('/home/michaelwu/cpca_PC2_upper_sample%d.png' % i)


# def enhance_contrast(mat, a=1.5, b=-10000):
#   mat2 = cv2.addWeighted(mat, a, mat, 0, b)
#   return mat2


