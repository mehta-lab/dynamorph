import os
import numpy as np
import pickle
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import umap
input_path = '/CompMicro/projects/cardiomyocytes/200721_CM_Mock_SPS_Fluor/20200721_CM_Mock_SPS/dnm_input'
train_path = '/CompMicro/projects/cardiomyocytes/200721_CM_Mock_SPS_Fluor/20200721_CM_Mock_SPS/dnm_train'
dats = pickle.load(open(os.path.join(input_path, 'im_latent_space_after.pkl'), 'rb'))
dats_ = pickle.load(open(os.path.join(input_path, 'im_latent_space_after_PCAed.pkl'), 'rb'))
# dats = pickle.load(open(os.path.join(input_path, 'im_latent_space.pkl'), 'rb'))
# dats_ = pickle.load(open(os.path.join(input_path, 'im_latent_space_PCAed.pkl'), 'rb'))
print('data shape:', dats.shape)
print('PCA data shape:', dats_.shape)
print('data:', dats[:21, :11])
#%%
plt.clf()
# sns.set_style('white')
fig, ax = plt.subplots()
ax.scatter(dats_[:, 0], dats_[:, 1], s=1, edgecolors='none')
plt.savefig(os.path.join(train_path, 'PCA.png'), dpi=300)
#%%
# a_s = [1.58, 1, 1, 0.5]
# b_s = [0.9, 0.9, 1.5, 1.5]
a_s = [1.58]
b_s = [0.9]
n_rows = 1
n_cols = 1
# xlim = [-7, 7]
# # ylim = [-7, 7]
fig, ax = plt.subplots(n_rows, n_cols, squeeze=False)
ax = ax.flatten()
fig.set_size_inches((5 * n_cols, 5 * n_rows))
axis_count = 0
zoom = True
# top and bottom % of data to cut off
zoom_cutoff = 10
for a, b in zip(a_s, b_s):
    reducer = umap.UMAP(a=a, b=b)
    embedding = reducer.fit_transform(dats)
    with open(os.path.join(train_path, 'umap.pkl'), 'wb') as f:
        pickle.dump(embedding, f)
    ax[axis_count].scatter(embedding[:, 0], embedding[:, 1], s=2, edgecolors='none')
    # ax[axis_count].set_title('a={}, b={}'.format(a, b), fontsize=12)
    if zoom:
        xlim = [np.percentile(embedding[:, 0], zoom_cutoff), np.percentile(embedding[:, 0], 100 - zoom_cutoff)]
        ylim = [np.percentile(embedding[:, 1], zoom_cutoff), np.percentile(embedding[:, 1], 100 - zoom_cutoff)]
        ax[axis_count].set_xlim(left=xlim[0], right=xlim[1])
        ax[axis_count].set_ylim(bottom=ylim[0], top=ylim[1])
    ax[axis_count].set_xlabel('UMAP 1')
    ax[axis_count].set_ylabel('UMAP 2')
    axis_count += 1
fig.savefig(os.path.join(train_path, 'UMAP.png'),
            dpi=300, bbox_inches='tight')
plt.close(fig)