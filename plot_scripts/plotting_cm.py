import os
import numpy as np
import pickle
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import umap

# train_dirs = ['/CompMicro/projects/cardiomyocytes/200721_CM_Mock_SPS_Fluor/20200721_CM_Mock_SPS/dnm_train',
#               '/CompMicro/projects/cardiomyocytes/20200722CM_LowMOI_SPS_Fluor/20200722 CM_LowMOI_SPS/dnm_train']
# train_dir = '/CompMicro/projects/cardiomyocytes/20200722CM_LowMOI_SPS_Fluor/20200722 CM_LowMOI_SPS/dnm_train'
output_dirs = ['/CompMicro/projects/cardiomyocytes/200721_CM_Mock_SPS_Fluor/20200721_CM_Mock_SPS/dnm_input/model_mock+low_moi',
            '/CompMicro/projects/cardiomyocytes/20200722CM_LowMOI_SPS_Fluor/20200722 CM_LowMOI_SPS/dnm_input/model_mock+low_moi']
dats = []
pcas = []
labels = []
label = 0
for output_dir in output_dirs:
    dat = pickle.load(open(os.path.join(output_dir, 'im_latent_space_after.pkl'), 'rb'))
    pca = pickle.load(open(os.path.join(output_dir, 'im_latent_space_after_PCAed.pkl'), 'rb'))
    # dats = pickle.load(open(os.path.join(input_path, 'im_latent_space.pkl'), 'rb'))
    # dats_ = pickle.load(open(os.path.join(input_path, 'im_latent_space_PCAed.pkl'), 'rb'))
    dats.append(dat)
    pcas.append(pca)
    labels += [label] * dat.shape[0]
    label += 1
    print('data shape:', dat.shape)
    print('PCA data shape:', pca.shape)
dats = np.concatenate(dats, axis=0)
pcas = np.concatenate(pcas, axis=0)
print('data shape:', dats.shape)
print('PCA data shape:', pcas.shape)
    # print('data:', dats[:21, :11])
#%%
plt.clf()
# sns.set_style('white')
fig, ax = plt.subplots()
ax.scatter(pcas[:, 0], pcas[:, 1], s=1, c=labels, edgecolors='none', cmap='tab10')
plt.savefig(os.path.join(output_dir, 'PCA.png'), dpi=300)
#%%
# a_s = [1.58, 1, 1, 0.5]
# b_s = [0.9, 0.9, 1.5, 1.5]
a_s = [1.58]
b_s = [0.9]
n_nbrs = [15, 50, 200, 1000]
n_rows = 2
n_cols = 2
# xlim = [-7, 7]
# # ylim = [-7, 7]
fig, ax = plt.subplots(n_rows, n_cols, squeeze=False)
ax = ax.flatten()
fig.set_size_inches((5 * n_cols, 5 * n_rows))
axis_count = 0
zoom = True
# top and bottom % of data to cut off
zoom_cutoff = 1
for n_nbr in n_nbrs:
    for a, b in zip(a_s, b_s):
        reducer = umap.UMAP(a=a, b=b, n_neighbors=n_nbr)
        embedding = reducer.fit_transform(dats)
        with open(os.path.join(output_dir, 'umap_{}_nbr.pkl'.format(n_nbr)), 'wb') as f:
            pickle.dump([embedding, labels], f)
        ax[axis_count].scatter(embedding[:, 0], embedding[:, 1], s=1, c=labels, edgecolors='none', cmap='tab10')
        ax[axis_count].set_title('n_neighbors={}'.format(n_nbr), fontsize=12)
        # ax[axis_count].set_title('a={}, b={}'.format(a, b), fontsize=12)
        if zoom:
            xlim = [np.percentile(embedding[:, 0], zoom_cutoff), np.percentile(embedding[:, 0], 100 - zoom_cutoff)]
            ylim = [np.percentile(embedding[:, 1], zoom_cutoff), np.percentile(embedding[:, 1], 100 - zoom_cutoff)]
            ax[axis_count].set_xlim(left=xlim[0], right=xlim[1])
            ax[axis_count].set_ylim(bottom=ylim[0], top=ylim[1])
        ax[axis_count].set_xlabel('UMAP 1')
        ax[axis_count].set_ylabel('UMAP 2')
        axis_count += 1
fig.savefig(os.path.join(output_dir, 'UMAP.png'),
            dpi=300, bbox_inches='tight')
plt.close(fig)