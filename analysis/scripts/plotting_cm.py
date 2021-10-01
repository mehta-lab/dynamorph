import os
import numpy as np
import pickle
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import umap

def zoom_axis(x, y, ax, zoom_cutoff=1):
    xlim = [np.percentile(x, zoom_cutoff), np.percentile(x, 100 - zoom_cutoff)]
    ylim = [np.percentile(y, zoom_cutoff), np.percentile(y, 100 - zoom_cutoff)]
    ax.set_xlim(left=xlim[0], right=xlim[1])
    ax.set_ylim(bottom=ylim[0], top=ylim[1])

# train_dirs = ['/CompMicro/projects/cardiomyocytes/200721_CM_Mock_SPS_Fluor/20200721_CM_Mock_SPS/dnm_train',
#               '/CompMicro/projects/cardiomyocytes/20200722CM_LowMOI_SPS_Fluor/20200722 CM_LowMOI_SPS/dnm_train']
# train_dir = '/CompMicro/projects/cardiomyocytes/20200722CM_LowMOI_SPS_Fluor/20200722 CM_LowMOI_SPS/dnm_train'
# input_dirs = ['/CompMicro/projects/cardiomyocytes/200721_CM_Mock_SPS_Fluor/20200721_CM_Mock_SPS/dnm_input_tstack/mock_matching_point2',
#             '/CompMicro/projects/cardiomyocytes/20200722CM_LowMOI_SPS_Fluor/20200722 CM_LowMOI_SPS/dnm_input_tstack/mock_matching_point2']
# input_dirs = ['/CompMicro/projects/cardiomyocytes/200721_CM_Mock_SPS_Fluor/20200721_CM_Mock_SPS/dnm_input_tstack/mock+low_moi_matching_point05',
#             '/CompMicro/projects/cardiomyocytes/20200722CM_LowMOI_SPS_Fluor/20200722 CM_LowMOI_SPS/dnm_input_tstack/mock+low_moi_matching_point05']
input_dirs = ['/CompMicro/projects/cardiomyocytes/200721_CM_Mock_SPS_Fluor/20200721_CM_Mock_SPS/dnm_input_tstack/mock_z32_nh16_nrh16_ne512_cc0.25',
    '/CompMicro/projects/cardiomyocytes/20200722CM_LowMOI_SPS_Fluor/20200722 CM_LowMOI_SPS/dnm_input_tstack/mock_z32_nh16_nrh16_ne512_cc0.25']



dats = []
pcas = []
labels = []
label = 0
for input_dir in input_dirs:
    dat = pickle.load(open(os.path.join(input_dir, 'im_latent_space_after.pkl'), 'rb'))
    pca = pickle.load(open(os.path.join(input_dir, 'im_latent_space_after_PCAed.pkl'), 'rb'))
    # dats = pickle.load(open(os.path.join(input_path, 'im_latent_space.pkl'), 'rb'))
    # dats_ = pickle.load(open(os.path.join(input_path, 'im_latent_space_PCAed.pkl'), 'rb'))
    dats.append(dat)
    pcas.append(pca)
    labels += [label] * dat.shape[0]
    label += 1
dats = np.concatenate(dats, axis=0)
pcas = np.concatenate(pcas, axis=0)
#%%
plt.clf()
zoom_cutoff = 1
conditions = ['mock', 'infected']
fig, ax = plt.subplots()
scatter = ax.scatter(pcas[:, 0], pcas[:, 1], s=7, c=labels, cmap='Paired', alpha=0.1)
scatter.set_facecolor("none")
zoom_axis(pcas[:, 0], pcas[:, 1], ax, zoom_cutoff=zoom_cutoff)
legend1 = ax.legend(handles=scatter.legend_elements()[0],
                            loc="upper right", title="condition", labels=conditions)
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
plt.savefig(os.path.join(input_dir, 'PCA.png'), dpi=300)
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
# top and bottom % of data to cut off
zoom_cutoff = 1
for n_nbr in n_nbrs:
    for a, b in zip(a_s, b_s):
        # embedding, labels = pickle.load(open(os.path.join(input_dir, 'umap_{}_nbr.pkl'.format(n_nbr)), 'rb'))

        reducer = umap.UMAP(a=a, b=b, n_neighbors=n_nbr)
        embedding = reducer.fit_transform(dats)
        with open(os.path.join(input_dir, 'umap_{}_nbr.pkl'.format(n_nbr)), 'wb') as f:
            pickle.dump([embedding, labels], f)

        scatter = ax[axis_count].scatter(embedding[:, 0], embedding[:, 1], s=7, c=labels,
                               facecolors='none', cmap='Paired', alpha=0.1)
        scatter.set_facecolor("none")
        ax[axis_count].set_title('n_neighbors={}'.format(n_nbr), fontsize=12)
        # ax[axis_count].set_title('a={}, b={}'.format(a, b), fontsize=12)
        zoom_axis(embedding[:, 0], embedding[:, 1], ax[axis_count], zoom_cutoff=zoom_cutoff)
        if axis_count == 0:
            legend1 = ax[axis_count].legend(handles=scatter.legend_elements()[0],
                            loc="upper right", title="condition", labels=conditions)
        ax[axis_count].set_xlabel('UMAP 1')
        ax[axis_count].set_ylabel('UMAP 2')

        axis_count += 1
        fig.savefig(os.path.join(input_dir, 'UMAP.png'),
                    dpi=300, bbox_inches='tight')
plt.close(fig)