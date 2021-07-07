import os
import numpy as np
import pickle
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import umap
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
import itertools

def zoom_axis(x, y, ax, zoom_cutoff=1):
    xlim = [np.percentile(x, zoom_cutoff), np.percentile(x, 100 - zoom_cutoff)]
    ylim = [np.percentile(y, zoom_cutoff), np.percentile(y, 100 - zoom_cutoff)]
    ax.set_xlim(left=xlim[0], right=xlim[1])
    ax.set_ylim(bottom=ylim[0], top=ylim[1])

def mp_sample_im_pixels(fn_args, workers):
    """Read and computes statistics of images with multiprocessing

    :param list of tuple fn_args: list with tuples of function arguments
    :param int workers: max number of workers
    :return: list of returned df from get_im_stats
    """

    with ProcessPoolExecutor(workers) as ex:
        # can't use map directly as it works only with single arg functions
        res = ex.map(sample_im_pixels, *zip(*fn_args))
    return list(res)

def grid_sample_pixel_values(im, grid_spacing):
    """Sample pixel values in the input image at the grid. Any incomplete
    grids (remainders of modulus operation) will be ignored.

    :param np.array im: 2D image
    :param int grid_spacing: spacing of the grid
    :return int row_ids: row indices of the grids
    :return int col_ids: column indices of the grids
    :return np.array sample_values: sampled pixel values
    """

    im_shape = im.shape
    assert grid_spacing < im_shape[0], "grid spacing larger than image height"
    assert grid_spacing < im_shape[1], "grid spacing larger than image width"
    # leave out the grid points on the edges
    sample_coords = np.array(list(itertools.product(
        np.arange(grid_spacing, im_shape[0], grid_spacing),
        np.arange(grid_spacing, im_shape[1], grid_spacing))))
    row_ids = sample_coords[:, 0]
    col_ids = sample_coords[:, 1]
    sample_values = im[row_ids, col_ids, :]
    return row_ids, col_ids, sample_values

def sample_im_pixels(im, grid_spacing, meta_row):
    """Read and computes statistics of images

    """

    # im = image_utils.read_image(im_path)
    row_ids, col_ids, sample_values = \
        grid_sample_pixel_values(im, grid_spacing)

    meta_rows = \
        [{**meta_row,
          'row_idx': row_idx,
          'col_idx': col_idx,
          'intensity': sample_value}
          for row_idx, col_idx, sample_value
          in zip(row_ids, col_ids, sample_values)]
    return meta_rows

def ints_meta_generator(
        input_dir,
        order='cztp',
        num_workers=4,
        block_size=256,
        ):
    """
    Generate pixel intensity metadata for estimating image normalization
    parameters during preprocessing step. Pixels are sub-sampled from the image
    following a grid pattern defined by block_size to for efficient estimation of
    median and interquatile range. Grid sampling is preferred over random sampling
    in the case due to the spatial correlation in images.
    Will write found data in ints_meta.csv in input directory.
    Assumed default file naming convention is:
    dir_name
    |
    |- im_c***_z***_t***_p***.png
    |- im_c***_z***_t***_p***.png

    c is channel
    z is slice in stack (z)
    t is time
    p is position (FOV)

    Other naming convention is:
    img_channelname_t***_p***_z***.tif for parse_sms_name

    :param list args:    parsed args containing
        str input_dir:   path to input directory containing images
        str name_parser: Function in aux_utils for parsing indices from file name
        int num_workers: number of workers for multiprocessing
        int block_size: block size for the grid sampling pattern. Default value works
        well for 2048 X 2048 images.
    """
    mp_block_args = []

    # Fill dataframe with rows from image names
    for i in range(len(im_names)):
        kwargs = {"im_name": im_names[i]}
        if name_parser == 'parse_idx_from_name':
            kwargs["order"] = order
        elif name_parser == 'parse_sms_name':
            kwargs["channel_names"] = channel_names
        meta_row = parse_func(**kwargs)
        meta_row['dir_name'] = input_dir
        im_path = os.path.join(input_dir, im_names[i])
        mp_fn_args.append(im_path)
        mp_block_args.append((im_path, block_size, meta_row))

    im_ints_list = mp_sample_im_pixels(mp_block_args, num_workers)
    im_ints_list = list(itertools.chain.from_iterable(im_ints_list))
    ints_meta = pd.DataFrame.from_dict(im_ints_list)

    ints_meta_filename = os.path.join(input_dir, 'ints_meta.csv')
    ints_meta.to_csv(ints_meta_filename, sep=",")
    return ints_meta

def distribution_plot(frames_metadata,
                      y_col,
                      output_path,
                      output_fname):
    my_palette = {'F-actin': 'g', 'nuclei': 'm'}

    fig = plt.figure()
    fig.set_size_inches((18, 9))
    ax = sns.violinplot(x='channel_name', y=y_col,
                        hue='dir_name',
                        bw=0.01,
                        data=frames_metadata, scale='area',
                        linewidth=1, inner='quartile',
                        split=False)
    # ax.set_xticklabels(labels=['retardance',
    #                            'BF',
    #                            'retardance+slow axis+BF'])
    plt.xticks(rotation=25)
    # plt.title(''.join([metric_name, '_']))
    # ax.set_ylim(bottom=0.5, top=1)
    # ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", borderaxespad=0)
    ax.legend(loc="upper left", borderaxespad=0.1)
    # ax.get_legend().remove()
    ax.set_ylabel('Mean intensity')
    plt.savefig(os.path.join(output_path, ''.join([output_fname, '.png'])),
                dpi=300, bbox_inches='tight')



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