import os
import numpy as np
import pickle
from sklearn.decomposition import PCA


def fit_PCA(train_data):
    """ Fit a PCA (50% variance) on train_data

    Args:
        train_data (np.array): array of training data,
            should be directly extracted from VAE latent space

    Returns:
        sklearn PCA model: trained PCA instance
        np.array: transformed (top PCs of) training data

    """
    pca = PCA(0.5)
    dat_transformed = pca.fit_transform(train_data)
    return pca, dat_transformed


if __name__ == '__main__':
    # input_path = '/gpfs/CompMicro/projects/dynamorph/microglia/JUNE_data_processed'
    # dats = pickle.load(open(os.input_path.join(input_path, 'D_latent_space.pkl'), 'rb'))
    path = '/CompMicro/projects/cardiomyocytes/200721_CM_Mock_SPS_Fluor/20200721_CM_Mock_SPS/dnm_input'
    dats = pickle.load(open(os.path.join(path, 'im_latent_space_after.pkl'), 'rb'))
    pca_model, dats_transformed = fit_PCA(dats)
    with open(os.path.join(path, 'pca_save.pkl'), 'wb') as f:
        pickle.dump(pca_model, f)