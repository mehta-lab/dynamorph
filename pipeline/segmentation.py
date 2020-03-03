# bchhun, {2020-02-21}


# pipeline:
# 1. check input: (n_frames * 2048 * 2048 * 2) channel 0 - phase, channel 1 - retardance
# 2. adjust channel range
#     a. phase: 32767 plus/minus 1600~2000
#     b. retardance: 1400~1600 plus/minus 1500~1800
# 3. save as '$SITE_NAME.npy' numpy array, dtype=uint16
# 4. run segmentation using saved model: `/data/michaelwu/CellVAE/NNSegmentation/temp_save_unsaturated/final.h5`
# 5. run instance segmentation
# 6. save individual cell patches
# 7. connect individual cells into trajectories
# 8. collect patches and assemble for VAE encoding
# 9. PCA of VAE encoded latent vectors

import os
from NNsegmentation.models import Segment
from NNsegmentation.data import predict_whole_map
from keras import backend as K
import tensorflow as tf

from SingleCellPatch.extract_patches import process_site_instance_segmentation


# 4
def segmentation(paths):
    """
    # loads: 'NNsegmentation/temp_save_unsaturated/final.h5', 'site.npy' (pre=generated using preprocess.py)
    # generates '_NNProbabilities.npy', '
    #           .png',
    #           '_NNpred.png',
    #            '%s-supps' FOLDER

    # prints: "Predicting %d" % t
    :param paths:
    :return:
    """
    temp_folder, supp_folder, target, sites = paths[0], paths[1], paths[2], paths[3]

    for site in sites:
        site_path = os.path.join(temp_folder+'/'+site+'.npy')

        site_supp_files_folder = os.path.join(supp_folder, '%s-supps' % site[:2], '%s' % site)

        if not os.path.exists(site_supp_files_folder):
            os.makedirs(site_supp_files_folder)

        # Generate semantic segmentation
        model = Segment(input_shape=(256, 256, 2),
                        unet_feat=32,
                        fc_layers=[64, 32],
                        n_classes=3)
        model.load('NNsegmentation/temp_save_unsaturated/final.h5')
        predict_whole_map(site_path, model, n_classes=3, batch_size=8, n_supp=5)


# 5
def instance_segmentation(paths):
    """
    # loads
    # generates 'site-supps/cell_positions.pkl',
    #           'site-supps/cell_pixel_assignments.pkl',
    #           'site-supps/segmentation_%d.png'
    #           'site-supps/segmentation_%d.png'

    # prints 'Clustering time %d' % timepoint
    :param paths:
    :return:
    """
    temp_folder, supp_folder, target, sites = paths[0], paths[1], paths[2], paths[3]

    for site in sites:
        site_path = os.path.join(temp_folder + '/' + site + '.npy')

        site_segmentation_path = os.path.join(temp_folder, '%s_NNProbabilities.npy' % site)
        site_supp_files_folder = os.path.join(supp_folder, '%s-supps' % site[:2], '%s' % site)

        process_site_instance_segmentation(site_path, site_segmentation_path, site_supp_files_folder)

