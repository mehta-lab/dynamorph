# bchhun, {2020-02-21}

import os
from NNsegmentation.models import Segment
from NNsegmentation.data import predict_whole_map
from keras import backend as K
import tensorflow as tf
from SingleCellPatch.extract_patches import process_site_instance_segmentation


def segmentation(paths):
    """ Wrapper method for semantic segmentation

    This function loads a saved model at:
        "NNsegmentation/temp_save_unsaturated/final.h5"
    and performs predicion on all specified sites included in the input paths.
    
    Resulting segmentation results and sample segentation image will be saved.

    Args:
        paths (list): list of paths, containing:
            0 - folder for raw data and segmentation results (in .npy)
            1 - folder for supplementary data
            2 - deprecated
            3 - list of site names

    """
        
    temp_folder, supp_folder, target, sites = paths[0], paths[1], paths[2], paths[3]
    model = Segment(input_shape=(256, 256, 2),
                    unet_feat=32,
                    fc_layers=[64, 32],
                    n_classes=3)
    # model.load('NNsegmentation/temp_save_unsaturated/final.h5')
    model.load(target)

    for site in sites:
        site_path = os.path.join(temp_folder, '%s.npy' % site)
        if not os.path.exists(site_path):
            print("Site not found %s" % site_path, flush=True)
        else:
            print("Predicting %s" % site_path, flush=True)
        try:
            # Generate semantic segmentation
            predict_whole_map(site_path, model, n_classes=3, batch_size=8, n_supp=5)
        except Exception as e:
            print("Error in predicting site %s" % site, flush=True)


def instance_segmentation(paths):
    """ Helper function for instance segmentation

    Wrapper method `process_site_instance_segmentation` will be called, which
    loads "*_NNProbabilities.npy" files and performs instance segmentation.

    Results will be saved in supplementary data folder, including:
        "cell_positions.pkl": list of cells in each frame (IDs and positions);
        "cell_pixel_assignments.pkl": pixel compositions of cells;
        "segmentation_*.png": image of instance segmentation results.

    Args:
        paths (list): list of paths, containing:
            0 - folder for raw data and segmentation results (in .npy)
            1 - folder for supplementary data
            2 - deprecated
            3 - list of site names

    """

    temp_folder, supp_folder, target, sites = paths[0], paths[1], paths[2], paths[3]

    for site in sites:
        site_path = os.path.join(temp_folder, '%s.npy' % site)
        site_segmentation_path = os.path.join(temp_folder, '%s_NNProbabilities.npy' % site)
        if not os.path.exists(site_path) or not os.path.exists(site_segmentation_path):
            print("Site not found %s" % site_path, flush=True)
        else:
            print("Clustering %s" % site_path, flush=True)

        site_supp_files_folder = os.path.join(supp_folder, '%s-supps' % site[:2], '%s' % site)
        if not os.path.exists(site_supp_files_folder):
            os.makedirs(site_supp_files_folder)

        process_site_instance_segmentation(site_path, site_segmentation_path, site_supp_files_folder)
