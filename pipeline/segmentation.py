# bchhun, {2020-02-21}

import os
import pickle

import matplotlib
import numpy as np
matplotlib.use('AGG')
import matplotlib.pyplot as plt
from NNsegmentation.models import Segment
from NNsegmentation.data import predict_whole_map
from keras import backend as K
import tensorflow as tf
from SingleCellPatch.extract_patches import check_segmentation_dim
from sklearn.cluster import DBSCAN


def segmentation(paths):
    """ Wrapper method for semantic segmentation

    This method performs predicion on all specified sites included in the input paths.

    Model weight path should be provided, if not a default path will be used:
        UNet: "NNsegmentation/temp_save_unsaturated/final.h5"
    
    Resulting segmentation results and sample segentation image will be saved
    in the summary folder.

    Args:
        paths (list): list of paths, containing:
            0 - folder for raw data, segmentation and summarized results
            1 - folder for supplementary data
            2 - path to model weight (not used in this method)
            3 - list of site names

    """
        
    summary_folder, supp_folder, model_path, sites = paths[0], paths[1], paths[2], paths[3]
    model = Segment(input_shape=(256, 256, 2),
                    unet_feat=32,
                    fc_layers=[64, 32],
                    n_classes=3)

    try:
        if not model_path is None:
            model.load(model_path)
        else:
            model.load('NNsegmentation/temp_save_unsaturated/final.h5')
    except Exception as ex:
        print(ex)
        raise ValueError("Error in loading UNet weights")

    for site in sites:
        site_path = os.path.join(summary_folder, '%s.npy' % site)
        if not os.path.exists(site_path):
            print("Site not found %s" % site_path, flush=True)
        else:
            print("Predicting %s" % site_path, flush=True)
            try:
                # Generate semantic segmentation
                predict_whole_map(site_path, model, n_classes=3, batch_size=8, n_supp=5)
            except Exception as e:
                print("Error in predicting site %s" % site, flush=True)
    return

def instance_segmentation(paths):
    """ Helper function for instance segmentation

    Wrapper method `process_site_instance_segmentation` will be called, which
    loads "*_NNProbabilities.npy" files and performs instance segmentation.

    Results will be saved in the supplementary data folder, including:
        "cell_positions.pkl": list of cells in each frame (IDs and positions);
        "cell_pixel_assignments.pkl": pixel compositions of cells;
        "segmentation_*.png": image of instance segmentation results.

    Args:
        paths (list): list of paths, containing:
            0 - folder for raw data, segmentation and summarized results
            1 - folder for supplementary data
            2 - path to model weight (not used in this method)
            3 - list of site names

    """

    summary_folder, supp_folder, model_path, sites = paths[0], paths[1], paths[2], paths[3]

    for site in sites:
        site_path = os.path.join(summary_folder, '%s.npy' % site)
        site_segmentation_path = os.path.join(summary_folder, '%s_NNProbabilities.npy' % site)
        if not os.path.exists(site_path) or not os.path.exists(site_segmentation_path):
            print("Site not found %s" % site_path, flush=True)
        else:
            print("Clustering %s" % site_path, flush=True)
            site_supp_files_folder = os.path.join(supp_folder, '%s-supps' % site[:2], '%s' % site)
            if not os.path.exists(site_supp_files_folder):
                os.makedirs(site_supp_files_folder)
            process_site_instance_segmentation(site_path, site_segmentation_path, site_supp_files_folder)
    return


def instance_clustering(cell_segmentation,
                        ct_thr=(500, 12000),
                        instance_map=True,
                        map_path=None,
                        fg_thr=0.3,
                        DBSCAN_thr=(10, 250)):
    """ Perform instance clustering on a static frame

    Args:
        cell_segmentation (np.array): segmentation mask for the frame
        ct_thr (tuple, optional): lower and upper threshold for cell size (number
            of pixels in segmentation mask)
        instance_map (bool, optional): if to save instance segmentation as an
            image
        map_path (str or None, optional): path to the image (if `instance_map`
            is True)
        fg_thr (float, optional): threshold of foreground, any pixel with
            predicted background prob less than this value would be regarded as
            foreground (MG or Non-MG)
        DBSCAN_thr (tuple, optional): parameters for DBSCAN, (eps, min_samples)

    Returns:
        (list * 3): 3 lists (MG, Non-MG, intermediate) of cell identifiers
            each entry in the list is a tuple of cell ID and cell center position
        np.array: array of x, y coordinates of foreground pixels
        np.array: array of cell IDs of foreground pixels

    """
    cell_segmentation = check_segmentation_dim(cell_segmentation)
    all_cells = cell_segmentation[:, :, 0] < fg_thr
    positions = np.array(list(zip(*np.where(all_cells))))
    if len(positions) < 1000:
        # No cell detected
        return ([], [], []), np.zeros((0, 2), dtype=int), np.zeros((0,), dtype=int)

    # DBSCAN clustering of cell pixels
    clustering = DBSCAN(eps=DBSCAN_thr[0], min_samples=DBSCAN_thr[1]).fit(positions)
    positions_labels = clustering.labels_
    cell_ids, point_cts = np.unique(positions_labels, return_counts=True)

    mg_cell_positions = []
    non_mg_cell_positions = []
    other_cells = []
    for cell_id, ct in zip(cell_ids, point_cts):
        if cell_id < 0:
            # neglect unclustered pixels
            continue
        if ct <= ct_thr[0] or ct >= ct_thr[1]:
            # neglect cells that are too small/big
            continue
        points = positions[np.where(positions_labels == cell_id)[0]]
        # calculate cell center
        mean_pos = np.mean(points, 0).astype(int)
        ## TODO: remove hardcoded window size
        ## define window
        # window = [(mean_pos[0]-128, mean_pos[0]+128), (mean_pos[1]-128, mean_pos[1]+128)]
        # skip if cell has too many outlying points
        # outliers = [p for p in points if not within_range(window, p)]
        # if len(outliers) > len(points) * 0.05:
        #     continue
        cell_segmentation_labels = cell_segmentation[points[:, 0], points[:, 1]]
        # Calculate if MG/Non-MG/intermediate
        mg_ratio = (np.argmax(cell_segmentation_labels, 1) == 1).sum()/len(points)
        non_mg_ratio = (np.argmax(cell_segmentation_labels, 1) == 2).sum()/len(points)
        if mg_ratio > 0.9:
            mg_cell_positions.append((cell_id, mean_pos))
        elif non_mg_ratio > 0.9:
            non_mg_cell_positions.append((cell_id, mean_pos))
        else:
            other_cells.append((cell_id, mean_pos))

    # Save instance segmentation results as image
    if instance_map and map_path is not None:
        # bg as -1
        segmented = np.zeros(cell_segmentation.shape[:2]) - 1
        for cell_id, mean_pos in mg_cell_positions:
            points = positions[np.where(positions_labels == cell_id)[0]]
            for p in points:
                segmented[p[0], p[1]] = cell_id%10
        for cell_id, mean_pos in non_mg_cell_positions:
            points = positions[np.where(positions_labels == cell_id)[0]]
            for p in points:
                segmented[p[0], p[1]] = cell_id%10
        plt.clf()
        cmap = matplotlib.cm.get_cmap('tab10')
        cmap.set_under(color='k')
        plt.imshow(segmented, cmap=cmap, vmin=-0.001, vmax=10.001)
        # MG will be marked with white text, Non-MG with red text
        font_mg = {'color': 'white', 'size': 4}
        font_non_mg = {'color': 'red', 'size': 4}
        for cell_id, mean_pos in mg_cell_positions:
            plt.text(mean_pos[1], mean_pos[0], str(cell_id), fontdict=font_mg)
        for cell_id, mean_pos in non_mg_cell_positions:
            plt.text(mean_pos[1], mean_pos[0], str(cell_id), fontdict=font_non_mg)
        plt.axis('off')
        plt.savefig(map_path, dpi=300)
    return (mg_cell_positions, non_mg_cell_positions, other_cells), positions, positions_labels


def process_site_instance_segmentation(site_path,
                                       site_segmentation_path,
                                       site_supp_files_folder):
    """ Wrapper method for instance segmentation

    Results will be saved to the supplementary data folder as:
        "cell_positions.pkl": list of cells in each frame (IDs and positions);
        "cell_pixel_assignments.pkl": pixel compositions of cells;
        "segmentation_*.png": image of instance segmentation results.

    Args:
        site_path (str): path to image stack (.npy)
        site_segmentation_path (str): path to semantic segmentation stack (.npy)
        site_supp_files_folder (str): path to the folder where supplementary
            files will be saved

    """

    # TODO: Size is hardcoded here
    # Should be of size (n_time_points, 2048, 2048, 2), uint16
    image_stack = np.load(site_path)
    # Should be of size (n_time_points, 2048, 2048, n_classes), float
    segmentation_stack = np.load(site_segmentation_path)

    cell_positions = {}
    cell_pixel_assignments = {}
    for t_point in range(image_stack.shape[0]):
        print("\tClustering time %d" % t_point)
        cell_segmentation = segmentation_stack[t_point]
        instance_map_path = os.path.join(site_supp_files_folder, 'segmentation_%d.png' % t_point)
        res = instance_clustering(
            cell_segmentation, instance_map=True, map_path=instance_map_path, ct_thr=[500, np.inf])
        cell_positions[t_point] = res[0] # MG, Non-MG, Chimeric Cells
        cell_pixel_assignments[t_point] = res[1:]
    with open(os.path.join(site_supp_files_folder, 'cell_positions.pkl'), 'wb') as f:
        pickle.dump(cell_positions, f)
    with open(os.path.join(site_supp_files_folder, 'cell_pixel_assignments.pkl'), 'wb') as f:
        pickle.dump(cell_pixel_assignments, f)
    return