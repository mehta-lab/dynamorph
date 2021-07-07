# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 14:22:51 2019

@author: michael.wu
"""
import numpy as np
import scipy
import os
# import ot
# import tifffile
import pandas as pd
import warnings
import imageio
import pickle
import h5py
import matplotlib

matplotlib.use('AGG')
import matplotlib.pyplot as plt
import numpy as np
import cv2
from ast import literal_eval
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def frame_matching(f1, f2, int1, int2, dist_cutoff=100, int_eff=1.4):
    """ Matching cells between two frames (LAP)

    Args:
        f1 (list of np.array): cell centroid positions in frame 1
        f2 (list of np.array): cell centroid positions in frame 2
        int1 (list of int): cell sizes in frame 1
        int2 (list of int): cell sizes in frame 2
        dist_cutoff (int, optional): default=100, cutoff threshold, any pairs 
            with larger distance are excluded
        int_eff (float, optional): default=1.4, intensity contribution on cost

    Returns:
        list of tuple: matched cell pairs
        dict: top 5 pairs with highest cost (for quality control)

    """
    f1 = np.array(f1).reshape((-1, 2))
    f2 = np.array(f2).reshape((-1, 2))
    int1 = np.array(int1).reshape((-1, 1))
    int2 = np.array(int2).reshape((-1, 1))

    int_dist_mat = int2.reshape((1, -1)) / int1.reshape((-1, 1))
    int_dist_mat = int_dist_mat + 1 / int_dist_mat
    int_dist_mat[np.where(int_dist_mat >= 2.5)] = 20.
    int_dist_mat = int_dist_mat ** int_eff
    int_dist_baseline = np.percentile(int_dist_mat, 10)

    cost_mat = np.ones((len(f1) + len(f2), len(f1) + len(f2))) * (dist_cutoff ** 2 * 10) * int_dist_baseline
    dist_mat = cdist(f1, f2) ** 2
    dist_mat[np.where(dist_mat >= (dist_cutoff ** 2))] = (dist_cutoff ** 2 * 10)
    cost_mat[:len(f1), :len(f2)] = dist_mat * int_dist_mat

    # Cost of no match placeholder
    for i in range(len(f1)):
        cost_mat[i, i + len(f2)] = 1.05 * (dist_cutoff ** 2) * int_dist_baseline
    for j in range(len(f2)):
        cost_mat[len(f1) + j, j] = 1.05 * (dist_cutoff ** 2) * int_dist_baseline

    cost_mat[len(f1):, len(f2):] = np.transpose(dist_mat)
    links = linear_sum_assignment(cost_mat)
    pairs = []
    costs = []
    for pair in zip(*links):
        if pair[0] < len(f1) and pair[1] < len(f2):
            pairs.append(pair)
            costs.append(cost_mat[pair[0], pair[1]])
    return pairs, {pairs[i]: costs[i] for i in np.argsort(costs)[-5:]}


# def frame_matching_ot(f1, f2, int1, int2, dist_cutoff=50):
#     """ Matching cells between two frames (optimal transport)
#
#     Args:
#         f1 (list of np.array): cell centroid positions in frame 1
#         f2 (list of np.array): cell centroid positions in frame 2
#         int1 (list of int): cell sizes in frame 1
#         int2 (list of int): cell sizes in frame 2
#         dist_cutoff (int, optional): default=100, cutoff threshold, any pairs 
#             with larger distance are excluded
#
#     Returns:
#         np.array: optimal transport matrix
#
#     """
#     dist_mat = cdist(f1, f2) ** 2
#     dist_mat[np.where(dist_mat>=(dist_cutoff**2))] = (dist_cutoff ** 2 * 10)
#     int1 = np.array(int1)/np.sum(int1)
#     int2 = np.array(int2)/np.sum(int2)
#     ot_mat = ot.sinkhorn(int1, int2, dist_mat, 10.)
#     return ot_mat


def trajectory_connection(trajectories,
                          trajectories_positions,
                          intensities_dict,
                          dist_cutoff=100,
                          only_gap=True):
    """ Model gap, split and merge

    Implementation of methods in:
        "Robust single-particle tracking in live-cell time-lapse sequences"

    Args:
        trajectories (list): list of trajectories (dict of t_point: cell_id)
        trajectories_positions (list): list of trajectory positions (dict of 
            t_point: cell center position)
        intensities_dict (dict): dict of cell sizes in each frame
        dist_cutoff (int, optional): default=100, cutoff threshold, any pairs 
            with larger distance are excluded
        only_gap (bool, optional): if True, only model trajectory gaps
            currently only support gaps
            TODO: add in trajectory merge/split

    Returns:
        list of dict: list of trajectories (dict of t_point: cell ID)

    """
    starts = [min(t.keys()) for t in trajectories_positions]
    ends = [max(t.keys()) for t in trajectories_positions]

    cost_mat_d1 = len(trajectories_positions)
    if not only_gap:
        cost_mat_d2 = sum(len(t) for t in trajectories_positions)
    cost_mat_d3 = len(trajectories_positions)

    # Upper left: gap connection
    cost_mat_upper_left = np.ones((cost_mat_d1, cost_mat_d1)) * (dist_cutoff ** 2 * 10)
    positions_x = [trajectories_positions[i][end] for i, end in enumerate(ends)]
    positions_y = [trajectories_positions[j][start] for j, start in enumerate(starts)]
    dist_mat = cdist(positions_x, positions_y) ** 2

    mask_mat = ((np.array(starts).reshape((1, -1)) - np.array(ends).reshape((-1, 1))) == 2) * 1 + \
               ((np.array(starts).reshape((1, -1)) - np.array(ends).reshape(
                   (-1, 1))) == 3) * 4  # Allow gap up to 3 time intervals
    mask_mat[np.where(dist_mat >= (dist_cutoff ** 2))] = 0
    cost_mat_upper_left = mask_mat * dist_mat + (1 - np.sign(mask_mat)) * cost_mat_upper_left

    if not only_gap:
        # Upper middle: merge
        cost_mat_upper_middle = np.ones((cost_mat_d1, cost_mat_d2)) * (dist_cutoff ** 2 * 10)
        positions_x = [trajectories_positions[i][end] for i, end in enumerate(ends)]
        positions_y = [[t[t_point] for t_point in sorted(t.keys())] for t in trajectories_positions]
        positions_y = np.concatenate(positions_y)
        dist_mat = cdist(positions_x, positions_y) ** 2

        strength_x = np.array([intensities_dict[end][trajectories[i][end]] for i, end in enumerate(ends)]).reshape(
            (-1, 1)).astype(float)
        strength_y = [[intensities_dict[t_point][t[t_point]] for t_point in sorted(t.keys())] for t in trajectories]
        strength_y = np.concatenate(strength_y).reshape((1, -1)).astype(float)
        strength_y_tminus1 = [
            [np.max(strength_y) * 2] + [intensities_dict[t_point][t[t_point]] for t_point in sorted(t.keys())[:-1]] for
            t in trajectories]
        strength_y_tminus1 = np.concatenate(strength_y_tminus1).reshape((1, -1)).astype(float)
        ratio_mat = np.clip(strength_y / (strength_x + strength_y_tminus1), 0.3, 3.)
        ratio_mat = (ratio_mat > 1.) * ratio_mat + (ratio_mat <= 1.) * (ratio_mat ** -2)

        join_points = np.concatenate([sorted(t.keys()) for t in trajectories])
        mask_mat = ((np.array(join_points).reshape((1, -1)) - np.array(ends).reshape((-1, 1))) == 1) * 1
        mask_mat[np.where((dist_mat * ratio_mat) >= (dist_cutoff ** 2))] = 0
        cost_mat_upper_middle = mask_mat * dist_mat * ratio_mat + (1 - np.sign(mask_mat)) * cost_mat_upper_middle

        # Middle left: split
        cost_mat_middle_left = np.ones((cost_mat_d2, cost_mat_d1)) * (dist_cutoff ** 2 * 10)
        positions_x = [[t[t_point] for t_point in sorted(t.keys())] for t in trajectories_positions]
        positions_x = np.concatenate(positions_x)
        positions_y = [trajectories_positions[j][start] for j, start in enumerate(starts)]
        dist_mat = cdist(positions_x, positions_y) ** 2

        strength_x = [[intensities_dict[t_point][t[t_point]] for t_point in sorted(t.keys())] for t in trajectories]
        strength_x = np.concatenate(strength_x).reshape((-1, 1)).astype(float)
        strength_x_tplus1 = [[intensities_dict[t_point][t[t_point]] for t_point in sorted(t.keys())[1:]] + [0.] for t in
                             trajectories]
        strength_x_tplus1 = np.concatenate(strength_x_tplus1).reshape((-1, 1)).astype(float)
        strength_y = np.array(
            [intensities_dict[start][trajectories[j][start]] for j, start in enumerate(starts)]).reshape(
            (1, -1)).astype(float)
        ratio_mat = np.clip(strength_x / (strength_x_tplus1 + strength_y), 0.3, 3.)
        ratio_mat = (ratio_mat > 1.) * ratio_mat + (ratio_mat <= 1.) * (ratio_mat ** -2)

        join_points = np.concatenate([sorted(t.keys()) for t in trajectories])
        mask_mat = ((np.array(starts).reshape((1, -1)) - np.array(join_points).reshape((-1, 1))) == 1) * 1
        mask_mat[np.where((dist_mat * ratio_mat) >= (dist_cutoff ** 2))] = 0
        cost_mat_middle_left = mask_mat * dist_mat * ratio_mat + (1 - np.sign(mask_mat)) * cost_mat_middle_left

        # Middle right: split alternative
        cost_mat_middle_right = np.ones((cost_mat_d2, cost_mat_d3)) * (dist_cutoff ** 2 * 10)
        begin_ind = 0
        for i, t in enumerate(trajectories_positions):
            aver_movement = np.array([t[t_point] for t_point in sorted(t.keys())])
            aver_movement = ((aver_movement[1:] - aver_movement[:-1]) ** 2).sum(1).mean()

            t_inds = trajectories[i]
            strength_x = [intensities_dict[t_point][t_inds[t_point]] for t_point in sorted(t_inds.keys())]
            strength_x_tplus1 = strength_x[1:] + [strength_x[-1]]
            ratio = np.array(strength_x).astype(float) / np.array(strength_x_tplus1).astype(float)
            ratio = (ratio > 1) * ratio + (ratio <= 1) * (ratio ** -2)

            filling = np.clip(aver_movement * ratio, 0, (dist_cutoff ** 2 * 10))
            cost_mat_middle_right[begin_ind:(begin_ind + len(t)), i] = filling
            begin_ind += len(t)

        # Lower middle: merge alternative
        cost_mat_lower_middle = np.ones((cost_mat_d3, cost_mat_d2)) * (dist_cutoff ** 2 * 10)
        begin_ind = 0
        for i, t in enumerate(trajectories_positions):
            aver_movement = np.array([t[t_point] for t_point in sorted(t.keys())])
            aver_movement = ((aver_movement[1:] - aver_movement[:-1]) ** 2).sum(1).mean()

            t_inds = trajectories[i]
            strength_x = [intensities_dict[t_point][t_inds[t_point]] for t_point in sorted(t_inds.keys())]
            strength_x_tminus1 = [strength_x[0]] + strength_x[:-1]

            ratio = np.array(strength_x).astype(float) / np.array(strength_x_tminus1).astype(float)
            ratio = (ratio > 1) * ratio + (ratio <= 1) * (ratio ** -2)

            filling = np.clip(aver_movement * ratio, 0, (dist_cutoff ** 2 * 10))
            cost_mat_lower_middle[i, begin_ind:(begin_ind + len(t))] = filling
            begin_ind += len(t)

        cost_mat_middle_middle = np.ones((cost_mat_d2, cost_mat_d2)) * (dist_cutoff ** 2 * 10)

    # Upper right: normal termination
    cost_mat_upper_right = np.ones((cost_mat_d1, cost_mat_d3)) * (dist_cutoff ** 2 * 10)
    if only_gap:
        diagonal_assignment = cost_mat_upper_left[np.where(cost_mat_upper_left < np.max(cost_mat_upper_left))]
    else:
        diagonal_assignment = np.concatenate(
            [cost_mat_upper_left[np.where(cost_mat_upper_left < np.max(cost_mat_upper_left))],
             cost_mat_upper_middle[np.where(cost_mat_upper_middle < np.max(cost_mat_upper_middle))]])
    if len(diagonal_assignment) > 0:
        diagonal_assignment = np.percentile(diagonal_assignment, 90)
    else:
        # In cases when upper left mat is all invalid
        diagonal_assignment = np.max(cost_mat_upper_left) * 0.9
    np.fill_diagonal(cost_mat_upper_right, diagonal_assignment)

    # Lower left: normal initialization
    cost_mat_lower_left = np.ones((cost_mat_d3, cost_mat_d1)) * (dist_cutoff ** 2 * 10)
    if only_gap:
        diagonal_assignment = cost_mat_upper_left[np.where(cost_mat_upper_left < np.max(cost_mat_upper_left))]
    else:
        diagonal_assignment = np.concatenate(
            [cost_mat_upper_left[np.where(cost_mat_upper_left < np.max(cost_mat_upper_left))],
             cost_mat_middle_left[np.where(cost_mat_middle_left < np.max(cost_mat_middle_left))]])
    if len(diagonal_assignment) > 0:
        diagonal_assignment = np.percentile(diagonal_assignment, 90)
    else:
        # In cases when upper left mat is all invalid
        diagonal_assignment = np.max(cost_mat_upper_left) * 0.9
    np.fill_diagonal(cost_mat_lower_left, diagonal_assignment)

    # Lower right: transpose of upper left
    cost_mat_lower_right = np.transpose(cost_mat_upper_left)

    if only_gap:
        cost_mat = np.concatenate(
            [np.concatenate([cost_mat_upper_left, cost_mat_upper_right], 1),
             np.concatenate([cost_mat_lower_left, cost_mat_lower_right], 1)], 0)
        links = linear_sum_assignment(cost_mat)

        connection_maps = {}
        for pair in zip(*links):
            if pair[0] < cost_mat_d1 and pair[1] < cost_mat_d1:
                assert pair[1] > pair[0]
                connection_maps[pair[0]] = pair[1]

        connected = []
        involved = set()
        for i in range(len(trajectories)):
            if i in involved:
                continue
            else:
                con = [i]
                involved.add(i)
                start = i
                while i in connection_maps:
                    con.append(connection_maps[i])
                    involved.add(connection_maps[i])
                    i = connection_maps[i]
                connected.append(con)
        new_trajectories = []
        for con in connected:
            t = trajectories[con[0]]
            for c in con[1:]:
                t.update(trajectories[c])
            new_trajectories.append(t)
        return new_trajectories
    else:
        cost_mat = np.concatenate(
            [np.concatenate([cost_mat_upper_left, cost_mat_upper_middle, cost_mat_upper_right], 1),
             np.concatenate([cost_mat_middle_left, cost_mat_middle_middle, cost_mat_middle_right], 1),
             np.concatenate([cost_mat_lower_left, cost_mat_lower_middle, cost_mat_lower_right], 1)], 0)
        # TODO: finish the merge/split process


def generate_trajectories(matchings, df_meta, intensities_dict, min_len=11, track_dim='time', traj_id_start=0):
    """ Generate trajectories based on frame-to-frame matchings

    Args:
        matchings (dict): dict of frame-to-frame matching
        positions_dict (dict): dict of cell positions in each frame
        intensities_dict (dict): dict of cell sizes in each frame

    Returns:
        list of dict: list of trajectories (dict of idx: cell ID)
        list of dict: list of trajectory positions (dict of idx: cell
              center position)
    
    """
    # Initial pass of trajectory connection, connect based on `matchings`
    trajectories = []
    traj_id_col = ' '.join([track_dim, 'trajectory ID'])
    for idx in sorted(matchings.keys()):
        pairs = matchings[idx]
        for cell_id1, cell_id2 in pairs:
            for traj in trajectories:
                if idx in traj and traj[idx] == cell_id1:
                    traj[idx + 1] = cell_id2
                    break
            else:
                trajectories.append({idx: cell_id1, idx + 1: cell_id2})
    trajectories_positions = [{idx: df_meta[(df_meta[track_dim] == idx) &
                                            (df_meta['cell ID'] == traj[idx])]['cell position'].iat[0]
                               for idx in traj} for traj in
                              trajectories]
    # Second pass: connect gap, account for cell merging/splitting (TODO)
    trajectories = trajectory_connection(trajectories, trajectories_positions, intensities_dict, dist_cutoff=100.,
                                         only_gap=True)
    # Only select long trajectories
    # TODO: expose the minimal trajectory length in the config
    trajectories = [t for t in trajectories if len(t) >= min_len]
    trajectories_positions = [{idx: df_meta[(df_meta[track_dim] == idx) &
                                            (df_meta['cell ID'] == traj[idx])]['cell position']
                               for idx in traj} for traj in
                              trajectories]
    #TODO: rewrite double for loop assignment with pandas merge
    #df_traj = pd.DataFrame.from_dict(trajectories)
    traj_id = traj_id_start
    for traj in trajectories:
        for idx in traj:
            df_meta.loc[(df_meta[track_dim] == idx) & (df_meta['cell ID'] == traj[idx]), traj_id_col] = traj_id
        traj_id += 1

    return trajectories, trajectories_positions, df_meta, traj_id


def process_site_build_trajectory(site_supp_files_folder, min_len=11, track_dim='slice'):
    """ Wrapper method for assembling and building trajectories

    This function requires supplementary data generated by 
    `process_site_instance_segmentation`. Trajectories will be generated by 
    matching cells in adjacent frames and connect pairs of cells through time.

    Results will be saved to the supplementary data folder as:
        "cell_traj.pkl": list of trajectories and list of trajectory positions
            trajectories are dict of track_dim_idx: cell ID
            trajectory positions are dict of track_dim_idx: cell center position

    Args:
        site_supp_files_folder (str): path to the folder where supplementary 
            files will be saved

    """
    print(f"\tloading cell_positions {os.path.join(site_supp_files_folder, 'cell_positions.pkl')}")
    cell_positions = pickle.load(open(os.path.join(site_supp_files_folder, 'cell_positions.pkl'), 'rb'))

    print(f"\tloading cell pixel assignments {os.path.join(site_supp_files_folder, 'cell_pixel_assignments.pkl')}")
    cell_pixel_assignments = pickle.load(open(os.path.join(site_supp_files_folder, 'cell_pixel_assignments.pkl'), 'rb'))
    meta_path = os.path.join(site_supp_files_folder, 'patch_meta.csv')
    # Cell position is an numpy array of [row col] and need a separate parser
    df_meta = pd.read_csv(meta_path, index_col=0, converters={
        'cell position': lambda x: np.fromstring(x.strip("[]"), sep=' ', dtype=np.int32)})
    # dimension to fix while tracking along the track dimension
    trackable_dim = ('time', 'slice')
    fix_dim = [dim for dim in trackable_dim if not dim == track_dim][0]
    track_dim_ids = df_meta[track_dim].unique()
    fix_dim_ids = df_meta[fix_dim].unique()
    track_dim_ids.sort()
    fix_dim_ids.sort()
    # track_dim_ids = sorted(cell_positions.keys())
    assert np.allclose(np.array(track_dim_ids)[1:] - 1, np.array(track_dim_ids)[:-1])

    # Mapping to centroid positions
    cell_positions_dict = {k: dict(cell_positions[k]) for k in cell_positions}

    # track along time dimension
    # Generate Frame-frame matching
    cell_matchings = {}
    # TODO: save top cost pairs
    pairs_to_be_checked = {}
    # Initiate trajectory ID column. Assign nan to unmatched cells/patches
    traj_id_col = ' '.join([track_dim, 'trajectory ID'])
    df_meta[traj_id_col] = np.nan
    traj_id_start = 0
    for fix_dim_idx in fix_dim_ids:
        try:
            df_meta_sub = df_meta.copy().loc[df_meta[fix_dim] == fix_dim_idx, :]
            cell_size_dict = {}
            for track_dim_idx in track_dim_ids[:-1]:
                # Mapping to size of segmentation
                # positions, positions_labels = cell_pixel_assignments[track_dim_idx][fix_dim_idx]
                # all_cells = cell_positions[track_dim_idx][fix_dim_idx]
                # cell_size_d = dict(zip(*np.unique(positions_labels, return_counts=True)))
                # cell_size_d = {id: cell_size_d[id] for id, position in all_cells}
                cell_size_d = df_meta_sub.loc[df_meta_sub[track_dim] == track_dim_idx,
                                              ['cell ID', 'cell size']].set_index('cell ID')['cell size'].to_dict()
                cell_size_dict[track_dim_idx] = cell_size_d
                ids1 = df_meta_sub[df_meta_sub[track_dim] == track_dim_idx]['cell ID'].tolist()
                ids2 = df_meta_sub[df_meta_sub[track_dim] == (track_dim_idx + 1)]['cell ID'].tolist()
                pos1 = df_meta_sub[df_meta_sub[track_dim] == track_dim_idx]['cell position'].tolist()
                pos2 = df_meta_sub[df_meta_sub[track_dim] == (track_dim_idx + 1)]['cell position'].tolist()
                size1 = df_meta_sub[df_meta_sub[track_dim] == track_dim_idx]['cell size'].tolist()
                size2 = df_meta_sub[df_meta_sub[track_dim] == (track_dim_idx + 1)]['cell size'].tolist()
                if len(ids1) == 0 or len(ids2) == 0:
                    cell_matchings[track_dim_idx] = []
                else:
                    pairs, top_cost_pairs = frame_matching(pos1, pos2, size1, size2, dist_cutoff=100)
                    for p in top_cost_pairs:
                        pairs_to_be_checked[
                            ('%d_%d' % (track_dim_idx, ids1[p[0]]), '%d_%d' % (track_dim_idx + 1, ids2[p[1]]))] = \
                        top_cost_pairs[p]
                    cell_matchings[track_dim_idx] = [(ids1[p1], ids2[p2]) for p1, p2 in pairs]
            # Connect to trajectories
            cell_trajectories, cell_trajectories_positions, df_meta_sub, traj_id_start = \
                generate_trajectories(cell_matchings, df_meta_sub, cell_size_dict, min_len=min_len, track_dim=track_dim, traj_id_start=traj_id_start)
            # Update metadata
            df_meta.loc[df_meta[fix_dim] == fix_dim_idx, :] = df_meta_sub
        except Exception as e:
            cell_trajectories = cell_trajectories_positions = []
            warnings.warn('No trajectory is generated due to the following error: {}'.format(e))
        output_path = os.path.join(site_supp_files_folder, 'cell_traj_{}{}.pkl'.format(fix_dim ,fix_dim_idx))
        with open(output_path, 'wb') as f:
            print(f"\tsaving cell_traj {output_path}")
            pickle.dump([cell_trajectories, cell_trajectories_positions], f)
    df_meta.to_csv(meta_path, sep=',')
    # elif track_dim == 'slice':
    #     for track_dim_idx in track_dim_ids:
    #         # try:
    #         df_meta_sub = df_meta[df_meta['time'] == track_dim_idx]
    #         cell_size_dict = {}
    #         for fix_dim_idx in fix_dim_ids[:-1]:
    #             # Mapping to size of segmentation
    #             positions, positions_labels = cell_pixel_assignments[track_dim_idx][fix_dim_idx]
    #             all_cells = cell_positions[track_dim_idx][fix_dim_idx]
    #             cell_size_d = dict(zip(*np.unique(positions_labels, return_counts=True)))
    #             cell_size_d = {id: cell_size_d[id] for id, position in all_cells}
    #             cell_size_dict[fix_dim_idx] = cell_size_d
    #             # ids1 = sorted(cell_positions_dict[track_dim_idx].keys())
    #             # ids2 = sorted(cell_positions_dict[track_dim_idx+1].keys())
    #             # f1 = [cell_positions_dict[track_dim_idx][i] for i in ids1]
    #             # f2 = [cell_positions_dict[track_dim_idx+1][i] for i in ids2]
    #             # int1 = [cell_size_dict[track_dim_idx][i] for i in ids1]
    #             # int2 = [cell_size_dict[track_dim_idx+1][i] for i in ids2]
    #
    #             ids1 = df_meta_sub[df_meta_sub['slice'] == fix_dim_idx]['cell ID'].tolist()
    #             ids2 = df_meta_sub[df_meta_sub['slice'] == (fix_dim_idx + 1)]['cell ID'].tolist()
    #             pos1 = df_meta_sub[df_meta_sub['slice'] == fix_dim_idx]['cell position'].tolist()
    #             pos2 = df_meta_sub[df_meta_sub['slice'] == (fix_dim_idx + 1)]['cell position'].tolist()
    #             size1 = df_meta_sub[df_meta_sub['slice'] == fix_dim_idx]['cell size'].tolist()
    #             size2 = df_meta_sub[df_meta_sub['slice'] == (fix_dim_idx + 1)]['cell size'].tolist()
    #             if len(ids1) == 0 or len(ids2) == 0:
    #                 cell_matchings[fix_dim_idx] = []
    #             else:
    #                 pairs, top_cost_pairs = frame_matching(pos1, pos2, size1, size2, dist_cutoff=100)
    #                 for p in top_cost_pairs:
    #                     pairs_to_be_checked[('%d_%d' % (fix_dim_idx, ids1[p[0]]), '%d_%d' % (fix_dim_idx + 1, ids2[p[1]]))] = \
    #                     top_cost_pairs[p]
    #                 cell_matchings[fix_dim_idx] = [(ids1[p1], ids2[p2]) for p1, p2 in pairs]
    #         # Connect to trajectories
    #         cell_trajectories, cell_trajectories_positions, df_meta_sub = \
    #             generate_trajectories(cell_matchings, df_meta_sub, cell_size_dict, min_len=min_len, track_dim=track_dim)
    #         # except Exception as e:
    #         #     cell_trajectories = cell_trajectories_positions = []
    #         #     warnings.warn('No trajectory is generated due to the following error: {}'.format(e))
    #
    #         with open(os.path.join(site_supp_files_folder, 'cell_traj_t{}.pkl'.format(track_dim_idx)), 'wb') as f:
    #             print(f"\tsaving cell_traj {os.path.join(site_supp_files_folder, 'cell_traj_t{}.pkl'.format(track_dim_idx))}")
    #             pickle.dump([cell_trajectories, cell_trajectories_positions], f)
    # else:
    #     raise ValueError('Wrong track_dim value')
    return


def process_well_generate_trajectory_relations(df_meta,
                                               track_dim='slice',
                                               **kwargs):
    """ Find pair relations (adjacent frame, same trajectory) in static patches
    Results will be saved under `raw_folder`

    used by run_vae.py -m "assemble" to generate relations used by matching loss in vae training

    "relations" is a dictionary of (key, value) = ((patch_id, patch_id), 2),
        where 2 is the value for "adjacent frame, same trajectory"
        where 1 is the value for "non-adjacent frame, same trajectory"
        and where 0 is the value for "different trajectory"

    Args:
        fs (list of str): all individual cell patch names
        sites (list of str): sites (from the same well)
        well_supp_files_folder (str): path to save supplementary data

    Returns:
        relations, labels:
    """

    # assert len(set(s[:2] for s in sites)) == 1  # Sites should all come from the same well
    traj_id_col = ' '.join([track_dim, 'trajectory ID'])
    labels = df_meta[traj_id_col].to_numpy()
    # assign labels to orphan patches
    labels[np.isnan(labels)] = np.arange(np.nanmax(labels) + 1, np.nanmax(labels) + 1 + np.isnan(labels).sum())
    traj_ids = df_meta[traj_id_col].unique()
    traj_ids.sort()
    # relations = {}

    # def patch_name_to_tuple(f):
    #     f = [seg for seg in f.split('/') if len(seg) > 0]
    #     site_name = f[-2]
    #     assert site_name in sites
    #     t_point = int(f[-1].split('_')[0])
    #     cell_id = int(f[-1].split('_')[1].split('.')[0])
    #     return (site_name, t_point, cell_id)
    #
    # patch_id_mapping = {patch_name_to_tuple(f): i for i, f in enumerate(fs)}

    # initialize label vector for patches


    # set diagonal relation
    relations = {(patch_id, patch_id): 2 for patch_id in df_meta.index.values.astype(int)}

    # for site in sites:
    #     print('site:', site)
    #
    #     # trajectories is a list of dict, with each dict (key, value) = (timepoint, cell_id)
    #     trajectories = pickle.load(open(os.path.join(well_supp_files_folder,
    #                                                  site,
    #                                                  "cell_traj.pkl"), 'rb'))[0]
    for traj_id in traj_ids:
        df_traj = df_meta.loc[df_meta[traj_id_col] == traj_id, :]
        track_dim_ids = sorted(df_traj[track_dim].to_list())
        patch_ids = []
        for track_dim_idx in track_dim_ids:
            # get reference patch ID
            ref_patch_id = df_traj[df_traj[track_dim] == track_dim_idx].index.values[0]
            patch_ids.append(ref_patch_id)
            if track_dim_idx + 1 in track_dim_ids:
                adj_patch_id = df_traj[df_traj[track_dim] == track_dim_idx + 1].index.values[0]
                relations[(ref_patch_id, adj_patch_id)] = 2
                relations[(adj_patch_id, ref_patch_id)] = 2

        # Same trajectory
        for i in patch_ids:
            for j in patch_ids:
                if not (i, j) in relations:
                    relations[(i, j)] = 1
    return relations, labels


def save_traj_bbox(trajectory, trajectory_positions, image_stack, path):
    """ Save trajectory as gif

    Trajectory will be saved as a gif of the full field of view, target cell
    will be marked with a red bounding box

    Args:
        trajectory (dict): dict of t_point: cell ID
        trajectory_positions (dict): dict of t_point: cell center position
        image_stack (np.array): stack of raw data
        path (str): path for saving images

    """
    output_images = np.zeros((len(trajectory), 512, 512))
    for i, k in enumerate(sorted(trajectory.keys())):
        output_images[i] = cv2.resize(image_stack[k, :, :, 0], (512, 512))
    output_images = np.stack([output_images] * 3, 3)

    output_images = output_images / 65535.

    for i, k in enumerate(sorted(trajectory.keys())):
        box_center = trajectory_positions[k] / (2048 / 512)
        box_range = [(max(box_center[0] - 16., 0), min(box_center[0] + 16., 512)),
                     (max(box_center[1] - 16., 0), min(box_center[1] + 16., 512))]

        # Left edge
        x = box_range[0][0]
        x_ = (int(max(x - 1., 0)), int(min(x + 1., 512)))
        output_images[i, x_[0]:x_[1], int(box_range[1][0]):int(box_range[1][1])] = np.array([1., 0., 0.]).reshape(
            (1, 1, 3))
        # Right edge
        x = box_range[0][1]
        x_ = (int(max(x - 1., 0)), int(min(x + 1., 512)))
        output_images[i, x_[0]:x_[1], int(box_range[1][0]):int(box_range[1][1])] = np.array([1., 0., 0.]).reshape(
            (1, 1, 3))
        # Top edge
        y = box_range[1][0]
        y_ = (int(max(y - 1., 0)), int(min(y + 1., 512)))
        output_images[i, int(box_range[0][0]):int(box_range[0][1]), y_[0]:y_[1]] = np.array([1., 0., 0.]).reshape(
            (1, 1, 3))
        # Bottom edge
        y = box_range[1][1]
        y_ = (int(max(y - 1., 0)), int(min(y + 1., 512)))
        output_images[i, int(box_range[0][0]):int(box_range[0][1]), y_[0]:y_[1]] = np.array([1., 0., 0.]).reshape(
            (1, 1, 3))
    # tifffile.imwrite(path, (output_images*255).astype('uint8'))
    imageio.mimwrite(path, (output_images * 255).astype('uint8'))
    return
