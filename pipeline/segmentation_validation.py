import pickle
import numpy as np
import os
# import tifffile
import imageio


def find_rim(cell_positions):
  masks = set(tuple(r) for r in cell_positions)
  inner_masks = set((r[0]-1, r[1]) for r in masks) & \
                set((r[0]+1, r[1]) for r in masks) & \
                set((r[0], r[1]-1) for r in masks) & \
                set((r[0], r[1]+1) for r in masks)
  edge_positions = np.array(list(masks - inner_masks))
  return edge_positions


def segmentation_validation(paths):

  temp_folder, supp_folder, target, sites = paths[0], paths[1], paths[2], paths[3]

  for site in sites:
    raw_input_stack = os.path.join(temp_folder + '/' + site + '.npy')

    NN_predictions_stack = np.load(temp_folder + '/%s_NNProbabilities.npy' % site)
    cell_pixels = pickle.load(open(supp_folder + f"/{site[0]}-supps/{site}/cell_pixel_assignments.pkl", 'rb'))

    full_output_stack = os.path.join(target + '/')

    stack = []
    for t_point in range(len(raw_input_stack)):
      mat = raw_input_stack[t_point, :, :, 0]
      mat = np.stack([mat] * 3, 2)
      positions, inds = cell_pixels[t_point]
      for cell_ind in np.unique(inds):
        if cell_ind < 0:
          continue
        cell_positions = positions[np.where(inds == cell_ind)]
        outer_rim = find_rim(cell_positions)
        mask_identities = NN_predictions_stack[t_point][(cell_positions[:, 0], cell_positions[:, 1])].mean(0)
        if mask_identities[1] > mask_identities[2]:
          c = 'b'
          mat[(outer_rim[:, 0], outer_rim[:, 1])] = np.array([0, 65535, 0]).reshape((1, 3))
        else:
          c = 'r'
          mat[(outer_rim[:, 0], outer_rim[:, 1])] = np.array([65535, 0, 0]).reshape((1, 3))
      stack.append(mat)

    imageio.mimwrite(full_output_stack+site+'.tiff', np.stack(stack, 0))
    # tifffile.imwrite('%s_predictions.tiff' % site, np.stack(stack, 0))